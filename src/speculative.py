"""Heterogeneous Speculative Decoding Coordinator for Chimera-Inference Engine.

Coordinates a fast draft proposer (Prompt Lookup / shallow draft engine) with a
full-scale target verifier (NVIDIA RTX 3050 CUDA / Intel P-core AVX2 engine).
Implements mathematically lossless speculative verification (Leviathan et al. 2023).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    HAS_TORCH = False

from src.generator import ChimeraGenerator
from src.sampling import SamplingParams


@dataclass
class SpeculativeStats:
    """Metrics tracking for speculative decoding throughput and acceptance."""

    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    target_eval_calls: int = 0
    total_generated_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Percentage of proposed draft tokens accepted by the verifier."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def tokens_per_target_step(self) -> float:
        """Effective speedup: tokens generated per target forward verification step."""
        if self.target_eval_calls == 0:
            return 0.0
        return self.total_generated_tokens / self.target_eval_calls


class DraftProposer(ABC):
    """Abstract base class for speculative draft proposer."""

    @abstractmethod
    def propose(
        self,
        prefix_tokens: list[int],
        gamma: int,
    ) -> tuple[list[int], list[Any] | None]:
        """Propose `gamma` speculative draft tokens and optional draft probabilities."""


class PromptLookupDraftProposer(DraftProposer):
    """Zero-parameter, zero-FLOP SOTA Prompt Lookup Draft Proposer.

    Matches n-grams in the existing prompt and generated sequence to propose candidate
    continuations. Highly effective for code generation, JSON structures, chat formatting,
    and repeated phrases without consuming GPU VRAM or CPU FLOPs.
    """

    def __init__(self, min_ngram_size: int = 2, max_ngram_size: int = 4) -> None:
        self.min_ngram_size = min_ngram_size
        self.max_ngram_size = max_ngram_size

    def propose(
        self,
        prefix_tokens: list[int],
        gamma: int,
    ) -> tuple[list[int], list[Any] | None]:
        if len(prefix_tokens) < self.min_ngram_size:
            return [], None

        seq_len = len(prefix_tokens)
        best_match_idx = -1
        best_match_len = 0

        # Try matching longest possible n-gram from the end of prefix
        max_k = min(self.max_ngram_size, seq_len - 1)
        for k in range(max_k, self.min_ngram_size - 1, -1):
            pattern = prefix_tokens[-k:]
            # Search pattern in preceding tokens prefix_tokens[:-k]
            # Search from recent history towards beginning
            sub_len = len(prefix_tokens) - k
            for i in range(sub_len - k, -1, -1):
                if prefix_tokens[i : i + k] == pattern:
                    best_match_idx = i + k
                    best_match_len = k
                    break
            if best_match_len > 0:
                break

        if best_match_idx == -1:
            return [], None

        # Extract up to gamma tokens following the match
        draft = prefix_tokens[best_match_idx : best_match_idx + gamma]
        return draft, None


class ModelDraftProposer(DraftProposer):
    """Draft proposer utilizing a fast lightweight ChimeraGenerator."""

    def __init__(self, draft_generator: ChimeraGenerator) -> None:
        self.draft_generator = draft_generator

    def propose(
        self,
        prefix_tokens: list[int],
        gamma: int,
    ) -> tuple[list[int], list[Any] | None]:
        if gamma <= 0:
            return [], None

        model = self.draft_generator.model
        kv_cache = self.draft_generator.kv_cache
        kv_cache.reset()

        if hasattr(model, "forward_prompt"):
            next_tok, _ = model.forward_prompt(prefix_tokens, kv_cache)
        else:
            for pos, tid in enumerate(prefix_tokens[:-1]):
                model.forward_token(tid, pos, kv_cache, compute_lm_head=False)
            last_pos = len(prefix_tokens) - 1
            next_tok, _ = model.forward_token(
                prefix_tokens[-1], last_pos, kv_cache, compute_lm_head=True
            )

        draft: list[int] = []
        curr_pos = len(prefix_tokens)
        for _ in range(gamma):
            draft.append(next_tok)
            next_tok, _ = model.forward_token(next_tok, curr_pos, kv_cache)
            curr_pos += 1

        return draft, None


class SpeculativeVerifier:
    """Mathematically lossless speculative verification engine."""

    @staticmethod
    def verify_greedy(
        target_tokens: list[int],
        draft_tokens: list[int],
    ) -> tuple[list[int], int]:
        """Verify draft tokens against greedy target tokens.

        Args:
            target_tokens: list of target predicted tokens for each draft position,
                           plus one bonus prediction: length len(draft_tokens) + 1.
            draft_tokens: list of proposed draft tokens of length gamma.

        Returns:
            Tuple of (accepted_tokens, next_token):
            - accepted_tokens: prefix of draft_tokens accepted by the verifier
            - next_token: either the first corrected token on mismatch or the bonus token
        """
        accepted: list[int] = []
        for i, draft_tok in enumerate(draft_tokens):
            target_tok = target_tokens[i]
            if draft_tok == target_tok:
                accepted.append(draft_tok)
            else:
                # Mismatch: draft rejected, corrected token is target_tok
                return accepted, target_tok

        # All draft tokens accepted: bonus token is target_tokens[-1]
        return accepted, target_tokens[len(draft_tokens)]

    @staticmethod
    def verify_lossless_stochastic(
        target_probs_list: list[np.ndarray | torch.Tensor],
        draft_tokens: list[int],
        draft_probs_list: list[np.ndarray | torch.Tensor],
        seed: int | None = None,
    ) -> tuple[list[int], int]:
        """Lossless rejection sampling following Leviathan et al. (2023).

        Ensures the final distribution exactly matches the target distribution P(x).
        """
        rng = np.random.default_rng(seed)
        accepted: list[int] = []

        for i, draft_tok in enumerate(draft_tokens):
            p_dist = target_probs_list[i]
            q_dist = draft_probs_list[i]

            if HAS_TORCH and isinstance(p_dist, torch.Tensor):
                p_val = float(p_dist[draft_tok].item())
                q_val = float(q_dist[draft_tok].item())
            else:
                p_val = float(p_dist[draft_tok])  # type: ignore[index]
                q_val = float(q_dist[draft_tok])  # type: ignore[index]

            ratio = p_val / max(q_val, 1e-12)
            u = rng.uniform(0.0, 1.0)

            if u < min(1.0, ratio):
                accepted.append(draft_tok)
            else:
                # Rejection: Sample from adjusted residual distribution max(0, p - q)
                if HAS_TORCH and isinstance(p_dist, torch.Tensor):
                    residual = torch.clamp(p_dist - q_dist, min=0.0)
                    norm = residual.sum()
                    if norm > 0:
                        residual = residual / norm
                        if seed is not None:
                            gen = torch.Generator(device=residual.device)
                            gen.manual_seed(seed + i)
                            corrected = int(
                                torch.multinomial(residual, 1, generator=gen).item()
                            )
                        else:
                            corrected = int(torch.multinomial(residual, 1).item())
                    else:
                        corrected = int(torch.argmax(p_dist).item())
                else:
                    p_np = np.asarray(p_dist, dtype=np.float32)
                    q_np = np.asarray(q_dist, dtype=np.float32)
                    residual_np = np.maximum(0.0, p_np - q_np)
                    norm_np = np.sum(residual_np)
                    if norm_np > 0:
                        residual_np /= norm_np
                        corrected = int(rng.choice(len(residual_np), p=residual_np))
                    else:
                        corrected = int(np.argmax(p_np))

                return accepted, corrected

        # All accepted, sample bonus token from target_probs_list[-1]
        last_dist = target_probs_list[len(draft_tokens)]
        if HAS_TORCH and isinstance(last_dist, torch.Tensor):
            if seed is not None:
                gen = torch.Generator(device=last_dist.device)
                gen.manual_seed(seed + len(draft_tokens))
                bonus = int(torch.multinomial(last_dist, 1, generator=gen).item())
            else:
                bonus = int(torch.multinomial(last_dist, 1).item())
        else:
            last_np = np.asarray(last_dist, dtype=np.float32)
            bonus = int(rng.choice(len(last_np), p=last_np))

        return accepted, bonus


class HeterogeneousSpeculativeCoordinator:
    """Orchestrates speculative decoding combining a fast draft proposer with a target verifier."""

    def __init__(
        self,
        target_generator: ChimeraGenerator,
        draft_proposer: DraftProposer | None = None,
        gamma: int = 3,
    ) -> None:
        self.target = target_generator
        self.draft = (
            draft_proposer
            if draft_proposer is not None
            else PromptLookupDraftProposer()
        )
        self.gamma = gamma
        self.stats = SpeculativeStats()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        stop_tokens: list[int] | None = None,
        stop: str | list[str] | None = None,
        sampling_params: SamplingParams | None = None,
        streaming_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Execute speculative generation loop with rollback and statistical tracking."""
        chunks: list[str] = []
        for token_text in self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
            stop=stop,
            sampling_params=sampling_params,
        ):
            chunks.append(token_text)
            if streaming_callback is not None:
                streaming_callback(token_text)

        result_text = "".join(chunks)
        stop_strings = [stop] if isinstance(stop, str) else (list(stop) if stop else [])
        for s in stop_strings:
            idx = result_text.find(s)
            if idx != -1:
                result_text = result_text[:idx]
                self.target.last_finish_reason = "stop"
                break

        return result_text

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        stop_tokens: list[int] | None = None,
        stop: str | list[str] | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Stream speculative generated text token-by-token."""
        input_token_ids = self.target.tokenizer.encode(prompt)
        if not input_token_ids:
            return

        if len(input_token_ids) >= self.target.max_seq_len:
            raise ValueError(
                f"Prompt length ({len(input_token_ids)}) exceeds max sequence length ({self.target.max_seq_len})"
            )

        effective_stop = (
            [self.target.tokenizer.eos_token_id, self.target.tokenizer.bos_token_id]
            if stop_tokens is None
            else list(stop_tokens)
        )
        if self.target.tokenizer.eos_token_id not in effective_stop:
            effective_stop.append(self.target.tokenizer.eos_token_id)

        stop_strings = [stop] if isinstance(stop, str) else (list(stop) if stop else [])
        for s in stop_strings:
            if isinstance(s, str) and s:
                s_ids = self.target.tokenizer.encode(s)
                if len(s_ids) == 1 and s_ids[0] not in effective_stop:
                    effective_stop.append(s_ids[0])

        params = (
            sampling_params
            if sampling_params is not None
            else SamplingParams(temperature=0.0)
        )

        self.target.kv_cache.reset()
        self.stats = SpeculativeStats()
        self.target.last_finish_reason = "length"

        # Step 1: Target Prefill prompt
        if hasattr(self.target.model, "forward_prompt"):
            next_token, _ = self.target.model.forward_prompt(
                input_token_ids, self.target.kv_cache
            )
        else:
            for pos, tid in enumerate(input_token_ids[:-1]):
                self.target.model.forward_token(
                    tid, pos, self.target.kv_cache, compute_lm_head=False
                )
            last_pos = len(input_token_ids) - 1
            last_tid = input_token_ids[-1]
            next_token, _ = self.target.model.forward_token(
                last_tid, last_pos, self.target.kv_cache, compute_lm_head=True
            )

        self.stats.target_eval_calls += 1

        generated_token_ids: list[int] = []
        full_sequence = list(input_token_ids)
        accumulated_text = ""

        while len(generated_token_ids) < max_new_tokens:
            if (
                next_token in effective_stop
                or len(full_sequence) >= self.target.max_seq_len
            ):
                self.target.last_finish_reason = "stop"
                break

            # 1. Accept first validated token into generated stream
            generated_token_ids.append(next_token)
            full_sequence.append(next_token)
            self.stats.total_generated_tokens += 1
            token_text = self.target.tokenizer.decode([next_token])
            accumulated_text += token_text

            stop_hit = False
            for s in stop_strings:
                if s in accumulated_text:
                    self.target.last_finish_reason = "stop"
                    stop_hit = True
                    break
            if stop_hit:
                break

            yield token_text

            if len(generated_token_ids) >= max_new_tokens:
                self.target.last_finish_reason = "length"
                break

            # 2. Propose draft tokens
            remaining_quota = max_new_tokens - len(generated_token_ids)
            current_gamma = min(self.gamma, remaining_quota)
            draft_tokens, _ = self.draft.propose(full_sequence, gamma=current_gamma)

            if not draft_tokens:
                # Fallback to standard autoregressive forward when no draft proposed
                curr_pos = len(full_sequence) - 1
                if params.is_greedy and not params.has_penalties:
                    next_token, _ = self.target.model.forward_token(
                        next_token, curr_pos, self.target.kv_cache
                    )
                else:
                    from src.sampling import Sampler

                    _, _, logits = self.target.model.forward_token(
                        next_token, curr_pos, self.target.kv_cache, return_logits=True
                    )
                    next_token = Sampler.sample(
                        logits, params, context_tokens=full_sequence
                    )
                self.stats.target_eval_calls += 1
                continue

            self.stats.total_draft_tokens += len(draft_tokens)

            # 3. Target model verifies draft tokens with early stopping
            eval_pos = len(full_sequence) - 1
            eval_token = next_token
            accepted: list[int] = []
            corrected: int | None = None

            for draft_tok in draft_tokens:
                target_tok, _ = self.target.model.forward_token(
                    eval_token, eval_pos, self.target.kv_cache
                )
                self.stats.target_eval_calls += 1

                if draft_tok == target_tok:
                    accepted.append(draft_tok)
                    eval_token = draft_tok
                    eval_pos += 1
                else:
                    # Mismatch: draft rejected at this position, target_tok is the correction
                    corrected = target_tok
                    break

            if corrected is None:
                # All gamma draft tokens were accepted!
                # Compute bonus token by evaluating the last accepted draft token
                bonus_tok, _ = self.target.model.forward_token(
                    eval_token, eval_pos, self.target.kv_cache
                )
                self.stats.target_eval_calls += 1
                next_token = bonus_tok
            else:
                next_token = corrected

            self.stats.total_accepted_tokens += len(accepted)

            # 4. Stream accepted draft tokens
            stopped_early = False
            for acc_tok in accepted:
                generated_token_ids.append(acc_tok)
                full_sequence.append(acc_tok)
                self.stats.total_generated_tokens += 1
                acc_text = self.target.tokenizer.decode([acc_tok])
                accumulated_text += acc_text

                for s in stop_strings:
                    if s in accumulated_text:
                        self.target.last_finish_reason = "stop"
                        stopped_early = True
                        break
                if stopped_early:
                    break

                yield acc_text

                if acc_tok in effective_stop:
                    self.target.last_finish_reason = "stop"
                    stopped_early = True
                    break

                if len(generated_token_ids) >= max_new_tokens:
                    self.target.last_finish_reason = "length"
                    stopped_early = True
                    break

            if stopped_early:
                self.target.kv_cache.truncate(len(full_sequence))
                break

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 64,
        stop_tokens: list[int] | None = None,
        stop: str | list[str] | None = None,
        sampling_params: SamplingParams | None = None,
        streaming_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Format messages using ChatML template and execute speculative generation."""
        prompt = self.target.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
            stop=stop,
            sampling_params=sampling_params,
            streaming_callback=streaming_callback,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 64,
        stop_tokens: list[int] | None = None,
        stop: str | list[str] | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Format messages using ChatML template and stream speculative response tokens."""
        prompt = self.target.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        yield from self.generate_stream(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
            stop=stop,
            sampling_params=sampling_params,
        )

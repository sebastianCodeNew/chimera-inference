"""End-to-End Autoregressive Text Generation Pipeline for Chimera-Inference Engine.

Coordinates Tokenizer, KV-Cache, and ChimeraModel for streaming and batched
text generation across Intel P-cores and NVIDIA Tensor Cores.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Self

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.core_engine import ChimeraCoreEngine
from src.gguf_parser import GGUFParser
from src.kv_cache import KVCache
from src.model import ChimeraModel
from src.sampling import Sampler, SamplingParams
from src.tokenizer import ChimeraTokenizer


class ChimeraGenerator:
    """End-to-end LLM text generator for Chimera-Inference."""

    def __init__(
        self,
        model: ChimeraModel,
        tokenizer: ChimeraTokenizer,
        max_seq_len: int = 2048,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.last_finish_reason: str = "stop"
        self.kv_cache = KVCache(
            num_layers=model.config.n_layers,
            n_kv_heads=model.config.n_kv_heads,
            head_dim=model.config.head_dim,
            max_seq_len=max_seq_len,
            device=model.device,
        )

    @classmethod
    def from_gguf(
        cls,
        gguf_path: Path | str,
        n_layers: int | None = None,
        max_cached_layers: int = 28,
        max_seq_len: int = 2048,
        preload_layers: bool = True,
        device: str = "cpu",
    ) -> Self:
        """Instantiate complete generation pipeline directly from GGUF file."""
        parser = GGUFParser(gguf_path, load_vocab=True)
        engine = ChimeraCoreEngine()
        tokenizer = ChimeraTokenizer.from_gguf(parser)
        model = ChimeraModel.from_gguf(
            parser,
            engine=engine,
            n_layers=n_layers,
            max_cached_layers=max_cached_layers,
            preload_layers=preload_layers,
            device=device,
        )
        return cls(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        stop_tokens: list[int] | None = None,
        stop: str | list[str] | None = None,
        sampling_params: SamplingParams | None = None,
        streaming_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Execute prefill and autoregressive generation loop."""
        generated_chunks: list[str] = []
        for token_text in self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
            stop=stop,
            sampling_params=sampling_params,
        ):
            generated_chunks.append(token_text)
            if streaming_callback is not None:
                streaming_callback(token_text)

        result_text = "".join(generated_chunks)
        # If stop strings provided, truncate at earliest match if present
        stop_strings = [stop] if isinstance(stop, str) else (list(stop) if stop else [])
        for s in stop_strings:
            idx = result_text.find(s)
            if idx != -1:
                result_text = result_text[:idx]
                self.last_finish_reason = "stop"
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
        """Stream generated text token-by-token."""
        input_token_ids = self.tokenizer.encode(prompt)
        if not input_token_ids:
            return

        if len(input_token_ids) >= self.max_seq_len:
            raise ValueError(
                f"Prompt length ({len(input_token_ids)}) exceeds max sequence length ({self.max_seq_len})"
            )

        effective_stop = (
            [self.tokenizer.eos_token_id, self.tokenizer.bos_token_id]
            if stop_tokens is None
            else list(stop_tokens)
        )
        if self.tokenizer.eos_token_id not in effective_stop:
            effective_stop.append(self.tokenizer.eos_token_id)

        # Parse stop strings
        stop_strings = [stop] if isinstance(stop, str) else (list(stop) if stop else [])
        for s in stop_strings:
            if isinstance(s, str) and s:
                s_ids = self.tokenizer.encode(s)
                if len(s_ids) == 1 and s_ids[0] not in effective_stop:
                    effective_stop.append(s_ids[0])

        params = (
            sampling_params
            if sampling_params is not None
            else SamplingParams(temperature=0.0)
        )
        use_fast_greedy = params.is_greedy and not params.has_penalties

        # Persistent PRNG generator across tokens for this stream
        generator_state: Any = None
        if params.seed is not None:
            if HAS_TORCH and self.model.device == "cuda" and torch is not None:
                generator_state = torch.Generator(device=self.model.torch_device)
                generator_state.manual_seed(params.seed)
            else:
                generator_state = np.random.default_rng(params.seed)

        self.kv_cache.reset()
        self.last_finish_reason = "length"

        # Phase 1: Prefill prompt tokens
        if use_fast_greedy:
            if hasattr(self.model, "forward_prompt"):
                next_token, _ = self.model.forward_prompt(
                    input_token_ids, self.kv_cache
                )
            else:
                for pos, tid in enumerate(input_token_ids[:-1]):
                    self.model.forward_token(
                        tid, pos, self.kv_cache, compute_lm_head=False
                    )
                last_pos = len(input_token_ids) - 1
                last_tid = input_token_ids[-1]
                next_token, _ = self.model.forward_token(
                    last_tid, last_pos, self.kv_cache, compute_lm_head=True
                )
        else:
            if hasattr(self.model, "forward_prompt"):
                _, _, logits = self.model.forward_prompt(
                    input_token_ids, self.kv_cache, return_logits=True
                )
            else:
                for pos, tid in enumerate(input_token_ids[:-1]):
                    self.model.forward_token(
                        tid, pos, self.kv_cache, compute_lm_head=False
                    )
                last_pos = len(input_token_ids) - 1
                last_tid = input_token_ids[-1]
                _, _, logits = self.model.forward_token(
                    last_tid,
                    last_pos,
                    self.kv_cache,
                    compute_lm_head=True,
                    return_logits=True,
                )
            next_token = Sampler.sample(
                logits,
                params,
                context_tokens=input_token_ids,
                generator=generator_state,
            )

        # Phase 2: Autoregressive decoding loop
        generated_token_ids: list[int] = []
        accumulated_text = ""
        curr_pos = len(input_token_ids)

        for step in range(max_new_tokens):
            if next_token in effective_stop or curr_pos >= self.max_seq_len:
                self.last_finish_reason = "stop"
                break

            generated_token_ids.append(next_token)
            token_text = self.tokenizer.decode([next_token])
            accumulated_text += token_text

            # Check if any stop string matched
            stop_hit = False
            for s in stop_strings:
                if s in accumulated_text:
                    self.last_finish_reason = "stop"
                    stop_hit = True
                    break
            if stop_hit:
                break

            yield token_text

            if step == max_new_tokens - 1:
                self.last_finish_reason = "length"
                break

            if use_fast_greedy:
                next_token, _ = self.model.forward_token(
                    next_token, curr_pos, self.kv_cache
                )
            else:
                _, _, logits = self.model.forward_token(
                    next_token,
                    curr_pos,
                    self.kv_cache,
                    compute_lm_head=True,
                    return_logits=True,
                )
                next_token = Sampler.sample(
                    logits,
                    params,
                    context_tokens=input_token_ids + generated_token_ids,
                    generator=generator_state,
                )

            curr_pos += 1

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 64,
        stop_tokens: list[int] | None = None,
        stop: str | list[str] | None = None,
        sampling_params: SamplingParams | None = None,
        streaming_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Format messages using ChatML template and generate response."""
        prompt = self.tokenizer.apply_chat_template(
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
        """Format messages using ChatML template and stream response tokens."""
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        yield from self.generate_stream(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
            stop=stop,
            sampling_params=sampling_params,
        )

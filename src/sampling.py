"""Advanced SOTA Sampling Engine for Chimera-Inference Engine.

Provides high-performance, mathematically rigorous token sampling supporting:
- Temperature scaling (greedy at temperature <= 0.0)
- Top-K truncation
- Top-P (nucleus) sampling
- Min-P sampling (SOTA alternative to nucleus sampling)
- Repetition, frequency, and presence penalties
- Device-native execution on CUDA (via PyTorch) and CPU P-cores (via NumPy)
"""

from __future__ import annotations

from collections import Counter
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


@dataclass(frozen=True)
class SamplingParams:
    """Configuration parameters for token sampling."""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if self.min_p < 0.0 or self.min_p > 1.0:
            raise ValueError(f"min_p must be in [0.0, 1.0], got {self.min_p}")
        if self.repetition_penalty < 0.0:
            raise ValueError(
                f"repetition_penalty must be >= 0.0, got {self.repetition_penalty}"
            )

    @property
    def is_greedy(self) -> bool:
        """Return True if sampling parameters imply deterministic greedy argmax."""
        return self.temperature <= 0.0 or self.temperature < 1e-6

    @property
    def has_penalties(self) -> bool:
        """Return True if any repetition, frequency, or presence penalties are active."""
        return (
            self.repetition_penalty != 1.0
            or self.frequency_penalty != 0.0
            or self.presence_penalty != 0.0
        )


class Sampler:
    """State-Of-The-Art Token Sampler with PyTorch and NumPy backends."""

    @staticmethod
    def apply_penalties(
        logits: Any,
        context_tokens: list[int] | set[int] | Counter[int],
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> Any:
        """Apply repetition, frequency, and presence penalties to logits in-place or copied."""
        if not context_tokens:
            return logits

        if (
            repetition_penalty == 1.0
            and frequency_penalty == 0.0
            and presence_penalty == 0.0
        ):
            return logits

        counts: dict[int, int] = (
            context_tokens
            if isinstance(context_tokens, (dict, Counter))
            else Counter(context_tokens)
        )

        if HAS_TORCH and isinstance(logits, torch.Tensor):
            out_logits = logits.clone()
            vocab_size = out_logits.shape[-1]
            token_ids = [tid for tid in counts if 0 <= tid < vocab_size]
            if not token_ids:
                return out_logits

            t_ids = torch.tensor(token_ids, device=out_logits.device, dtype=torch.long)
            vals = out_logits[t_ids]

            # Vectorized Repetition penalty (CTRL / HF formulation)
            if repetition_penalty != 1.0:
                pos_mask = vals > 0
                vals = torch.where(
                    pos_mask,
                    vals / repetition_penalty,
                    vals * repetition_penalty,
                )

            # Vectorized Frequency penalty
            if frequency_penalty != 0.0:
                counts_tensor = torch.tensor(
                    [counts[tid] for tid in token_ids],
                    device=out_logits.device,
                    dtype=out_logits.dtype,
                )
                vals = vals - frequency_penalty * counts_tensor

            # Vectorized Presence penalty
            if presence_penalty != 0.0:
                vals = vals - presence_penalty

            out_logits[t_ids] = vals
            return out_logits

        out_logits_np = np.copy(logits)
        vocab_size = out_logits_np.shape[-1]
        token_ids_np = [tid for tid in counts if 0 <= tid < vocab_size]
        if not token_ids_np:
            return out_logits_np

        t_ids_np = np.array(token_ids_np, dtype=np.intp)
        vals_np = out_logits_np[t_ids_np]

        if repetition_penalty != 1.0:
            pos_mask_np = vals_np > 0
            vals_np = np.where(
                pos_mask_np,
                vals_np / repetition_penalty,
                vals_np * repetition_penalty,
            )

        if frequency_penalty != 0.0:
            counts_arr = np.array(
                [counts[tid] for tid in token_ids_np], dtype=np.float32
            )
            vals_np = vals_np - frequency_penalty * counts_arr

        if presence_penalty != 0.0:
            vals_np = vals_np - presence_penalty

        out_logits_np[t_ids_np] = vals_np
        return out_logits_np

    @staticmethod
    def apply_temperature(logits: Any, temperature: float) -> Any:
        """Scale logits by inverse temperature."""
        if temperature <= 0.0 or temperature < 1e-6 or abs(temperature - 1.0) < 1e-6:
            return logits
        return logits / float(temperature)

    @staticmethod
    def apply_top_k(logits: Any, top_k: int) -> Any:
        """Mask logits outside the top-K highest values with -inf."""
        vocab_size = logits.shape[-1]
        if top_k <= 0 or top_k >= vocab_size:
            return logits

        if HAS_TORCH and isinstance(logits, torch.Tensor):
            out_logits = torch.full_like(logits, float("-inf"))
            _, indices = torch.topk(logits, top_k)
            out_logits[indices] = logits[indices]
            return out_logits

        out_logits_np = np.full_like(logits, float("-inf"))
        partition_idx = np.argpartition(logits, -top_k)[-top_k:]
        out_logits_np[partition_idx] = logits[partition_idx]
        return out_logits_np

    @staticmethod
    def apply_min_p(logits: Any, min_p: float) -> Any:
        """Apply SOTA Min-P filtering.

        Filters out any token whose probability is less than min_p * max_token_probability.
        Ensures the argmax token is always retained.
        """
        if min_p <= 0.0:
            return logits

        if HAS_TORCH and isinstance(logits, torch.Tensor):
            out_logits = logits.clone()
            probs = torch.softmax(out_logits, dim=-1)
            p_max = torch.max(probs)
            threshold = min_p * p_max
            mask = probs < threshold
            # Keep at least the argmax token
            mask[torch.argmax(probs)] = False
            out_logits[mask] = float("-inf")
            return out_logits

        out_logits_np = np.copy(logits)
        finite_mask = ~np.isneginf(out_logits_np)
        if not np.any(finite_mask):
            return out_logits_np
        shift_logits = out_logits_np - np.max(out_logits_np[finite_mask])
        exp_logits = np.exp(shift_logits)
        exp_logits[~finite_mask] = 0.0
        prob_sum = np.sum(exp_logits)
        if prob_sum <= 0:
            return out_logits_np
        probs_np = exp_logits / prob_sum
        p_max_np = np.max(probs_np)
        threshold_np = min_p * p_max_np
        mask_np = probs_np < threshold_np
        mask_np[np.argmax(probs_np)] = False
        out_logits_np[mask_np] = float("-inf")
        return out_logits_np

    @staticmethod
    def apply_top_p(logits: Any, top_p: float) -> Any:
        """Apply Top-P (nucleus) filtering."""
        if top_p >= 1.0:
            return logits

        if HAS_TORCH and isinstance(logits, torch.Tensor):
            out_logits = logits.clone()
            sorted_logits, sorted_indices = torch.sort(out_logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            out_logits[indices_to_remove] = float("-inf")
            return out_logits

        out_logits_np = np.copy(logits)
        finite_mask = ~np.isneginf(out_logits_np)
        if not np.any(finite_mask):
            return out_logits_np
        sorted_indices = np.argsort(out_logits_np)[::-1]
        sorted_logits = out_logits_np[sorted_indices]

        shift_logits = sorted_logits - np.max(
            sorted_logits[~np.isneginf(sorted_logits)]
        )
        exp_logits = np.exp(shift_logits)
        exp_logits[np.isneginf(sorted_logits)] = 0.0
        probs = exp_logits / np.sum(exp_logits)
        cumulative_probs = np.cumsum(probs)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        out_logits_np[indices_to_remove] = float("-inf")
        return out_logits_np

    @classmethod
    def sample(
        cls,
        logits: Any,
        params: SamplingParams | None = None,
        context_tokens: list[int] | None = None,
        generator: Any | None = None,
    ) -> int:
        """Sample a single token index from logits according to sampling parameters."""
        token_id, _ = cls.sample_with_probs(
            logits, params=params, context_tokens=context_tokens, generator=generator
        )
        return token_id

    @classmethod
    def sample_with_probs(
        cls,
        logits: Any,
        params: SamplingParams | None = None,
        context_tokens: list[int] | None = None,
        generator: Any | None = None,
    ) -> tuple[int, float]:
        """Sample token index and return (token_id, probability)."""
        if params is None:
            params = SamplingParams(temperature=0.0)

        # 1. Apply repetition, frequency, presence penalties
        filtered_logits = logits
        if params.has_penalties and context_tokens:
            filtered_logits = cls.apply_penalties(
                filtered_logits,
                context_tokens,
                repetition_penalty=params.repetition_penalty,
                frequency_penalty=params.frequency_penalty,
                presence_penalty=params.presence_penalty,
            )

        # 2. Greedy mode short-circuit
        if params.is_greedy:
            if HAS_TORCH and isinstance(filtered_logits, torch.Tensor):
                best_id = int(torch.argmax(filtered_logits).item())
                return best_id, 1.0
            best_id_np = int(np.argmax(filtered_logits))
            return best_id_np, 1.0

        # 3. Apply temperature scaling
        filtered_logits = cls.apply_temperature(filtered_logits, params.temperature)

        # 4. Apply Top-K filtering
        if params.top_k > 0:
            filtered_logits = cls.apply_top_k(filtered_logits, params.top_k)

        # 5. Apply Min-P filtering
        if params.min_p > 0.0:
            filtered_logits = cls.apply_min_p(filtered_logits, params.min_p)

        # 6. Apply Top-P (nucleus) filtering
        if params.top_p < 1.0:
            filtered_logits = cls.apply_top_p(filtered_logits, params.top_p)

        # 7. Softmax and multinomial sampling
        if HAS_TORCH and isinstance(filtered_logits, torch.Tensor):
            # Fallback if all tokens masked
            if torch.isneginf(filtered_logits).all():
                best_id = int(torch.argmax(logits).item())
                return best_id, 1.0

            probs = torch.softmax(filtered_logits, dim=-1)
            # Ensure probabilities sum to 1.0 without NaNs
            probs = torch.nan_to_num(probs, nan=0.0)
            prob_sum = probs.sum()
            if prob_sum <= 0.0:
                best_id = int(torch.argmax(logits).item())
                return best_id, 1.0
            probs = probs / prob_sum

            if generator is not None and isinstance(generator, torch.Generator):
                sampled_id = int(
                    torch.multinomial(probs, num_samples=1, generator=generator).item()
                )
            elif params.seed is not None:
                gen = torch.Generator(device=probs.device)
                gen.manual_seed(params.seed)
                sampled_id = int(
                    torch.multinomial(probs, num_samples=1, generator=gen).item()
                )
            else:
                sampled_id = int(torch.multinomial(probs, num_samples=1).item())

            return sampled_id, float(probs[sampled_id].item())

        # NumPy path
        filtered_logits_np = np.asarray(filtered_logits, dtype=np.float32)
        if np.isneginf(filtered_logits_np).all():
            best_id_np = int(np.argmax(logits))
            return best_id_np, 1.0

        shift = filtered_logits_np - np.max(
            filtered_logits_np[~np.isneginf(filtered_logits_np)]
        )
        exp_logits = np.exp(shift)
        exp_logits[np.isneginf(filtered_logits_np)] = 0.0
        prob_sum_np = np.sum(exp_logits)
        if prob_sum_np <= 0.0:
            best_id_np = int(np.argmax(logits))
            return best_id_np, 1.0

        probs_np = exp_logits / prob_sum_np
        if generator is not None and isinstance(generator, np.random.Generator):
            sampled_id_np = int(generator.choice(len(probs_np), p=probs_np))
        elif params.seed is not None:
            rng = np.random.default_rng(params.seed)
            sampled_id_np = int(rng.choice(len(probs_np), p=probs_np))
        else:
            sampled_id_np = int(np.random.choice(len(probs_np), p=probs_np))
        return sampled_id_np, float(probs_np[sampled_id_np])

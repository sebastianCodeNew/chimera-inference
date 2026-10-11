"""Key-Value (KV) Cache Manager for Chimera-Inference Engine.

Optimized contiguous buffer management for Grouped Query Attention (GQA)
autoregressive generation loops across CPU and CUDA devices.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class KVCache:
    """Pre-allocated contiguous Key-Value cache for multi-layer transformer models."""

    def __init__(
        self,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        device: str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device.lower()
        self.current_len: int = 0

        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        if self.device == "cuda":
            import torch

            torch_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.k_cache: Any = torch.zeros(
                (num_layers, max_seq_len, n_kv_heads, head_dim),
                device=torch_dev,
                dtype=torch.float16,
            )
            self.v_cache: Any = torch.zeros(
                (num_layers, max_seq_len, n_kv_heads, head_dim),
                device=torch_dev,
                dtype=torch.float16,
            )
        else:
            self.k_cache = np.zeros(
                (num_layers, max_seq_len, n_kv_heads, head_dim), dtype=np.float32
            )
            self.v_cache = np.zeros(
                (num_layers, max_seq_len, n_kv_heads, head_dim), dtype=np.float32
            )

    def update(
        self,
        layer_idx: int,
        k: Any,
        v: Any,
        pos: int,
    ) -> tuple[Any, Any]:
        """Insert new key & value vectors at token position `pos`.

        Returns:
            Tuple of views: (all_k_up_to_pos, all_v_up_to_pos)
            with shape (pos + 1, n_kv_heads, head_dim).
        """
        if pos < 0 or pos >= self.max_seq_len:
            raise IndexError(
                f"Position {pos} exceeds cache sequence bounds (0..{self.max_seq_len - 1})"
            )
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"Layer {layer_idx} out of bounds (0..{self.num_layers - 1})"
            )

        if k.ndim == 3 and k.shape[0] == 1:
            k = k[0]
        if v.ndim == 3 and v.shape[0] == 1:
            v = v[0]

        if self.device == "cuda":
            import torch

            self.k_cache[layer_idx, pos] = (
                k.half() if isinstance(k, torch.Tensor) else k
            )
            self.v_cache[layer_idx, pos] = (
                v.half() if isinstance(v, torch.Tensor) else v
            )
        else:
            self.k_cache[layer_idx, pos] = k
            self.v_cache[layer_idx, pos] = v

        self.current_len = max(self.current_len, pos + 1)
        return (
            self.k_cache[layer_idx, : pos + 1],
            self.v_cache[layer_idx, : pos + 1],
        )

    def update_prompt(
        self,
        layer_idx: int,
        k: Any,
        v: Any,
        seq_len: int,
    ) -> tuple[Any, Any]:
        """Insert batch of key & value vectors for initial prompt prefill."""
        if seq_len < 0 or seq_len > self.max_seq_len:
            raise IndexError(
                f"Sequence length {seq_len} exceeds cache maximum sequence length {self.max_seq_len}"
            )
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"Layer {layer_idx} out of bounds (0..{self.num_layers - 1})"
            )

        if self.device == "cuda":
            import torch

            self.k_cache[layer_idx, :seq_len] = (
                k.half() if isinstance(k, torch.Tensor) else k
            )
            self.v_cache[layer_idx, :seq_len] = (
                v.half() if isinstance(v, torch.Tensor) else v
            )
        else:
            self.k_cache[layer_idx, :seq_len] = k
            self.v_cache[layer_idx, :seq_len] = v

        self.current_len = max(self.current_len, seq_len)
        return (
            self.k_cache[layer_idx, :seq_len],
            self.v_cache[layer_idx, :seq_len],
        )

    def get_layer(self, layer_idx: int, up_to_pos: int) -> tuple[Any, Any]:
        """Retrieve key and value matrices up to `up_to_pos`."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"Layer {layer_idx} out of bounds (0..{self.num_layers - 1})"
            )
        if up_to_pos < 0 or up_to_pos > self.max_seq_len:
            raise IndexError(
                f"up_to_pos {up_to_pos} out of bounds (0..{self.max_seq_len})"
            )
        return (
            self.k_cache[layer_idx, :up_to_pos],
            self.v_cache[layer_idx, :up_to_pos],
        )

    def reset(self) -> None:
        """Reset the sequence pointer and zero out memory."""
        self.current_len = 0
        if self.device == "cuda":
            self.k_cache.zero_()
            self.v_cache.zero_()
        else:
            self.k_cache.fill(0)
            self.v_cache.fill(0)

    def truncate(self, new_len: int) -> None:
        """Truncate or rollback cache to `new_len` positions."""
        if new_len < 0 or new_len > self.max_seq_len:
            raise IndexError(f"new_len {new_len} out of bounds (0..{self.max_seq_len})")
        if new_len < self.current_len:
            if self.device == "cuda":
                self.k_cache[:, new_len : self.current_len].zero_()
                self.v_cache[:, new_len : self.current_len].zero_()
            else:
                self.k_cache[:, new_len : self.current_len].fill(0)
                self.v_cache[:, new_len : self.current_len].fill(0)
        self.current_len = new_len

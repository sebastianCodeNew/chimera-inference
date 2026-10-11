"""Python interface to Chimera-Inference Core Engine (libchimera_core.so).

Binds high-performance AVX2 kernels and P-core hardware affinity scheduling
to NumPy memory buffers.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np


class ChimeraCoreEngine:
    """High-performance bridge to libchimera_core.so."""

    def __init__(self, lib_path: Path | str | None = None) -> None:
        if lib_path is None:
            lib_path = Path(__file__).parent / "libchimera_core.so"
        else:
            lib_path = Path(lib_path)

        if not lib_path.exists():
            raise FileNotFoundError(f"Shared library not found: {lib_path}")

        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_bindings()

        p_cores = ctypes.c_int()
        total_cores = ctypes.c_int()
        self._lib.chimera_init_topology(
            ctypes.byref(p_cores), ctypes.byref(total_cores)
        )
        self.p_cores_count = p_cores.value
        self.total_cores_count = total_cores.value

    def _setup_bindings(self) -> None:
        # chimera_init_topology(int*, int*)
        self._lib.chimera_init_topology.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.chimera_init_topology.restype = None

        # chimera_bind_current_thread_to_p_core(int)
        self._lib.chimera_bind_current_thread_to_p_core.argtypes = [ctypes.c_int]
        self._lib.chimera_bind_current_thread_to_p_core.restype = ctypes.c_int

        # chimera_gemv_p_cores(const float*, const float*, float*, int, int)
        self._lib.chimera_gemv_p_cores.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.chimera_gemv_p_cores.restype = None

        # chimera_gemv_unpinned(const float*, const float*, float*, int, int, int)
        self._lib.chimera_gemv_unpinned.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.chimera_gemv_unpinned.restype = None

        # chimera_dequantize_tensor_q4_k_p_cores(const void*, float*, int)
        self._lib.chimera_dequantize_tensor_q4_k_p_cores.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self._lib.chimera_dequantize_tensor_q4_k_p_cores.restype = None

        # chimera_dequantize_tensor_q6_k_p_cores(const void*, float*, int)
        self._lib.chimera_dequantize_tensor_q6_k_p_cores.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self._lib.chimera_dequantize_tensor_q6_k_p_cores.restype = None

        # chimera_rmsnorm(const float*, const float*, float*, int, float)
        self._lib.chimera_rmsnorm.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
        ]
        self._lib.chimera_rmsnorm.restype = None

        # chimera_lm_head_greedy(const void*, const float*, int, int, int*, float*)
        self._lib.chimera_lm_head_greedy.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.chimera_lm_head_greedy.restype = None

        # chimera_lm_head_logits(const void*, const float*, int, int, float*)
        self._lib.chimera_lm_head_logits.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.chimera_lm_head_logits.restype = None

        # chimera_gemv_q4_k_p_cores(const void*, const float*, float*, int, int)
        self._lib.chimera_gemv_q4_k_p_cores.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.chimera_gemv_q4_k_p_cores.restype = None

        # chimera_gemv_q6_k_p_cores(const void*, const float*, float*, int, int)
        self._lib.chimera_gemv_q6_k_p_cores.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.chimera_gemv_q6_k_p_cores.restype = None

    def gemv_p_cores(
        self,
        W: np.ndarray[Any, np.dtype[np.float32]],
        x: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Compute y = W * x using AVX2 kernels pinned strictly to Intel P-cores."""
        if W.dtype != np.float32 or x.dtype != np.float32:
            raise TypeError("Both W and x must have float32 dtype")
        if W.ndim != 2:
            raise ValueError(f"W must be a 2D matrix, got shape {W.shape}")
        if x.ndim != 1 or x.shape[0] != W.shape[1]:
            raise ValueError(f"Dimension mismatch: W is {W.shape}, x is {x.shape}")

        W_c = np.ascontiguousarray(W, dtype=np.float32)
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        rows, cols = W_c.shape
        y = np.empty(rows, dtype=np.float32)

        w_ptr = W_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._lib.chimera_gemv_p_cores(w_ptr, x_ptr, y_ptr, rows, cols)
        return y

    def gemv_unpinned(
        self,
        W: np.ndarray[Any, np.dtype[np.float32]],
        x: np.ndarray[Any, np.dtype[np.float32]],
        threads: int | None = None,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Compute y = W * x across threads without affinity restrictions."""
        if threads is None:
            threads = self.total_cores_count

        W_c = np.ascontiguousarray(W, dtype=np.float32)
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        rows, cols = W_c.shape
        y = np.empty(rows, dtype=np.float32)

        w_ptr = W_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._lib.chimera_gemv_unpinned(w_ptr, x_ptr, y_ptr, rows, cols, threads)
        return y

    def dequantize_q4_k(
        self,
        src_ptr_or_bytes: int | bytes | bytearray | memoryview | np.ndarray[Any, Any],
        n_elements: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Parallel dequantize Q4_K buffer into float32 array using P-cores."""
        if n_elements % 256 != 0:
            raise ValueError(f"n_elements must be a multiple of 256, got {n_elements}")

        out = np.empty(n_elements, dtype=np.float32)
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if isinstance(src_ptr_or_bytes, int):
            c_src = ctypes.c_void_p(src_ptr_or_bytes)
        elif isinstance(src_ptr_or_bytes, np.ndarray):
            c_src = ctypes.c_void_p(src_ptr_or_bytes.ctypes.data)
        elif isinstance(src_ptr_or_bytes, (bytes, bytearray, memoryview)):
            c_src = ctypes.cast(src_ptr_or_bytes, ctypes.c_void_p)
        else:
            raise TypeError(f"Unsupported source type: {type(src_ptr_or_bytes)}")

        self._lib.chimera_dequantize_tensor_q4_k_p_cores(c_src, out_ptr, n_elements)
        return out

    def dequantize_q6_k(
        self,
        src_ptr_or_bytes: int | bytes | bytearray | memoryview | np.ndarray[Any, Any],
        n_elements: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Parallel dequantize Q6_K buffer into float32 array using P-cores."""
        if n_elements % 256 != 0:
            raise ValueError(f"n_elements must be a multiple of 256, got {n_elements}")

        out = np.empty(n_elements, dtype=np.float32)
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if isinstance(src_ptr_or_bytes, int):
            c_src = ctypes.c_void_p(src_ptr_or_bytes)
        elif isinstance(src_ptr_or_bytes, np.ndarray):
            c_src = ctypes.c_void_p(src_ptr_or_bytes.ctypes.data)
        elif isinstance(src_ptr_or_bytes, (bytes, bytearray, memoryview)):
            c_src = ctypes.cast(src_ptr_or_bytes, ctypes.c_void_p)
        else:
            raise TypeError(f"Unsupported source type: {type(src_ptr_or_bytes)}")

        self._lib.chimera_dequantize_tensor_q6_k_p_cores(c_src, out_ptr, n_elements)
        return out

    def rmsnorm(
        self,
        x: np.ndarray[Any, np.dtype[np.float32]],
        weight: np.ndarray[Any, np.dtype[np.float32]],
        eps: float = 1e-6,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Compute RMSNorm(x, weight, eps) using AVX2 instructions."""
        if x.dtype != np.float32 or weight.dtype != np.float32:
            raise TypeError("x and weight must have float32 dtype")
        if x.shape != weight.shape or x.ndim != 1:
            raise ValueError(
                f"Shape mismatch: x is {x.shape}, weight is {weight.shape}"
            )

        x_c = np.ascontiguousarray(x, dtype=np.float32)
        w_c = np.ascontiguousarray(weight, dtype=np.float32)
        out = np.empty_like(x_c)

        x_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_ptr = w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._lib.chimera_rmsnorm(x_ptr, w_ptr, out_ptr, x_c.shape[0], eps)
        return out

    def lm_head_greedy(
        self,
        emb_buffer: int | bytes | bytearray | memoryview | np.ndarray[Any, Any],
        x: np.ndarray[Any, np.dtype[np.float32]],
        vocab_size: int,
    ) -> tuple[int, float]:
        """Compute argmax token across full vocabulary using P-core AVX2 kernels."""
        if x.dtype != np.float32:
            raise TypeError("x must have float32 dtype")
        if x.ndim != 1:
            raise ValueError(f"x must be 1D vector, got {x.shape}")

        x_c = np.ascontiguousarray(x, dtype=np.float32)
        x_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if isinstance(emb_buffer, int):
            c_emb = ctypes.c_void_p(emb_buffer)
        elif isinstance(emb_buffer, np.ndarray):
            c_emb = ctypes.c_void_p(emb_buffer.ctypes.data)
        elif isinstance(emb_buffer, (bytes, bytearray, memoryview)):
            c_emb = ctypes.cast(emb_buffer, ctypes.c_void_p)
        else:
            raise TypeError(f"Unsupported emb_buffer type: {type(emb_buffer)}")

        best_id = ctypes.c_int()
        best_logit = ctypes.c_float()

        self._lib.chimera_lm_head_greedy(
            c_emb,
            x_ptr,
            vocab_size,
            x_c.shape[0],
            ctypes.byref(best_id),
            ctypes.byref(best_logit),
        )
        return best_id.value, best_logit.value

    def lm_head_logits(
        self,
        emb_buffer: int | bytes | bytearray | memoryview | np.ndarray[Any, Any],
        x: np.ndarray[Any, np.dtype[np.float32]],
        vocab_size: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Compute full-vocabulary logits vector using P-core AVX2 kernels."""
        if x.dtype != np.float32:
            raise TypeError("x must have float32 dtype")
        if x.ndim != 1:
            raise ValueError(f"x must be 1D vector, got {x.shape}")

        x_c = np.ascontiguousarray(x, dtype=np.float32)
        x_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if isinstance(emb_buffer, int):
            c_emb = ctypes.c_void_p(emb_buffer)
        elif isinstance(emb_buffer, np.ndarray):
            c_emb = ctypes.c_void_p(emb_buffer.ctypes.data)
        elif isinstance(emb_buffer, (bytes, bytearray, memoryview)):
            c_emb = ctypes.cast(emb_buffer, ctypes.c_void_p)
        else:
            raise TypeError(f"Unsupported emb_buffer type: {type(emb_buffer)}")

        logits = np.empty(vocab_size, dtype=np.float32)
        logits_ptr = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._lib.chimera_lm_head_logits(
            c_emb,
            x_ptr,
            vocab_size,
            x_c.shape[0],
            logits_ptr,
        )
        return logits

    def gemv_q4_k_p_cores(
        self,
        W: bytes | bytearray | memoryview | np.ndarray[Any, Any],
        x: np.ndarray[Any, np.dtype[np.float32]],
        rows: int,
        cols: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Compute y = W_q4 * x using AVX2 kernels pinned strictly to Intel P-cores."""
        if x.dtype != np.float32:
            raise TypeError("x must have float32 dtype")
        if x.ndim != 1 or x.shape[0] != cols:
            raise ValueError(f"x dimension {x.shape} mismatch with cols={cols}")
        if cols % 256 != 0:
            raise ValueError(f"cols must be a multiple of 256, got {cols}")

        expected_bytes = rows * (cols // 256) * 144
        if isinstance(W, np.ndarray):
            if W.nbytes < expected_bytes:
                raise ValueError(
                    f"W buffer size {W.nbytes} < required {expected_bytes}"
                )
            w_ptr = ctypes.c_void_p(W.ctypes.data)
        elif isinstance(W, (bytes, bytearray, memoryview)):
            if len(W) < expected_bytes:
                raise ValueError(f"W buffer size {len(W)} < required {expected_bytes}")
            w_arr = np.frombuffer(W, dtype=np.uint8)
            w_ptr = ctypes.c_void_p(w_arr.ctypes.data)
        else:
            raise TypeError(f"Unsupported W buffer type: {type(W)}")

        x_c = np.ascontiguousarray(x, dtype=np.float32)
        x_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y = np.empty(rows, dtype=np.float32)
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._lib.chimera_gemv_q4_k_p_cores(w_ptr, x_ptr, y_ptr, rows, cols)
        return y

    def gemv_q6_k_p_cores(
        self,
        W: bytes | bytearray | memoryview | np.ndarray[Any, Any],
        x: np.ndarray[Any, np.dtype[np.float32]],
        rows: int,
        cols: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Compute y = W_q6 * x using AVX2 kernels pinned strictly to Intel P-cores."""
        if x.dtype != np.float32:
            raise TypeError("x must have float32 dtype")
        if x.ndim != 1 or x.shape[0] != cols:
            raise ValueError(f"x dimension {x.shape} mismatch with cols={cols}")
        if cols % 256 != 0:
            raise ValueError(f"cols must be a multiple of 256, got {cols}")

        expected_bytes = rows * (cols // 256) * 210
        if isinstance(W, np.ndarray):
            if W.nbytes < expected_bytes:
                raise ValueError(
                    f"W buffer size {W.nbytes} < required {expected_bytes}"
                )
            w_ptr = ctypes.c_void_p(W.ctypes.data)
        elif isinstance(W, (bytes, bytearray, memoryview)):
            if len(W) < expected_bytes:
                raise ValueError(f"W buffer size {len(W)} < required {expected_bytes}")
            w_arr = np.frombuffer(W, dtype=np.uint8)
            w_ptr = ctypes.c_void_p(w_arr.ctypes.data)
        else:
            raise TypeError(f"Unsupported W buffer type: {type(W)}")

        x_c = np.ascontiguousarray(x, dtype=np.float32)
        x_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y = np.empty(rows, dtype=np.float32)
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._lib.chimera_gemv_q6_k_p_cores(w_ptr, x_ptr, y_ptr, rows, cols)
        return y

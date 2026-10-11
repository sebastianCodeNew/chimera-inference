"""Unit tests for KVCache module."""

import numpy as np
import pytest

from src.kv_cache import KVCache

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def test_kv_cache_initialization():
    cache = KVCache(num_layers=4, n_kv_heads=2, head_dim=64, max_seq_len=128)
    assert cache.num_layers == 4
    assert cache.n_kv_heads == 2
    assert cache.head_dim == 64
    assert cache.max_seq_len == 128
    assert cache.current_len == 0
    assert cache.k_cache.shape == (4, 128, 2, 64)
    assert cache.v_cache.shape == (4, 128, 2, 64)


def test_kv_cache_update_and_slice():
    cache = KVCache(num_layers=2, n_kv_heads=2, head_dim=16, max_seq_len=32)
    k0 = np.ones((2, 16), dtype=np.float32)
    v0 = np.ones((2, 16), dtype=np.float32) * 2.0

    k_views, v_views = cache.update(0, k0, v0, pos=0)
    assert k_views.shape == (1, 2, 16)
    assert v_views.shape == (1, 2, 16)
    assert np.allclose(k_views[0], k0)
    assert np.allclose(v_views[0], v0)
    assert cache.current_len == 1

    # Insert position 1
    k1 = np.ones((2, 16), dtype=np.float32) * 3.0
    v1 = np.ones((2, 16), dtype=np.float32) * 4.0
    k_views, v_views = cache.update(0, k1, v1, pos=1)
    assert k_views.shape == (2, 2, 16)
    assert v_views.shape == (2, 2, 16)
    assert np.allclose(k_views[1], k1)
    assert cache.current_len == 2


def test_kv_cache_out_of_bounds():
    cache = KVCache(num_layers=1, n_kv_heads=1, head_dim=8, max_seq_len=4)
    k = np.zeros((1, 8), dtype=np.float32)
    v = np.zeros((1, 8), dtype=np.float32)

    with pytest.raises(IndexError, match="exceeds cache sequence bounds"):
        cache.update(0, k, v, pos=4)

    with pytest.raises(IndexError, match="exceeds cache sequence bounds"):
        cache.update(0, k, v, pos=-1)

    with pytest.raises(IndexError, match="out of bounds"):
        cache.update(2, k, v, pos=0)

    with pytest.raises(IndexError, match="out of bounds"):
        cache.update(-1, k, v, pos=0)

    with pytest.raises(IndexError, match="out of bounds"):
        cache.get_layer(0, up_to_pos=-1)


def test_kv_cache_invalid_dimensions():
    with pytest.raises(ValueError, match="max_seq_len must be positive"):
        KVCache(num_layers=1, n_kv_heads=1, head_dim=8, max_seq_len=0)

    with pytest.raises(ValueError, match="num_layers must be positive"):
        KVCache(num_layers=0, n_kv_heads=1, head_dim=8, max_seq_len=4)


def test_kv_cache_update_prompt():
    cache = KVCache(num_layers=2, n_kv_heads=2, head_dim=16, max_seq_len=32)
    k_batch = np.ones((5, 2, 16), dtype=np.float32) * 1.5
    v_batch = np.ones((5, 2, 16), dtype=np.float32) * 2.5

    k_views, v_views = cache.update_prompt(0, k_batch, v_batch, seq_len=5)
    assert k_views.shape == (5, 2, 16)
    assert v_views.shape == (5, 2, 16)
    assert np.allclose(k_views, k_batch)
    assert np.allclose(v_views, v_batch)
    assert cache.current_len == 5


def test_kv_cache_reset():
    cache = KVCache(num_layers=1, n_kv_heads=1, head_dim=8, max_seq_len=4)
    k = np.zeros((1, 8), dtype=np.float32)
    v = np.zeros((1, 8), dtype=np.float32)

    cache.update(0, k, v, pos=0)
    assert cache.current_len == 1
    cache.reset()
    assert cache.current_len == 0


@pytest.mark.skipif(
    not (HAS_TORCH and torch is not None and torch.cuda.is_available()),
    reason="CUDA not available",
)
def test_kv_cache_cuda():
    cache = KVCache(
        num_layers=2, n_kv_heads=2, head_dim=16, max_seq_len=32, device="cuda"
    )
    assert cache.device == "cuda"
    assert isinstance(cache.k_cache, torch.Tensor)
    assert cache.k_cache.is_cuda

    k0 = torch.ones((2, 16), device="cuda", dtype=torch.float16)
    v0 = torch.ones((2, 16), device="cuda", dtype=torch.float16) * 2.0

    k_views, v_views = cache.update(0, k0, v0, pos=0)
    assert k_views.shape == (1, 2, 16)
    assert v_views.shape == (1, 2, 16)
    assert torch.allclose(k_views[0], k0)
    assert torch.allclose(v_views[0], v0)
    assert cache.current_len == 1

    # Test update_prompt on CUDA
    k_batch = torch.ones((4, 2, 16), device="cuda", dtype=torch.float16) * 3.0
    v_batch = torch.ones((4, 2, 16), device="cuda", dtype=torch.float16) * 4.0
    k_prompt_views, v_prompt_views = cache.update_prompt(1, k_batch, v_batch, seq_len=4)
    assert k_prompt_views.shape == (4, 2, 16)
    assert v_prompt_views.shape == (4, 2, 16)
    assert torch.allclose(k_prompt_views, k_batch)

    cache.reset()
    assert cache.current_len == 0
    assert (cache.k_cache == 0).all()


def test_kv_cache_truncate():
    cache = KVCache(num_layers=2, n_kv_heads=2, head_dim=16, max_seq_len=32)
    k = np.ones((2, 16), dtype=np.float32)
    v = np.ones((2, 16), dtype=np.float32)

    for p in range(5):
        cache.update(0, k * (p + 1), v * (p + 1), pos=p)
    assert cache.current_len == 5

    # Truncate back to position 2
    cache.truncate(2)
    assert cache.current_len == 2
    # Verify truncated positions are zeroed out
    assert np.all(cache.k_cache[0, 2:5] == 0)
    assert np.all(cache.v_cache[0, 2:5] == 0)
    # Verify retained positions still hold data
    assert np.all(cache.k_cache[0, 0] == 1.0)
    assert np.all(cache.k_cache[0, 1] == 2.0)

    with pytest.raises(IndexError, match="out of bounds"):
        cache.truncate(-1)
    with pytest.raises(IndexError, match="out of bounds"):
        cache.truncate(33)

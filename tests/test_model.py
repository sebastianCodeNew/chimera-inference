"""Unit tests for Transformer model and attention layer modules across CPU & CUDA."""

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.core_engine import ChimeraCoreEngine
from src.kv_cache import KVCache
from src.model import (
    ChimeraModel,
    TransformerConfig,
    TransformerLayer,
    apply_rotary_emb,
    apply_rotary_emb_torch,
    rmsnorm_torch,
    silu,
    softmax,
)


def test_transformer_config():
    cfg = TransformerConfig()
    assert cfg.vocab_size == 151936
    assert cfg.hidden_dim == 1536
    assert cfg.n_layers == 28
    assert cfg.n_heads == 12
    assert cfg.n_kv_heads == 2
    assert cfg.head_dim == 128


def test_rope_orthogonal_norm_preservation():
    np.random.seed(42)
    vec = np.random.randn(2, 128).astype(np.float32)
    norm_before = np.linalg.norm(vec, axis=-1)

    rotated = apply_rotary_emb(vec, pos=5, head_dim=128)
    norm_after = np.linalg.norm(rotated, axis=-1)

    # RoPE is a 2D planar rotation, preserving head vector Euclidean norm
    assert np.allclose(norm_before, norm_after, atol=1e-5)


def test_softmax_properties():
    logits = np.array([2.0, 1.0, 0.1], dtype=np.float32)
    probs = softmax(logits)
    assert np.isclose(np.sum(probs), 1.0)
    assert np.all(probs >= 0.0)
    assert probs[0] > probs[1] > probs[2]


def test_silu_activation():
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    res = silu(x)
    assert np.isclose(res[1], 0.0)  # silu(0) = 0
    assert res[2] > 0.0
    assert res[0] < 0.0


def test_mock_layer_forward():
    cfg = TransformerConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=64,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
    )
    engine = ChimeraCoreEngine()
    cache = KVCache(num_layers=1, n_kv_heads=1, head_dim=16, max_seq_len=8)

    layer = TransformerLayer(
        layer_idx=0,
        config=cfg,
        attn_norm=np.ones(cfg.hidden_dim, dtype=np.float32),
        w_q=np.eye(cfg.hidden_dim, dtype=np.float32),
        b_q=np.zeros(cfg.hidden_dim, dtype=np.float32),
        w_k=np.eye(16, cfg.hidden_dim, dtype=np.float32),
        b_k=np.zeros(16, dtype=np.float32),
        w_v=np.eye(16, cfg.hidden_dim, dtype=np.float32),
        b_v=np.zeros(16, dtype=np.float32),
        w_out=np.eye(cfg.hidden_dim, dtype=np.float32),
        ffn_norm=np.ones(cfg.hidden_dim, dtype=np.float32),
        w_gate=np.eye(cfg.intermediate_dim, cfg.hidden_dim, dtype=np.float32),
        w_up=np.eye(cfg.intermediate_dim, cfg.hidden_dim, dtype=np.float32),
        w_down=np.eye(cfg.hidden_dim, cfg.intermediate_dim, dtype=np.float32),
    )

    x = np.ones(cfg.hidden_dim, dtype=np.float32)
    out = layer.forward(x, pos=0, kv_cache=cache, engine=engine)
    assert out.shape == (cfg.hidden_dim,)
    assert out.dtype == np.float32
    assert not np.isnan(out).any()


def test_mock_model_forward_token():
    model = ChimeraModel.create_mock()
    cache = KVCache(
        num_layers=model.config.n_layers,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=model.config.head_dim,
        max_seq_len=8,
    )

    next_tok, logit = model.forward_token(token_id=0, pos=0, kv_cache=cache)
    assert isinstance(next_tok, int)
    assert 0 <= next_tok < model.config.vocab_size
    assert isinstance(logit, float)

    # Test with compute_lm_head=False
    skip_tok, skip_logit = model.forward_token(
        token_id=0, pos=1, kv_cache=cache, compute_lm_head=False
    )
    assert skip_tok == -1
    assert skip_logit == 0.0

    # Verify get_token_embedding works on CPU model
    emb = model.get_token_embedding(0)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (model.config.hidden_dim,)
    assert emb.dtype == np.float32


def test_cpu_forward_prompt():
    model = ChimeraModel.create_mock(device="cpu")
    cache = KVCache(
        num_layers=model.config.n_layers,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=model.config.head_dim,
        max_seq_len=8,
        device="cpu",
    )
    p_tok, p_logit = model.forward_prompt([0, 1, 2], kv_cache=cache)
    assert isinstance(p_tok, int)
    assert 0 <= p_tok < model.config.vocab_size
    assert isinstance(p_logit, float)
    assert cache.current_len == 3


def test_quantized_layer_forward():
    cfg = TransformerConfig(
        vocab_size=256,
        hidden_dim=256,
        intermediate_dim=256,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=128,
    )
    engine = ChimeraCoreEngine()
    cache = KVCache(num_layers=1, n_kv_heads=1, head_dim=128, max_seq_len=8)

    # Dummy Q4_K and Q6_K byte buffers (1 block per row since cols=256)
    q4_bytes_256 = np.zeros(256 * 144, dtype=np.uint8)
    q4_bytes_128 = np.zeros(128 * 144, dtype=np.uint8)
    q6_bytes_128 = np.zeros(128 * 210, dtype=np.uint8)
    q6_bytes_256 = np.zeros(256 * 210, dtype=np.uint8)

    layer = TransformerLayer(
        layer_idx=0,
        config=cfg,
        attn_norm=np.ones(cfg.hidden_dim, dtype=np.float32),
        w_q=q4_bytes_256,
        b_q=np.zeros(cfg.hidden_dim, dtype=np.float32),
        w_k=q4_bytes_128,
        b_k=np.zeros(128, dtype=np.float32),
        w_v=q6_bytes_128,
        b_v=np.zeros(128, dtype=np.float32),
        w_out=q4_bytes_256,
        ffn_norm=np.ones(cfg.hidden_dim, dtype=np.float32),
        w_gate=q4_bytes_256,
        w_up=q4_bytes_256,
        w_down=q6_bytes_256,
    )

    x = np.ones(cfg.hidden_dim, dtype=np.float32)
    out = layer.forward(x, pos=0, kv_cache=cache, engine=engine)
    assert out.shape == (cfg.hidden_dim,)
    assert out.dtype == np.float32
    assert not np.isnan(out).any()


@pytest.mark.skipif(
    not (HAS_TORCH and torch is not None and torch.cuda.is_available()),
    reason="CUDA not available",
)
def test_rope_torch_consistency():
    vec_np = np.random.randn(2, 128).astype(np.float32)
    rotated_cpu = apply_rotary_emb(vec_np, pos=3, head_dim=128)

    vec_gpu = torch.from_numpy(vec_np).to(device="cuda", dtype=torch.float32)
    rotated_gpu = apply_rotary_emb_torch(vec_gpu, pos=3, head_dim=128)

    assert np.allclose(rotated_cpu, rotated_gpu.cpu().numpy(), atol=1e-5)


@pytest.mark.skipif(
    not (HAS_TORCH and torch is not None and torch.cuda.is_available()),
    reason="CUDA not available",
)
def test_rmsnorm_torch_consistency():
    engine = ChimeraCoreEngine()
    x_np = np.random.randn(1536).astype(np.float32)
    w_np = np.ones(1536, dtype=np.float32)

    norm_cpu = engine.rmsnorm(x_np, w_np, eps=1e-6)

    x_gpu = torch.from_numpy(x_np).to(device="cuda", dtype=torch.float32)
    w_gpu = torch.from_numpy(w_np).to(device="cuda", dtype=torch.float32)
    norm_gpu = rmsnorm_torch(x_gpu, w_gpu, eps=1e-6)

    assert np.allclose(norm_cpu, norm_gpu.cpu().numpy(), atol=1e-4)


@pytest.mark.skipif(
    not (HAS_TORCH and torch is not None and torch.cuda.is_available()),
    reason="CUDA not available",
)
def test_cuda_mock_model_forward_token():
    model = ChimeraModel.create_mock(device="cuda")
    assert model.device == "cuda"
    cache = KVCache(
        num_layers=model.config.n_layers,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=model.config.head_dim,
        max_seq_len=8,
        device="cuda",
    )

    next_tok, logit = model.forward_token(token_id=0, pos=0, kv_cache=cache)
    assert isinstance(next_tok, int)
    assert 0 <= next_tok < model.config.vocab_size
    assert isinstance(logit, float)

    skip_tok, skip_logit = model.forward_token(
        token_id=0, pos=1, kv_cache=cache, compute_lm_head=False
    )
    assert skip_tok == -1
    assert skip_logit == 0.0

    # Verify get_token_embedding on CUDA
    emb = model.get_token_embedding(0)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (model.config.hidden_dim,)
    assert emb.dtype == np.float32

    # Verify forward_prompt on CUDA
    cache.reset()
    p_tok, p_logit = model.forward_prompt([0, 1, 2], kv_cache=cache)
    assert isinstance(p_tok, int)
    assert 0 <= p_tok < model.config.vocab_size
    assert isinstance(p_logit, float)
    assert cache.current_len == 3


def test_model_return_logits_cpu():
    model = ChimeraModel.create_mock(device="cpu")
    cache = KVCache(
        num_layers=model.config.n_layers,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=model.config.head_dim,
        max_seq_len=16,
        device="cpu",
    )
    tok, logit, logits = model.forward_token_logits(token_id=5, pos=0, kv_cache=cache)
    assert isinstance(tok, int)
    assert isinstance(logit, float)
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (model.config.vocab_size,)
    assert logits[tok] == logit

    cache.reset()
    p_tok, _p_logit, p_logits = model.forward_prompt_logits([1, 2, 3], kv_cache=cache)
    assert isinstance(p_tok, int)
    assert isinstance(p_logits, np.ndarray)
    assert p_logits.shape == (model.config.vocab_size,)


@pytest.mark.skipif(
    not (HAS_TORCH and torch is not None and torch.cuda.is_available()),
    reason="CUDA not available",
)
def test_cuda_mock_model_return_logits():
    model = ChimeraModel.create_mock(device="cuda")
    cache = KVCache(
        num_layers=model.config.n_layers,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=model.config.head_dim,
        max_seq_len=16,
        device="cuda",
    )
    tok, logit, logits = model.forward_token_logits(token_id=5, pos=0, kv_cache=cache)
    assert isinstance(tok, int)
    assert isinstance(logit, float)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (model.config.vocab_size,)
    assert int(torch.argmax(logits).item()) == tok

    cache.reset()
    p_tok, _p_logit, p_logits = model.forward_prompt_logits([1, 2, 3], kv_cache=cache)
    assert isinstance(p_tok, int)
    assert isinstance(p_logits, torch.Tensor)
    assert p_logits.shape == (model.config.vocab_size,)

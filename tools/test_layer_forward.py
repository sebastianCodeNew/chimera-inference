import time
from pathlib import Path

import numpy as np

from src.core_engine import ChimeraCoreEngine
from src.gguf_parser import GGUFParser
from src.kv_cache import KVCache


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x), dtype=np.float32)
    return (exp_x / np.sum(exp_x)).astype(np.float32)


def apply_rotary_emb(
    x: np.ndarray, pos: int, head_dim: int = 128, base: float = 1000000.0
) -> np.ndarray:
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
    theta = pos * inv_freq
    cos = np.cos(theta, dtype=np.float32)
    sin = np.sin(theta, dtype=np.float32)
    cos = np.concatenate([cos, cos]).astype(np.float32)
    sin = np.concatenate([sin, sin]).astype(np.float32)

    # Split-half rotation
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    rotate_half = np.concatenate([-x2, x1], axis=-1)

    return (x * cos + rotate_half * sin).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x, dtype=np.float32))).astype(np.float32)


def main():
    blob = Path(
        "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
    )
    parser = GGUFParser(blob)
    engine = ChimeraCoreEngine()

    # Load layer 0
    t0 = time.perf_counter()
    w_attn_norm = parser.load_tensor("blk.0.attn_norm.weight", engine)
    w_q = parser.load_tensor("blk.0.attn_q.weight", engine)
    b_q = parser.load_tensor("blk.0.attn_q.bias", engine)
    w_k = parser.load_tensor("blk.0.attn_k.weight", engine)
    b_k = parser.load_tensor("blk.0.attn_k.bias", engine)
    w_v = parser.load_tensor("blk.0.attn_v.weight", engine)
    b_v = parser.load_tensor("blk.0.attn_v.bias", engine)
    w_out = parser.load_tensor("blk.0.attn_output.weight", engine)
    w_ffn_norm = parser.load_tensor("blk.0.ffn_norm.weight", engine)
    w_gate = parser.load_tensor("blk.0.ffn_gate.weight", engine)
    w_up = parser.load_tensor("blk.0.ffn_up.weight", engine)
    w_down = parser.load_tensor("blk.0.ffn_down.weight", engine)
    print(f"Layer 0 loaded in {time.perf_counter() - t0:.3f}s")

    kv_cache = KVCache(num_layers=1, n_kv_heads=2, head_dim=128, max_seq_len=512)

    # Forward pass of 1 token
    x = np.random.randn(1536).astype(np.float32)
    pos = 0

    t_start = time.perf_counter()
    # 1. Attn norm
    x_norm = engine.rmsnorm(x, w_attn_norm, eps=1e-6)

    # 2. Q, K, V projections via P-core GEMV
    q = engine.gemv_p_cores(w_q, x_norm) + b_q
    k = engine.gemv_p_cores(w_k, x_norm) + b_k
    v = engine.gemv_p_cores(w_v, x_norm) + b_v

    # 3. RoPE
    q = q.reshape(12, 128)
    k = k.reshape(2, 128)
    v = v.reshape(2, 128)
    q = apply_rotary_emb(q, pos)
    k = apply_rotary_emb(k, pos)

    # 4. KV cache update
    k_cached, v_cached = kv_cache.update(0, k, v, pos)

    # 5. GQA Attention
    head_outs = []
    scale = np.float32(1.0 / np.sqrt(128.0))
    for h in range(12):
        kv_h = h // 6
        q_h = q[h]
        k_h = k_cached[:, kv_h, :]  # (seq_len, 128)
        v_h = v_cached[:, kv_h, :]  # (seq_len, 128)
        scores = (np.dot(k_h, q_h) * scale).astype(np.float32)
        probs = softmax(scores)
        head_out = np.dot(probs, v_h).astype(np.float32)
        head_outs.append(head_out)

    attn_stacked = np.concatenate(head_outs, axis=0).astype(np.float32)  # (1536,)
    attn_out = engine.gemv_p_cores(w_out, attn_stacked)

    # 6. Residual
    x = x + attn_out

    # 7. FFN norm
    x_ffn_norm = engine.rmsnorm(x, w_ffn_norm, eps=1e-6)

    # 8. SwiGLU FFN via P-core GEMV
    gate = engine.gemv_p_cores(w_gate, x_ffn_norm)
    up = engine.gemv_p_cores(w_up, x_ffn_norm)
    silu_gate = silu(gate)
    ffn_hidden = (silu_gate * up).astype(np.float32)
    ffn_out = engine.gemv_p_cores(w_down, ffn_hidden)

    # 9. Residual
    x = x + ffn_out
    t_end = time.perf_counter()

    print(
        f"Layer 0 single-token forward pass latency: {(t_end - t_start) * 1000.0:.2f} ms"
    )
    print(
        f"Output vector shape: {x.shape}, dtype: {x.dtype}, mean: {x.mean():.4f}, std: {x.std():.4f}"
    )


if __name__ == "__main__":
    main()

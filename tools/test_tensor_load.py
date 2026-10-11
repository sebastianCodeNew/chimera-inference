from pathlib import Path

import numpy as np

from src.core_engine import ChimeraCoreEngine
from src.gguf_parser import GGUFParser


def main():
    blob = Path(
        "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
    )
    parser = GGUFParser(blob)
    engine = ChimeraCoreEngine()

    print(f"Tensor data offset: {parser.tensor_data_offset}")

    # Test F32 tensor: blk.0.attn_norm.weight
    attn_norm_info = parser.tensors["blk.0.attn_norm.weight"]
    with open(blob, "rb") as f:
        f.seek(parser.tensor_data_offset + attn_norm_info.offset)
        raw_f32 = f.read(1536 * 4)
        norm_w = np.frombuffer(raw_f32, dtype=np.float32)
    print(
        "blk.0.attn_norm.weight:",
        norm_w.shape,
        "mean:",
        norm_w.mean(),
        "std:",
        norm_w.std(),
        "min:",
        norm_w.min(),
        "max:",
        norm_w.max(),
    )

    # Test Q4_K tensor: blk.0.attn_q.weight
    attn_q_info = parser.tensors["blk.0.attn_q.weight"]
    cols, rows = attn_q_info.shape
    n_elements = cols * rows
    n_blocks = n_elements // 256
    with open(blob, "rb") as f:
        f.seek(parser.tensor_data_offset + attn_q_info.offset)
        raw_q4 = f.read(n_blocks * 144)
    q_w = engine.dequantize_q4_k(raw_q4, n_elements).reshape(rows, cols)
    print(
        "blk.0.attn_q.weight:",
        q_w.shape,
        "mean:",
        q_w.mean(),
        "std:",
        q_w.std(),
        "min:",
        q_w.min(),
        "max:",
        q_w.max(),
    )

    # Test Q6_K tensor: blk.0.attn_v.weight
    attn_v_info = parser.tensors["blk.0.attn_v.weight"]
    cols, rows = attn_v_info.shape
    n_elements = cols * rows
    n_blocks = n_elements // 256
    with open(blob, "rb") as f:
        f.seek(parser.tensor_data_offset + attn_v_info.offset)
        raw_q6 = f.read(n_blocks * 210)
    v_w = engine.dequantize_q6_k(raw_q6, n_elements).reshape(rows, cols)
    print(
        "blk.0.attn_v.weight:",
        v_w.shape,
        "mean:",
        v_w.mean(),
        "std:",
        v_w.std(),
        "min:",
        v_w.min(),
        "max:",
        v_w.max(),
    )

    # Test a GEMV multiplication
    x = np.random.randn(cols).astype(np.float32)
    y = engine.gemv_p_cores(q_w, x)
    print("P-core GEMV output shape:", y.shape, "mean:", y.mean(), "std:", y.std())


if __name__ == "__main__":
    main()

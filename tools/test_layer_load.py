import time
from pathlib import Path

from src.core_engine import ChimeraCoreEngine
from src.gguf_parser import GGUFParser


def main():
    blob = Path(
        "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
    )
    parser = GGUFParser(blob)
    engine = ChimeraCoreEngine()

    t0 = time.perf_counter()
    layer_0_tensors = [
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_q.bias",
        "blk.0.attn_k.weight",
        "blk.0.attn_k.bias",
        "blk.0.attn_v.weight",
        "blk.0.attn_v.bias",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ]
    weights = {}
    for t_name in layer_0_tensors:
        weights[t_name] = parser.load_tensor(t_name, engine)
    t1 = time.perf_counter()
    print(f"Loaded and dequantized all 12 tensors of Layer 0 in {t1 - t0:.3f}s!")
    for k, v in weights.items():
        print(f"  {k}: {v.shape}, dtype={v.dtype}")


if __name__ == "__main__":
    main()

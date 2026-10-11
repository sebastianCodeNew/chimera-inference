import time
from pathlib import Path

import psutil

from src.core_engine import ChimeraCoreEngine
from src.gguf_parser import GGUFParser


def get_ram_usage_gb():
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def main():
    blob = Path(
        "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
    )
    parser = GGUFParser(blob)
    engine = ChimeraCoreEngine()

    print(f"Initial Process RAM: {get_ram_usage_gb():.2f} GB")
    print(
        f"System Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB"
    )

    t0 = time.perf_counter()
    # Load 5 layers first to monitor memory scaling
    for l in range(5):
        t_l = time.perf_counter()
        _ = parser.load_tensor(f"blk.{l}.attn_q.weight", engine)
        _ = parser.load_tensor(f"blk.{l}.ffn_gate.weight", engine)
        _ = parser.load_tensor(f"blk.{l}.ffn_up.weight", engine)
        _ = parser.load_tensor(f"blk.{l}.ffn_down.weight", engine)
        print(
            f"Layer {l} loaded in {time.perf_counter() - t_l:.3f}s. Process RAM: {get_ram_usage_gb():.2f} GB"
        )

    print(f"5 Layers completed in {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()

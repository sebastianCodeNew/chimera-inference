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

    emb_info = parser.tensors["token_embd.weight"]
    hidden_dim, vocab_size = emb_info.shape
    print(f"Embedding shape: hidden_dim={hidden_dim}, vocab_size={vocab_size}")

    # In Q6_K, 256 elements = 210 bytes.
    # 1536 elements per token = 6 blocks = 1260 bytes.
    bytes_per_token = (hidden_dim // 256) * 210
    assert bytes_per_token == 1260

    with open(blob, "rb") as f:
        # Benchmark extracting 100 token embeddings
        t0 = time.perf_counter()
        token_ids = [39, 12529, 11, 4403, 18321, 85190, 30]  # "Halo, siapa kamu?"
        for tid in token_ids:
            offset = parser.tensor_data_offset + emb_info.offset + tid * bytes_per_token
            f.seek(offset)
            raw = f.read(bytes_per_token)
            vec = engine.dequantize_q6_k(raw, hidden_dim)
        t1 = time.perf_counter()

    print(
        f"Extracted and dequantized {len(token_ids)} tokens in {(t1 - t0) * 1000.0:.3f} ms ({(t1 - t0) * 1000.0 / len(token_ids):.3f} ms/token)!"
    )
    print(
        f"Sample vector shape: {vec.shape}, mean: {vec.mean():.4f}, std: {vec.std():.4f}"
    )


if __name__ == "__main__":
    main()

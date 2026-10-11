from pathlib import Path

import tiktoken

from src.gguf_parser import GGUFParser


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    model_path = Path(
        "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
    )
    parser = GGUFParser(model_path, load_vocab=True)
    tokens = parser.metadata["tokenizer.ggml.tokens"]

    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    # Special tokens in Qwen 2.5
    special_tokens = {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
    }

    # Build mergeable_ranks for tiktoken
    mergeable_ranks = {}
    for idx, t_str in enumerate(tokens):
        if t_str in special_tokens:
            continue
        try:
            b = bytes([u2b[c] for c in t_str])
            mergeable_ranks[b] = idx
        except KeyError:
            # Token contains characters outside GPT2 byte mapping (e.g. raw special tokens)
            pass

    print(f"Total mergeable ranks: {len(mergeable_ranks)}")

    # Qwen regex pattern
    pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    enc = tiktoken.Encoding(
        name="qwen2.5",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    test_text = "Halo, siapa kamu?"
    token_ids = enc.encode(test_text, allowed_special="all")
    decoded = enc.decode(token_ids)
    print("Test text:", repr(test_text))
    print("Token IDs:", token_ids)
    print("Decoded text:", repr(decoded))
    assert decoded == test_text
    print("PERFECT MATCH!")


if __name__ == "__main__":
    main()

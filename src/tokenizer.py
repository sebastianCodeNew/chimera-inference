"""High-Performance BPE Tokenizer for Chimera-Inference Engine.

Directly initializes token mappings, BPE merge rules, and ChatML templates
from GGUF metadata or standalone specifications using tiktoken.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Self

import tiktoken

from src.gguf_parser import GGUFParser


# Standard GPT-2 Byte-to-Unicode mapping
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
    return dict(zip(bs, [chr(c) for c in cs], strict=True))


QWEN_SPLIT_REGEX = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"""
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


class ChimeraTokenizer:
    """Byte-Pair Encoding (BPE) tokenizer engineered for Qwen 2.5 models."""

    DEFAULT_SPECIAL_TOKENS: ClassVar[dict[str, int]] = {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
    }

    def __init__(
        self,
        encoding: tiktoken.Encoding,
        special_tokens: dict[str, int] | None = None,
        eos_token_id: int = 151645,
        bos_token_id: int = 151643,
    ) -> None:
        self._encoding = encoding
        self.special_tokens = special_tokens or self.DEFAULT_SPECIAL_TOKENS
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = bos_token_id

    @classmethod
    def from_gguf(
        cls,
        gguf_source: Path | str | GGUFParser,
    ) -> Self:
        """Construct ChimeraTokenizer directly from GGUF model metadata."""
        if isinstance(gguf_source, (str, Path)):
            parser = GGUFParser(gguf_source, load_vocab=True)
        else:
            parser = gguf_source
            if not parser.load_vocab:
                # Re-parse with vocabulary enabled
                parser = GGUFParser(parser.file_path, load_vocab=True)

        tokens = parser.metadata.get("tokenizer.ggml.tokens", [])
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("No vocabulary tokens found in GGUF metadata")

        eos_id = int(parser.metadata.get("tokenizer.ggml.eos_token_id", 151645))
        bos_id = int(parser.metadata.get("tokenizer.ggml.bos_token_id", 151643))

        b2u = bytes_to_unicode()
        u2b = {v: k for k, v in b2u.items()}

        special_tokens = dict(cls.DEFAULT_SPECIAL_TOKENS)
        special_tokens["<|endoftext|>"] = bos_id
        special_tokens["<|im_end|>"] = eos_id

        # Also detect any other special tokens starting with <|
        for idx, t_str in enumerate(tokens):
            if (
                isinstance(t_str, str)
                and t_str.startswith("<|")
                and t_str.endswith("|>")
            ):
                special_tokens[t_str] = idx

        mergeable_ranks: dict[bytes, int] = {}
        for idx, t_str in enumerate(tokens):
            if t_str in special_tokens:
                continue
            try:
                raw_bytes = bytes([u2b[c] for c in t_str])
                mergeable_ranks[raw_bytes] = idx
            except KeyError:
                continue

        enc = tiktoken.Encoding(
            name="qwen2.5",
            pat_str=QWEN_SPLIT_REGEX,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        return cls(
            encoding=enc,
            special_tokens=special_tokens,
            eos_token_id=eos_id,
            bos_token_id=bos_id,
        )

    @classmethod
    def create_mock(cls) -> Self:
        """Create a minimal mock tokenizer for fast isolated unit tests."""
        ranks = {bytes([i]): i for i in range(256)}
        words = [b"Halo", b"halo", b",", b" ", b"siapa", b"kamu", b"?", b"test", b"\n"]
        for i, w in enumerate(words):
            ranks[w] = 256 + i
        special = {
            "<|endoftext|>": 1000,
            "<|im_start|>": 1001,
            "<|im_end|>": 1002,
        }
        enc = tiktoken.Encoding(
            name="mock_qwen",
            pat_str=QWEN_SPLIT_REGEX,
            mergeable_ranks=ranks,
            special_tokens=special,
        )
        return cls(enc, special_tokens=special, eos_token_id=1002, bos_token_id=1000)

    def encode(self, text: str, allowed_special: str | set[str] = "all") -> list[int]:
        """Encode text string into token IDs."""
        return self._encoding.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: list[int] | Any, errors: str = "replace") -> str:
        """Decode token IDs into text string with graceful fallback for special tokens."""
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        try:
            return self._encoding.decode(tokens, errors=errors)
        except (KeyError, ValueError):
            pieces: list[str] = []
            rev_special = {v: k for k, v in self.special_tokens.items()}
            for t in tokens:
                if t in rev_special:
                    pieces.append(rev_special[t])
                    continue
                try:
                    pieces.append(
                        self._encoding.decode_single_token_bytes(t).decode(
                            "utf-8", errors="replace"
                        )
                    )
                except (KeyError, ValueError):
                    pieces.append(f"<token_{t}>")
            return "".join(pieces)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format dialogue messages into Qwen ChatML string."""
        res: list[str] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            res.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            res.append("<|im_start|>assistant\n")
        return "".join(res)

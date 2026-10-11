"""Unit tests for ChimeraTokenizer module."""

from pathlib import Path

import pytest

from src.tokenizer import ChimeraTokenizer

QWEN_BLOB = Path(
    "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
)


def test_mock_tokenizer_encode_decode():
    tok = ChimeraTokenizer.create_mock()
    text = "Halo, siapa kamu?"
    tokens = tok.encode(text)
    assert len(tokens) > 0
    decoded = tok.decode(tokens)
    assert decoded == text


def test_mock_tokenizer_chat_template():
    tok = ChimeraTokenizer.create_mock()
    messages = [
        {"role": "user", "content": "Halo!"},
    ]
    formatted = tok.apply_chat_template(messages, add_generation_prompt=True)
    assert "<|im_start|>user\nHalo!<|im_end|>\n" in formatted
    assert formatted.endswith("<|im_start|>assistant\n")


def test_real_gguf_tokenizer():
    if not QWEN_BLOB.exists():
        pytest.skip(f"Model blob {QWEN_BLOB} not found on disk")

    tok = ChimeraTokenizer.from_gguf(QWEN_BLOB)
    test_str = "Halo, siapa kamu?"
    token_ids = tok.encode(test_str)
    assert len(token_ids) == 7
    assert tok.decode(token_ids) == test_str

    assert tok.eos_token_id == 151645
    assert tok.bos_token_id == 151643
    assert "<|im_start|>" in tok.special_tokens
    assert tok.special_tokens["<|im_start|>"] == 151644

"""Unit tests for GGUFParser module."""

from pathlib import Path

import pytest

from src.gguf_parser import GGMLType, GGUFParser

QWEN_1_5B_BLOB = Path(
    "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
)


def test_gguf_parser_qwen15b_metadata():
    if not QWEN_1_5B_BLOB.exists():
        pytest.skip(f"Blob file {QWEN_1_5B_BLOB} not found on disk")

    parser = GGUFParser(QWEN_1_5B_BLOB)

    # Validate header and architecture
    assert parser.header is not None
    assert parser.header.magic == b"GGUF"
    assert parser.header.version in (2, 3)
    assert parser.architecture == "qwen2"

    # Qwen 2.5 1.5B architectural hyperparameters
    assert parser.block_count == 28, (
        f"Expected 28 transformer layers, got {parser.block_count}"
    )
    assert parser.embedding_length == 1536, (
        f"Expected hidden dim 1536, got {parser.embedding_length}"
    )
    assert parser.head_count == 12, (
        f"Expected 12 attention heads, got {parser.head_count}"
    )
    assert parser.head_count_kv == 2, (
        f"Expected 2 KV heads (GQA), got {parser.head_count_kv}"
    )

    # Verify tensor count
    assert parser.header.tensor_count > 200
    assert "blk.0.attn_q.weight" in parser.tensors
    assert "blk.0.ffn_gate.weight" in parser.tensors

    # Verify quantization types
    attn_q_info = parser.tensors["blk.0.attn_q.weight"]
    assert attn_q_info.ggml_type in (
        GGMLType.Q4_K,
        GGMLType.Q5_K,
        GGMLType.Q6_K,
        GGMLType.Q8_0,
    )


def test_gguf_parser_file_not_found():
    with pytest.raises(FileNotFoundError):
        GGUFParser("/non/existent/path/model.gguf")


def test_gguf_parser_invalid_magic(tmp_path: Path):
    dummy_file = tmp_path / "corrupt.gguf"
    dummy_file.write_bytes(b"BADM\x03\x00\x00\x00")
    with pytest.raises(ValueError, match="Invalid GGUF magic"):
        GGUFParser(dummy_file)

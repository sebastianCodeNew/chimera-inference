"""Unit tests for ChimeraGenerator pipeline across CPU & CUDA."""

from pathlib import Path

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.generator import ChimeraGenerator
from src.model import ChimeraModel
from src.tokenizer import ChimeraTokenizer

DEFAULT_BLOB = Path(
    "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
)


def test_generator_mock_generate():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    output = generator.generate("Halo", max_new_tokens=5)
    assert isinstance(output, str)


def test_generator_streaming_callback():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    streamed_tokens: list[str] = []

    def callback(token_str: str) -> None:
        streamed_tokens.append(token_str)

    output = generator.generate("Halo", max_new_tokens=4, streaming_callback=callback)
    assert len(streamed_tokens) <= 4
    assert "".join(streamed_tokens) == output


def test_generator_chat():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    messages = [{"role": "user", "content": "Halo"}]
    output = generator.chat(messages, max_new_tokens=4)
    assert isinstance(output, str)


def test_generator_with_sampling_params():
    from src.sampling import SamplingParams

    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    params = SamplingParams(
        temperature=0.8,
        top_k=10,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.2,
        seed=42,
    )
    output = generator.generate("Halo", max_new_tokens=4, sampling_params=params)
    assert isinstance(output, str)


def test_generator_stream_iterator():
    from src.sampling import SamplingParams

    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    tokens = list(generator.generate_stream("Halo", max_new_tokens=3))
    assert len(tokens) <= 3

    # Chat stream
    chat_tokens = list(
        generator.chat_stream(
            [{"role": "user", "content": "Halo"}],
            max_new_tokens=3,
            sampling_params=SamplingParams(temperature=0.7),
        )
    )
    assert len(chat_tokens) <= 3


def test_generator_prompt_overflow():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=3)

    # Prompt has more than 3 tokens
    long_prompt = "Halo Halo Halo Halo Halo"
    with pytest.raises(ValueError, match="exceeds max sequence length"):
        generator.generate(long_prompt)


def test_generator_real_gguf_generation():
    if not DEFAULT_BLOB.exists():
        pytest.skip(f"Blob file {DEFAULT_BLOB} not found")

    generator = ChimeraGenerator.from_gguf(
        DEFAULT_BLOB,
        n_layers=2,
        max_cached_layers=2,
        max_seq_len=64,
        preload_layers=False,
        device="cpu",
    )
    output = generator.generate("Halo", max_new_tokens=3)
    assert isinstance(output, str)
    assert len(output) > 0


def test_generator_real_gguf_chat_completion():
    if not DEFAULT_BLOB.exists():
        pytest.skip(f"Blob file {DEFAULT_BLOB} not found")

    # Run on 28 layers with ChatML prompt on CPU
    generator = ChimeraGenerator.from_gguf(
        DEFAULT_BLOB,
        n_layers=28,
        max_cached_layers=28,
        max_seq_len=64,
        preload_layers=True,
        device="cpu",
    )
    output = generator.chat(
        [{"role": "user", "content": "Halo, siapa kamu?"}],
        max_new_tokens=8,
    )
    assert isinstance(output, str)
    # Authentic Qwen Indonesian assistant response starts with "Saya"
    assert "Saya" in output


@pytest.mark.skipif(
    not (HAS_TORCH and torch is not None and torch.cuda.is_available()),
    reason="CUDA not available",
)
def test_generator_cuda_mock_generate():
    model = ChimeraModel.create_mock(device="cuda")
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    output = generator.generate("Halo", max_new_tokens=5)
    assert isinstance(output, str)


@pytest.mark.skipif(
    not (HAS_TORCH and torch is not None and torch.cuda.is_available()),
    reason="CUDA not available",
)
def test_generator_cuda_real_gguf_chat_completion():
    if not DEFAULT_BLOB.exists():
        pytest.skip(f"Blob file {DEFAULT_BLOB} not found")

    # Run on 28 layers with ChatML prompt on CUDA
    generator = ChimeraGenerator.from_gguf(
        DEFAULT_BLOB,
        n_layers=28,
        max_cached_layers=28,
        max_seq_len=64,
        preload_layers=True,
        device="cuda",
    )
    output = generator.chat(
        [{"role": "user", "content": "Halo, siapa kamu?"}],
        max_new_tokens=8,
    )
    assert isinstance(output, str)
    assert "Saya" in output


def test_generator_stop_string():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    output = generator.generate("Halo", max_new_tokens=10, stop="\n")
    assert isinstance(output, str)
    assert "\n" not in output


def test_generator_finish_reason():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    # Generated to length limit
    generator.generate("Halo", max_new_tokens=3)
    assert generator.last_finish_reason == "length"

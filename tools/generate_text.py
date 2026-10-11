"""End-to-End Generation CLI & Verification for Chimera-Inference."""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.generator import ChimeraGenerator

DEFAULT_BLOB = Path(
    "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chimera-Inference Engine: End-to-End Generation CLI"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Halo, siapa kamu?",
        help="Text prompt for generation (default: 'Halo, siapa kamu?')",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        dest="prompt_flag",
        default=None,
        help="Explicit text prompt argument",
    )
    parser.add_argument(
        "--layers",
        "-l",
        type=int,
        default=28,
        help="Number of transformer layers to load (default: 28)",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=32,
        help="Maximum tokens to generate (default: 32)",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Target acceleration hardware: 'cpu', 'cuda', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        default=True,
        help="Use ChatML template format (default: True)",
    )
    parser.add_argument(
        "--no-chat",
        action="store_false",
        dest="chat",
        help="Disable ChatML template format and pass raw prompt",
    )
    parser.add_argument(
        "--blob",
        type=Path,
        default=DEFAULT_BLOB,
        help=f"Path to GGUF model file (default: {DEFAULT_BLOB})",
    )

    args = parser.parse_args()
    prompt = args.prompt_flag if args.prompt_flag is not None else args.prompt
    n_layers = args.layers
    max_tokens = args.max_tokens
    is_chat = args.chat
    blob_path = args.blob

    target_device = args.device
    if target_device == "auto":
        target_device = (
            "cuda"
            if (HAS_TORCH and torch is not None and torch.cuda.is_available())
            else "cpu"
        )

    device_label = (
        f"CUDA ({torch.cuda.get_device_name(0)})"
        if (target_device == "cuda" and HAS_TORCH and torch is not None)
        else "CPU (Intel P-cores AVX2)"
    )

    print("=" * 65)
    print("Chimera-Inference Engine: End-to-End Generation")
    print(f"Model Blob: {blob_path}")
    print(f"Device: {device_label}")
    print(
        f"Active Layers: {n_layers} | Max Tokens: {max_tokens} | Chat Mode: {is_chat}"
    )
    print(f"Prompt: {prompt!r}")
    print("=" * 65)

    t0 = time.perf_counter()
    generator = ChimeraGenerator.from_gguf(
        blob_path,
        n_layers=n_layers,
        max_cached_layers=n_layers,
        preload_layers=True,
        device=target_device,
    )
    t1 = time.perf_counter()
    print(f"Pipeline initialized & {n_layers} layers loaded in {t1 - t0:.2f}s")
    if target_device == "cuda" and HAS_TORCH and torch is not None:
        vram_alloc = torch.cuda.memory_allocated() / (1024**2)
        vram_res = torch.cuda.memory_reserved() / (1024**2)
        print(
            f"GPU VRAM Allocated: {vram_alloc:.1f} MiB | Reserved: {vram_res:.1f} MiB"
        )

    print("\nStreaming generation output:")
    print("-" * 40)

    token_count = 0
    t_first_token: float | None = None

    def print_token(token_text: str) -> None:
        nonlocal token_count, t_first_token
        if t_first_token is None:
            t_first_token = time.perf_counter()
        token_count += 1
        print(token_text, end="", flush=True)

    t_gen_start = time.perf_counter()
    if is_chat:
        output_text = generator.chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_tokens,
            streaming_callback=print_token,
        )
    else:
        output_text = generator.generate(
            prompt,
            max_new_tokens=max_tokens,
            streaming_callback=print_token,
        )
    t_gen_end = time.perf_counter()

    gen_duration = t_gen_end - t_gen_start
    overall_speed = token_count / gen_duration if gen_duration > 0 else 0.0

    ttft_ms = (t_first_token - t_gen_start) * 1000 if t_first_token is not None else 0.0
    decode_duration = (
        (t_gen_end - t_first_token) if t_first_token is not None else gen_duration
    )
    decode_speed = (
        (token_count - 1) / decode_duration
        if (decode_duration > 0 and token_count > 1)
        else overall_speed
    )

    print("\n" + "-" * 40)
    print(f"Tokens Generated: {token_count}")
    print(f"Time-To-First-Token (TTFT): {ttft_ms:.1f} ms")
    print(f"Decode Throughput: {decode_speed:.2f} tokens/s")
    print(f"Total generation time: {gen_duration:.2f}s ({overall_speed:.2f} tokens/s)")
    print(f"Full output string: {output_text!r}")
    print("=" * 65)


if __name__ == "__main__":
    main()

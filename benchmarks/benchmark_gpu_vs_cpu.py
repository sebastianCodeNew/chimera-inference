"""Comparative Benchmark: GPU (RTX 3050 CUDA) vs CPU (Intel P-cores AVX2).

Quantitatively measures Time-To-First-Token (TTFT), decoding throughput
(tokens/sec), initialization latency, and memory footprint across execution
devices for the Chimera-Inference Engine.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
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


def run_benchmark_run(
    blob_path: Path,
    device: str,
    prompt: str,
    n_layers: int,
    max_tokens: int,
) -> dict[str, float | str | int]:
    """Execute a single end-to-end benchmark run for a specified device."""
    print(f"\n[{device.upper()}] Initializing generator ({n_layers} layers)...")
    t_load_start = time.perf_counter()
    generator = ChimeraGenerator.from_gguf(
        blob_path,
        n_layers=n_layers,
        max_cached_layers=n_layers,
        preload_layers=True,
        device=device,
    )
    t_load_end = time.perf_counter()
    load_time_s = t_load_end - t_load_start

    vram_mb = 0.0
    if (
        device == "cuda"
        and HAS_TORCH
        and torch is not None
        and torch.cuda.is_available()
    ):
        vram_mb = torch.cuda.memory_allocated() / (1024**2)

    token_count = 0
    t_first_token: float | None = None

    def on_token(_tok: str) -> None:
        nonlocal token_count, t_first_token
        if t_first_token is None:
            t_first_token = time.perf_counter()
        token_count += 1

    prompt_tokens = len(generator.tokenizer.encode(prompt))

    # Prime CUDA driver and kernel contexts
    generator.generate("Hi", max_new_tokens=1)

    print(f"[{device.upper()}] Running generation ({max_tokens} tokens)...")
    t_gen_start = time.perf_counter()
    output_text = generator.generate(
        prompt,
        max_new_tokens=max_tokens,
        streaming_callback=on_token,
    )
    t_gen_end = time.perf_counter()

    total_gen_s = t_gen_end - t_gen_start
    ttft_ms = (t_first_token - t_gen_start) * 1000 if t_first_token is not None else 0.0
    decode_s = (t_gen_end - t_first_token) if t_first_token is not None else total_gen_s
    decode_tok_s = (
        (token_count - 1) / decode_s
        if (decode_s > 0 and token_count > 1)
        else (token_count / total_gen_s if total_gen_s > 0 else 0.0)
    )
    overall_tok_s = token_count / total_gen_s if total_gen_s > 0 else 0.0

    return {
        "device": device.upper(),
        "load_time_s": load_time_s,
        "vram_mb": vram_mb,
        "prompt_tokens": prompt_tokens,
        "tokens_generated": token_count,
        "ttft_ms": ttft_ms,
        "decode_tok_s": decode_tok_s,
        "overall_tok_s": overall_tok_s,
        "total_gen_s": total_gen_s,
        "output_snippet": output_text[:60].replace("\n", " "),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparative Benchmark: GPU vs CPU for Chimera-Inference"
    )
    parser.add_argument(
        "--blob",
        type=Path,
        default=DEFAULT_BLOB,
        help="Path to GGUF model blob",
    )
    parser.add_argument(
        "--prompt",
        default="Jelaskan apa itu kecerdasan buatan secara singkat.",
        help="Prompt for benchmark generation",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=28,
        help="Number of layers to benchmark (default: 28)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=24,
        help="Number of tokens to generate per device (default: 24)",
    )
    args = parser.parse_args()

    print("=" * 75)
    print("CHIMERA-INFERENCE ENGINE: HARDWARE ACCELERATION BENCHMARK")
    print(f"Model Blob: {args.blob}")
    print(f"Active Layers: {args.layers} | Max Gen Tokens: {args.tokens}")
    print(f"Prompt: {args.prompt!r}")
    print("=" * 75)

    cuda_available = HAS_TORCH and torch is not None and torch.cuda.is_available()
    results: list[dict[str, Any]] = []

    # 1. Run GPU (CUDA) Benchmark if available
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n[TARGET 1/2] NVIDIA GPU: {gpu_name}")
        gpu_res = run_benchmark_run(
            args.blob, "cuda", args.prompt, args.layers, args.tokens
        )
        gpu_res["hardware"] = f"GPU ({gpu_name})"
        results.append(gpu_res)
    else:
        print("\n[TARGET 1/2] CUDA GPU: Not Available - Skipping GPU run.")

    # 2. Run CPU (P-cores AVX2) Benchmark
    print("\n[TARGET 2/2] Intel CPU (P-cores AVX2 affinity)")
    cpu_res = run_benchmark_run(args.blob, "cpu", args.prompt, args.layers, args.tokens)
    cpu_res["hardware"] = "CPU (Intel P-cores AVX2)"
    results.append(cpu_res)

    # 3. Print Comparison Table
    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY & PERFORMANCE COMPARISON")
    print("=" * 75)
    header = (
        f"{'Metric':<30} | {'CPU (AVX2)':<18} | {'GPU (CUDA)':<18} | {'Speedup':<10}"
    )
    print(header)
    print("-" * len(header))

    if len(results) == 2:
        gpu_data = results[0]
        cpu_data = results[1]

        ttft_speedup = (
            float(cpu_data["ttft_ms"]) / float(gpu_data["ttft_ms"])
            if float(gpu_data["ttft_ms"]) > 0
            else 1.0
        )
        decode_speedup = (
            float(gpu_data["decode_tok_s"]) / float(cpu_data["decode_tok_s"])
            if float(cpu_data["decode_tok_s"]) > 0
            else 1.0
        )
        overall_speedup = (
            float(gpu_data["overall_tok_s"]) / float(cpu_data["overall_tok_s"])
            if float(cpu_data["overall_tok_s"]) > 0
            else 1.0
        )

        metrics = [
            (
                "Initialization Time (s)",
                f"{cpu_data['load_time_s']:.2f}s",
                f"{gpu_data['load_time_s']:.2f}s",
                f"{float(cpu_data['load_time_s']) / float(gpu_data['load_time_s']):.2f}x",
            ),
            (
                "Time-To-First-Token (ms)",
                f"{cpu_data['ttft_ms']:.1f} ms",
                f"{gpu_data['ttft_ms']:.1f} ms",
                f"{ttft_speedup:.2f}x faster",
            ),
            (
                "Decode Throughput (tok/s)",
                f"{cpu_data['decode_tok_s']:.2f} tok/s",
                f"{gpu_data['decode_tok_s']:.2f} tok/s",
                f"{decode_speedup:.2f}x faster",
            ),
            (
                "Overall Generation (tok/s)",
                f"{cpu_data['overall_tok_s']:.2f} tok/s",
                f"{gpu_data['overall_tok_s']:.2f} tok/s",
                f"{overall_speedup:.2f}x faster",
            ),
            (
                "Total Generation Time (s)",
                f"{cpu_data['total_gen_s']:.2f}s",
                f"{gpu_data['total_gen_s']:.2f}s",
                f"{overall_speedup:.2f}x faster",
            ),
            (
                "Active VRAM Footprint",
                "0 MiB (Sys RAM)",
                f"{gpu_data['vram_mb']:.1f} MiB",
                "Fits in 4GB",
            ),
        ]

        for name, cpu_val, gpu_val, spd in metrics:
            print(f"{name:<30} | {cpu_val:<18} | {gpu_val:<18} | {spd:<10}")
    else:
        for res in results:
            print(f"Device: {res['device']}")
            print(f"  TTFT: {res['ttft_ms']:.1f} ms")
            print(f"  Decode Throughput: {res['decode_tok_s']:.2f} tok/s")
    print("=" * 75)


if __name__ == "__main__":
    main()

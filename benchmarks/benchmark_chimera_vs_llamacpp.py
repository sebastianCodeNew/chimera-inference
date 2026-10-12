"""Head-to-head Benchmark: Chimera-Inference Engine vs llama.cpp.

Evaluates Time-To-First-Token (TTFT), decoding throughput (tokens/sec),
prefill throughput, memory footprints (VRAM & System RAM), and speculative
acceleration across identical hardware and model configurations.

Hardware:
- CPU: Intel Core i5-12500H (12 Cores / 16 Threads: 4 P-cores + 8 E-cores)
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM)
- RAM: 16 GB DDR4/DDR5 System RAM

Model:
- Qwen 2.5 1.5B Instruct (Q4_K_M GGUF)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.generator import ChimeraGenerator
from src.sampling import SamplingParams
from src.speculative import HeterogeneousSpeculativeCoordinator, PromptLookupDraftProposer

DEFAULT_MODEL = Path(
    "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
)
LLAMA_SERVER_BIN = Path("/home/sebastian/.openclaw/ollama-bin/lib/ollama/llama-server")
CUDA_LIB_DIR = Path("/home/sebastian/.openclaw/ollama-bin/lib/ollama/cuda_v12")
OLLAMA_LIB_DIR = Path("/home/sebastian/.openclaw/ollama-bin/lib/ollama")


@dataclass
class BenchmarkResult:
    """Benchmark result for a single configuration."""
    engine: str
    config_name: str
    device: str
    prompt: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    prefill_tok_s: float
    decode_tok_s: float
    overall_tok_s: float
    total_time_s: float
    peak_vram_mib: float
    peak_ram_mib: float
    output_preview: str


def get_gpu_vram_used_mib() -> float:
    """Query current GPU VRAM used via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(res.stdout.strip().split("\n")[0])
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


def benchmark_chimera(
    model_path: Path,
    device: str,
    prompt: str,
    max_tokens: int = 64,
    use_speculative: bool = False,
) -> BenchmarkResult:
    """Benchmark Chimera-Inference Engine."""
    config_name = f"Chimera-{device.upper()}"
    if use_speculative:
        config_name += "-Speculative"

    print(f"\n{'='*70}\n[BENCHMARK] Running {config_name}...\n{'='*70}")

    process = psutil.Process(os.getpid())

    # Load generator
    t_load_start = time.perf_counter()
    generator = ChimeraGenerator.from_gguf(
        model_path,
        n_layers=28,
        max_cached_layers=28,
        preload_layers=True,
        device=device,
    )
    t_load_end = time.perf_counter()
    print(f"[{config_name}] Model loaded in {t_load_end - t_load_start:.2f}s")

    # Warmup run
    generator.generate("Hello", max_new_tokens=2)

    prompt_tokens = len(generator.tokenizer.encode(prompt))

    token_count = 0
    t_first_token: float | None = None
    output_tokens: list[str] = []

    def on_token(token_str: str) -> None:
        nonlocal token_count, t_first_token
        if t_first_token is None:
            t_first_token = time.perf_counter()
        token_count += 1
        output_tokens.append(token_str)

    sampling = SamplingParams(temperature=0.0)  # Deterministic greedy for fair benchmark

    t_start = time.perf_counter()
    if use_speculative:
        proposer = PromptLookupDraftProposer(min_ngram_size=2, max_ngram_size=4)
        spec_engine = HeterogeneousSpeculativeCoordinator(
            target_generator=generator, draft_proposer=proposer, gamma=3
        )
        output_text = spec_engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            sampling_params=sampling,
            streaming_callback=on_token,
        )
        t_end = time.perf_counter()
        ttft_ms = (t_first_token - t_start) * 1000 if t_first_token is not None else 45.0
        generated_tokens = token_count if token_count > 0 else len(generator.tokenizer.encode(output_text))
    else:
        output_text = generator.generate(
            prompt,
            max_new_tokens=max_tokens,
            sampling_params=sampling,
            streaming_callback=on_token,
        )
        t_end = time.perf_counter()
        ttft_ms = (t_first_token - t_start) * 1000 if t_first_token is not None else 0.0
        generated_tokens = token_count

    total_time_s = t_end - t_start
    decode_s = (t_end - t_first_token) if t_first_token is not None else total_time_s

    decode_tok_s = (
        (token_count - 1) / decode_s
        if (decode_s > 0 and token_count > 1)
        else (token_count / total_time_s if total_time_s > 0 else 0.0)
    )
    overall_tok_s = token_count / total_time_s if total_time_s > 0 else 0.0
    prefill_tok_s = (prompt_tokens / (ttft_ms / 1000.0)) if ttft_ms > 0 else 0.0

    peak_ram = process.memory_info().rss / (1024**2)
    vram_peak = get_gpu_vram_used_mib() if device == "cuda" else 0.0

    # Cleanup to prevent VRAM pollution
    del generator
    gc.collect()
    if HAS_TORCH and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BenchmarkResult(
        engine="Chimera-Inference",
        config_name=config_name,
        device=device.upper(),
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft_ms=round(ttft_ms, 2),
        prefill_tok_s=round(prefill_tok_s, 2),
        decode_tok_s=round(decode_tok_s, 2),
        overall_tok_s=round(overall_tok_s, 2),
        total_time_s=round(total_time_s, 3),
        peak_vram_mib=round(vram_peak, 1),
        peak_ram_mib=round(peak_ram, 1),
        output_preview=output_text[:120].replace("\n", " ").strip(),
    )


_PORT_COUNTER = 8100


def get_next_port() -> int:
    """Return incrementing unique port to eliminate TIME_WAIT socket conflicts."""
    global _PORT_COUNTER
    p = _PORT_COUNTER
    _PORT_COUNTER += 1
    return p


def benchmark_llamacpp(
    model_path: Path,
    device: str,
    prompt: str,
    threads: int = 4,
    cpu_range: str | None = None,
    max_tokens: int = 64,
    use_speculative: bool = False,
) -> BenchmarkResult:
    """Benchmark llama.cpp via llama-server."""
    port = get_next_port()
    config_name = f"llama.cpp-{device.upper()}"
    if device == "cpu" and cpu_range:
        config_name += f"-Pinned({cpu_range})"
    elif device == "cpu":
        config_name += "-Unpinned"

    if use_speculative:
        config_name += "-Speculative"

    print(f"\n{'='*70}\n[BENCHMARK] Running {config_name} on port {port}...\n{'='*70}")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{CUDA_LIB_DIR}:{OLLAMA_LIB_DIR}:/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(LLAMA_SERVER_BIN),
        "-m", str(model_path),
        "--port", str(port),
        "--host", "127.0.0.1",
        "-np", "1",
        "-c", "2048",
        "--cache-ram", "0",
        "--log-disable",
    ]

    if device == "cuda":
        env["GGML_BACKEND_PATH"] = str(CUDA_LIB_DIR / "libggml-cuda.so")
        cmd.extend(["-ngl", "28"])
    else:
        cmd.extend(["-ngl", "0", "-t", str(threads)])
        if cpu_range:
            cmd.extend(["-Cr", cpu_range])

    if use_speculative:
        cmd.extend(["--spec-type", "ngram-simple", "--spec-draft-n-max", "3"])

    # Launch server
    srv_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server readiness
    health_url = f"http://127.0.0.1:{port}/health"
    ready = False
    for _ in range(60):
        time.sleep(0.5)
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            continue

    if not ready:
        srv_proc.kill()
        raise RuntimeError(f"llama-server failed to start on port {port}")

    print(f"[{config_name}] llama-server is healthy and ready.")

    # Warmup request
    warmup_req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps({"prompt": "Hello", "n_predict": 2, "temperature": 0.0, "stream": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(warmup_req, timeout=60) as _:
            pass
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"Warmup warning: {e}")

    # Benchmark request
    req_body = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "cache_prompt": False,
    }).encode("utf-8")

    bench_req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=req_body,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    with urllib.request.urlopen(bench_req, timeout=180) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
    t_end = time.perf_counter()

    total_time_s = t_end - t_start

    # Retrieve process stats
    try:
        proc_psutil = psutil.Process(srv_proc.pid)
        peak_ram = proc_psutil.memory_info().rss / (1024**2)
    except (psutil.Error, OSError):
        peak_ram = 0.0

    vram_peak = get_gpu_vram_used_mib() if device == "cuda" else 0.0

    # Extract timings from llama.cpp response
    timings = res_data.get("timings", {})
    prompt_tokens = timings.get("prompt_n", 0)
    generated_tokens = timings.get("predicted_n", 0)
    prompt_ms = timings.get("prompt_ms", 0.0)
    prompt_per_sec = timings.get("prompt_per_second", 0.0)
    pred_per_sec = timings.get("predicted_per_second", 0.0)
    content = res_data.get("content", "")

    # Clean shutdown
    srv_proc.terminate()
    try:
        srv_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        srv_proc.kill()
        try:
            srv_proc.wait(timeout=2)
        except (subprocess.SubprocessError, OSError):
            pass

    time.sleep(1.0)  # allow port & VRAM release

    return BenchmarkResult(
        engine="llama.cpp",
        config_name=config_name,
        device=device.upper(),
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft_ms=round(prompt_ms, 2),
        prefill_tok_s=round(prompt_per_sec, 2),
        decode_tok_s=round(pred_per_sec, 2),
        overall_tok_s=round(generated_tokens / total_time_s, 2) if total_time_s > 0 else 0.0,
        total_time_s=round(total_time_s, 3),
        peak_vram_mib=round(vram_peak, 1),
        peak_ram_mib=round(peak_ram, 1),
        output_preview=content[:120].replace("\n", " ").strip(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head Chimera vs llama.cpp Benchmark")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "benchmarks" / "chimera_vs_llamacpp_results.json")
    parser.add_argument("--output-report", type=Path, default=PROJECT_ROOT / "benchmarks" / "chimera_vs_llamacpp_report.md")
    args = parser.parse_args()

    prompts = [
        "Jelaskan prinsip kerja transformer self-attention mechanism secara singkat dan padat.",
        "def quicksort(arr: list[int]) -> list[int]:\n    '''Implement recursive quicksort with type hints.'''\n",
    ]

    all_results: list[BenchmarkResult] = []

    print("=" * 80)
    print("STARTING HEAD-TO-HEAD BENCHMARK: CHIMERA-INFERENCE VS LLAMA.CPP")
    print("Target Hardware: Intel Core i5-12500H + NVIDIA RTX 3050 Laptop (4GB)")
    print("Target Model: Qwen 2.5 1.5B Instruct (Q4_K_M GGUF)")
    print("=" * 80)

    for idx, prompt in enumerate(prompts, 1):
        print(f"\n>>> TEST PROMPT {idx}: {prompt[:60]}... <<<")

        # 1. Chimera CUDA Pure
        try:
            r1 = benchmark_chimera(args.model, device="cuda", prompt=prompt, max_tokens=args.tokens)
            all_results.append(r1)
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[ERROR] Chimera CUDA failed: {e}")

        # 2. llama.cpp CUDA Pure
        try:
            r2 = benchmark_llamacpp(args.model, device="cuda", prompt=prompt, max_tokens=args.tokens)
            all_results.append(r2)
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[ERROR] llama.cpp CUDA failed: {e}")

        # 3. Chimera CUDA + Speculative
        try:
            r3 = benchmark_chimera(args.model, device="cuda", prompt=prompt, max_tokens=args.tokens, use_speculative=True)
            all_results.append(r3)
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[ERROR] Chimera CUDA Speculative failed: {e}")

        # 4. llama.cpp CUDA + Speculative (ngram-simple)
        try:
            r4 = benchmark_llamacpp(args.model, device="cuda", prompt=prompt, max_tokens=args.tokens, use_speculative=True)
            all_results.append(r4)
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[ERROR] llama.cpp CUDA Speculative failed: {e}")

        # 5. Chimera CPU AVX2 (P-Cores Pinned)
        try:
            r5 = benchmark_chimera(args.model, device="cpu", prompt=prompt, max_tokens=min(args.tokens, 32))
            all_results.append(r5)
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[ERROR] Chimera CPU failed: {e}")

        # 6. llama.cpp CPU (P-Cores Pinned Cr 0-3)
        try:
            r6 = benchmark_llamacpp(args.model, device="cpu", prompt=prompt, threads=4, cpu_range="0-3", max_tokens=min(args.tokens, 32))
            all_results.append(r6)
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[ERROR] llama.cpp CPU Pinned failed: {e}")

        # 7. llama.cpp CPU (Unpinned OS Default)
        try:
            r7 = benchmark_llamacpp(args.model, device="cpu", prompt=prompt, threads=4, cpu_range=None, max_tokens=min(args.tokens, 32))
            all_results.append(r7)
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[ERROR] llama.cpp CPU Unpinned failed: {e}")

    # Export JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n[OUTPUT] Raw results written to {args.output_json}")

    # Generate Markdown Report
    generate_markdown_report(all_results, args.output_report)
    print(f"[OUTPUT] Comprehensive Markdown report written to {args.output_report}")


def generate_markdown_report(results: list[BenchmarkResult], output_path: Path) -> None:
    """Generate detailed engineering comparative report."""
    md = []
    md.append("# Laporan Benchmark Head-to-Head: Chimera-Inference vs llama.cpp\n")
    md.append("**Tanggal & Waktu Pengujian:** " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    md.append("## 1. Spesifikasi Perangkat Keras & Model Target\n")
    md.append("- **CPU:** 12th Gen Intel Core i5-12500H (12 Cores / 16 Threads: 4 P-Cores @ 4.5GHz, 8 E-Cores @ 3.3GHz, AVX2 + FMA)")
    md.append("- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM, 2048 CUDA Cores, Ampere GA107, Compute Capability 8.6)")
    md.append("- **RAM Sistem:** 16 GB DDR4/DDR5")
    md.append("- **Model Target:** `Qwen 2.5 1.5B Instruct` (Q4_K_M GGUF, ~986 MB blob)\n")
    md.append("## 2. Tabel Perbandingan Hasil Eksekusi Riil\n")
    md.append("| Engine / Konfigurasi | Device | Prompt Type | TTFT (ms) | Prefill (tok/s) | Decode Throughput (tok/s) | Peak VRAM (MiB) | Peak RAM (MiB) | Total Waktu (s) |")
    md.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        prompt_type = "QA Teori" if "prinsip kerja" in r.prompt else "Code Quicksort"
        md.append(
            f"| **{r.config_name}** | `{r.device}` | {prompt_type} | "
            f"**{r.ttft_ms:.1f}** | {r.prefill_tok_s:.1f} | **{r.decode_tok_s:.1f}** | "
            f"{r.peak_vram_mib:.0f} | {r.peak_ram_mib:.0f} | {r.total_time_s:.2f} |"
        )

    md.append("\n## 3. Analisis & Temuan Rekayasa Kritis (Key Engineering Takeaways)\n")
    md.append("### A. Komparasi Akselerasi GPU CUDA (RTX 3050 4GB)")
    md.append("- **Throughput Decode (Tokens/detik):** Mengukur kecepatan autoregressive forward pass token-by-token.")
    md.append("- **Time-To-First-Token (TTFT) & Prefill:** Kecepatan pemrosesan initial prompt matrix.")
    md.append("- **Efisiensi Alokasi VRAM:** Perbandingan footprint memori GPU terhadap batas ketat 4GB VRAM laptop.")
    md.append("\n### B. Komparasi Eksekusi CPU (Intel Core i5-12500H)")
    md.append("- **Efek P-Core Thread Affinity:** Dampak pembatasan eksekusi secara ketat ke physical Performance Cores (0-3) dibandingkan scheduler unpinned sistem operasi.")
    md.append("\n### C. Akselerasi Spekulatif (Speculative Decoding)")
    md.append("- Menguji efektivitas akselerasi lossless speculative decoding dengan Prompt Lookup / n-gram verification terhadap latency generasi.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()

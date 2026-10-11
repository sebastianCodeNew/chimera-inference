"""End-to-End Latency & Throughput Benchmark: P-Core Pinning vs Unpinned GEMV.

Benchmarks standard projection shapes from Qwen 2.5 1.5B and Qwen 2.5 7B.
"""

import time

import numpy as np

from src.core_engine import ChimeraCoreEngine


def run_benchmark():
    engine = ChimeraCoreEngine()
    print("=" * 65)
    print("Chimera-Inference: Heterogeneous Architecture Latency Report")
    print(
        f"Detected P-Cores: {engine.p_cores_count} | Total Logical Cores: {engine.total_cores_count}"
    )
    print("=" * 65)

    test_shapes = [
        ("Qwen 2.5 1.5B Self-Attention Proj", 1536, 1536),
        ("Qwen 2.5 1.5B Gate/Up FFN Proj", 8960, 1536),
        ("Qwen 2.5 7B Gate/Up FFN Proj", 18944, 3584),
    ]

    for name, rows, cols in test_shapes:
        np.random.seed(42)
        W = np.random.randn(rows, cols).astype(np.float32)
        x = np.random.randn(cols).astype(np.float32)

        # Warmup
        engine.gemv_unpinned(W, x)
        engine.gemv_p_cores(W, x)

        n_iters = 30

        # Benchmark Unpinned (simulating llama.cpp default thread pool)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            engine.gemv_unpinned(W, x)
        t1 = time.perf_counter()
        unpinned_latency_ms = ((t1 - t0) / n_iters) * 1000.0

        # Benchmark P-Core Pinned (Chimera-Inference)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            engine.gemv_p_cores(W, x)
        t1 = time.perf_counter()
        p_core_latency_ms = ((t1 - t0) / n_iters) * 1000.0

        speedup = unpinned_latency_ms / p_core_latency_ms
        latency_reduction_pct = (
            1.0 - (p_core_latency_ms / unpinned_latency_ms)
        ) * 100.0

        print(f"\nLayer Shape: {name} [{rows} x {cols}]")
        print(f"  Unpinned (Default All-Cores): {unpinned_latency_ms:.3f} ms")
        print(f"  Chimera P-Cores Pinned:       {p_core_latency_ms:.3f} ms")
        print(
            f"  Speedup Factor:               {speedup:.2f}x ({latency_reduction_pct:+.1f}% faster)"
        )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    run_benchmark()

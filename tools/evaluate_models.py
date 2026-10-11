"""CLI Tool: Multi-Criteria Model Valuation and Speculative Decoding Evaluator.

Executes weighted valuation across SOTA open-weight LLMs and speculative drafting
configurations for the NVIDIA RTX 3050 (4GB) + Intel Core i5-12500H laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluator import (
    EvaluationScore,
    EvaluationWeights,
    ModelEvaluator,
)

DEFAULT_BLOB = Path(
    "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
)


def run_live_verification(blob_path: Path) -> dict[str, float] | None:
    """Run an empirical live hardware benchmark run on the local GPU."""
    if not blob_path.exists():
        print(
            f"[Live Benchmark] Warning: Blob {blob_path} not found. Skipping live probe."
        )
        return None

    try:
        from src.generator import ChimeraGenerator

        print("[Live Benchmark] Initializing local Chimera-Inference GPU engine...")
        t0 = time.perf_counter()
        generator = ChimeraGenerator.from_gguf(
            blob_path,
            n_layers=28,
            max_cached_layers=28,
            preload_layers=True,
            device="cuda",
        )
        t_init = time.perf_counter() - t0

        prompt = "Explain quantum entanglement briefly in one paragraph."
        tokens_received = 0
        t_first: float | None = None

        def callback(_tok: str) -> None:
            nonlocal tokens_received, t_first
            if t_first is None:
                t_first = time.perf_counter()
            tokens_received += 1

        # Warmup
        generator.generate("Hi", max_new_tokens=1)

        t_gen_start = time.perf_counter()
        generator.generate(prompt, max_new_tokens=16, streaming_callback=callback)
        t_gen_end = time.perf_counter()

        ttft_ms = (t_first - t_gen_start) * 1000.0 if t_first else 0.0
        decode_duration = (
            (t_gen_end - t_first) if t_first else (t_gen_end - t_gen_start)
        )
        tps = (
            (tokens_received - 1) / decode_duration
            if (decode_duration > 0 and tokens_received > 1)
            else 0.0
        )

        import torch

        vram_mb = (
            torch.cuda.memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        )

        print(
            f"[Live Benchmark] Success: Measured {tps:.2f} tok/s | TTFT: {ttft_ms:.1f} ms | VRAM: {vram_mb:.1f} MiB"
        )
        return {"tps": tps, "ttft_ms": ttft_ms, "vram_mb": vram_mb, "init_s": t_init}
    except (RuntimeError, OSError, ValueError) as err:
        print(f"[Live Benchmark] Failed: {err}")
        return None


def print_theory_section() -> None:
    """Print the journal and research-backed explanation of speculative decoding."""
    theory = ModelEvaluator.explain_speculative_phenomenon()
    print("\n" + "=" * 80)
    print("RESEARCH & JOURNAL ANALYSIS: SPECULATIVE DRAFTING DYNAMICS")
    print("=" * 80)
    print(
        f"\n1. MATHEMATICAL LOSSLESS SAMPLING GUARANTEE:\n   {theory['lossless_distribution_theorem']}\n"
    )
    print("2. ROOT CAUSES OF OBSERVED OUTPUT DEGRADATION & NEGATIVE SPEEDUP:")
    for idx, item in enumerate(theory["root_causes_of_output_degradation"], start=1):
        print(f"   [{idx}] {item['cause']} -> Impact: {item['impact']}")
        print(f"       Mechanism: {item['mechanism']}\n")
    if "reasoning_model_cot_dynamics" in theory:
        rcot = theory["reasoning_model_cot_dynamics"]
        print(
            f"3. REASONING & CHAIN-OF-THOUGHT SPECULATIVE DYNAMICS ({rcot['phenomenon']}):"
        )
        for idx, item in enumerate(rcot["root_causes_and_mechanisms"], start=1):
            print(f"   [{idx}] {item['issue']}:")
            print(f"       {item['mechanism']}\n")
    print(
        f"4. HARDWARE ENVELOPE MITIGATION STRATEGY (RTX 3050 4GB VRAM):\n   {theory['hardware_mitigation_strategy']}"
    )
    print("=" * 80 + "\n")


def print_math_analysis_section() -> None:
    """Display first-principles speculative decoding mathematics, break-even alpha, and optimal gamma."""
    print("\n" + "=" * 85)
    print(
        "SPECULATIVE SAMPLING MATHEMATICAL MODEL (Leviathan et al. 2023 / Chen et al. 2023)"
    )
    print("=" * 85)
    print(
        f"{'Lookahead (gamma)':<18} | {'Cost Ratio (c)':<25} | {'Break-Even Alpha (alpha*)':<25} | {'E[N] at alpha=0.70'}"
    )
    print("-" * 85)
    for g in [2, 3, 4, 5, 6]:
        for c, label in [
            (0.005, "0.005 (Prompt Lookup)"),
            (0.12, "0.12 (CPU AVX2)"),
            (0.22, "0.22 (Large Drafter)"),
        ]:
            be_alpha = ModelEvaluator.calculate_break_even_alpha(g, c)
            en = ModelEvaluator.calculate_expected_acceptance_length(0.70, g)
            print(f"{g:<18} | {label:<25} | {be_alpha:<25.4f} | {en:<.2f} tokens/step")
    print("=" * 85)
    print(
        "Note: If observed drafter acceptance rate alpha < alpha*, the system enters negative speedup (slowdown).\n"
    )


def print_cli_table(scores: list[EvaluationScore], preset_name: str) -> None:
    """Print an aligned ASCII terminal table of evaluation results."""
    print("\n" + "=" * 115)
    print(
        f"CHIMERA-INFERENCE: SOTA MODEL & SPECULATIVE DECISION MATRIX ({preset_name.upper()} PRESET)"
    )
    print(
        "Target Hardware: NVIDIA RTX 3050 Laptop (4GB GDDR6 VRAM) + Intel Core i5-12500H (12 Cores / 16 Threads)"
    )
    print("=" * 115)

    header = (
        f"{'Rk':<3} | {'Candidate / Speculative Pair':<38} | {'Intel':<5} | "
        f"{'Speed (TPS)':<11} | {'VRAM (MiB)':<10} | {'Value Score':<11} | {'Pareto?':<7} | {'Verdict'}"
    )
    print(header)
    print("-" * 115)

    for s in scores:
        p_badge = "SOTA *" if s.is_pareto_optimal else " "
        print(
            f"{s.rank:<3} | {s.display_name:<38} | {s.intel_score:<5.1f} | "
            f"{s.effective_tps:<11.1f} | {s.vram_used_mb:<10.0f} | "
            f"{s.total_score:<11.1f} | {p_badge:<7} | {s.verdict}"
        )
    print("=" * 115)
    print(
        "* SOTA = Pareto-Optimal Candidate (Non-dominated in Intelligence vs Speed vs VRAM budget)\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chimera-Inference: SOTA LLM & Speculative Synergy Valuation Framework"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=["balanced", "coding_reasoning", "high_throughput", "low_vram"],
        help="Evaluation weight profile preset",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="all",
        choices=["all", "standalone", "speculative", "pareto"],
        help="Filter candidates in report",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "markdown", "json"],
        help="Output presentation format",
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=None,
        help="Optional file path to write report (Markdown or JSON)",
    )
    parser.add_argument(
        "--vram-budget",
        type=float,
        default=4095.0,
        help="Custom VRAM budget in MiB (default: 4095.0)",
    )
    parser.add_argument(
        "--theory",
        action="store_true",
        help="Display peer-reviewed research and journal insights on speculative drafting",
    )
    parser.add_argument(
        "--math-analysis",
        action="store_true",
        help="Display break-even alpha, optimal lookahead gamma, and throughput curves",
    )
    parser.add_argument(
        "--synthesize-pair",
        nargs=2,
        metavar=("TARGET_ID", "DRAFTER_ID"),
        help="Dynamically evaluate and synthesize a custom target + drafter pair",
    )
    parser.add_argument(
        "--drafter-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "zero_flop"],
        help="Device placement for synthesized drafter (default: cpu)",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=4,
        help="Lookahead tokens gamma for speculative drafting (default: 4)",
    )
    parser.add_argument(
        "--live-verify",
        action="store_true",
        help="Run live verification against local GPU model blob",
    )
    parser.add_argument(
        "--blob",
        type=Path,
        default=DEFAULT_BLOB,
        help="Path to GGUF model blob for live verification",
    )
    parser.add_argument(
        "--quant-analysis",
        type=str,
        metavar="MODEL_ID",
        help="Evaluate a model across the entire quantization spectrum (BF16 down to Q2_K)",
    )

    args = parser.parse_args()

    if args.theory:
        print_theory_section()

    if args.math_analysis:
        print_math_analysis_section()

    if args.live_verify:
        run_live_verification(args.blob)

    weights = EvaluationWeights.get_preset(args.preset)
    weights.vram_budget_mb = args.vram_budget

    evaluator = ModelEvaluator(weights=weights)

    if args.quant_analysis:
        spectrum_scores = evaluator.evaluate_quantization_spectrum(args.quant_analysis)
        target_model = evaluator.get_model(args.quant_analysis)
        print(
            "\n"
            + evaluator.format_quantization_spectrum_markdown(
                spectrum_scores, target_model.name
            )
            + "\n"
        )
        return

    if args.synthesize_pair:
        target_id, drafter_id = args.synthesize_pair
        drafter_arg = (
            None
            if drafter_id.lower() in ("none", "lookup", "prompt_lookup")
            else drafter_id
        )
        drafter_type = "prompt_lookup" if drafter_arg is None else "model"
        pair = evaluator.synthesize_pair(
            target_id=target_id,
            drafter_id=drafter_arg,
            drafter_type=drafter_type,
            drafter_device=args.drafter_device,
            gamma=args.gamma,
        )
        print(f"[Synthesized Pair] Created: {pair.pair_id}")
        print(f"  Target: {pair.target.name} | Drafter: {pair.drafter_name}")
        print(
            f"  Expected alpha: {pair.expected_alpha:.2f} | Speedup: {pair.speedup_ratio:.2f}x | Effective TPS: {pair.effective_tps:.1f}"
        )
        print(
            f"  VRAM: {pair.combined_vram_mb:.0f} MiB | Vocab Aligned: {pair.vocab_aligned} | Quality: {pair.quality_retention:.2f}"
        )

    scores = evaluator.evaluate_all(filter_type=args.filter)

    if args.format == "table":
        print_cli_table(scores, args.preset)
    elif args.format == "markdown":
        md_content = evaluator.format_comparison_markdown(
            scores, preset_name=args.preset
        )
        print(md_content)
    elif args.format == "json":
        json_data = evaluator.export_json(scores)
        print(json.dumps(json_data, indent=2))

    if args.export_path:
        args.export_path.parent.mkdir(parents=True, exist_ok=True)
        if args.export_path.suffix.lower() == ".json":
            json_data = evaluator.export_json(scores)
            with args.export_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
            print(f"[Export] Saved JSON benchmark report to: {args.export_path}")
        else:
            md_content = evaluator.format_comparison_markdown(
                scores, preset_name=args.preset
            )
            with args.export_path.open("w", encoding="utf-8") as f:
                f.write(md_content)
            print(f"[Export] Saved Markdown benchmark report to: {args.export_path}")


if __name__ == "__main__":
    main()

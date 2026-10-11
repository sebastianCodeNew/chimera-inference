"""Empirical Evaluation and MCDA Analysis for IFM/K2-Horizon-7B."""

from __future__ import annotations

from src.model_evaluator import (
    EvaluationWeights,
    ModelEvaluator,
    ModelFamily,
    ModelSpecs,
)


def run_k2_evaluation() -> None:
    evaluator = ModelEvaluator()

    m_k2_7b = ModelSpecs(
        model_id="k2-horizon-7b-q4",
        name="K2-Horizon-7B",
        family=ModelFamily.OTHER,
        parameter_count_b=8.999,
        quantization="Q4_K_M",
        vocab_size=250624,
        chat_template="k2_horizon",
        context_length=524288,
        weights_size_mb=5592.2,
        runtime_vram_mb=4095.0,
        runtime_ram_mb=5850.0,
        device_allocation="hybrid_cpu_gpu",
        standalone_tps=5.4,
        ttft_ms=790.0,
        mmlu=73.8,
        gsm8k=94.4,
        humaneval=87.2,
        arena_hard=82.5,
        is_fine_tuned=True,
        specialization="reasoning_coding_agentic",
        description="IFM 512K dense reasoning model (9.0B total params); exceptional math/coding capabilities, requires hybrid offload on 4GB VRAM.",
    )

    # 1. Standalone
    score_standalone = evaluator.score_standalone(m_k2_7b)
    print("=== 1. K2-Horizon-7B Standalone ===")
    print(f"Intel Score: {score_standalone.intel_score:.1f}")
    print(
        f"Speed Score: {score_standalone.speed_score:.1f} (TPS: {m_k2_7b.standalone_tps:.1f})"
    )
    print(
        f"Memory Score: {score_standalone.mem_score:.1f} (VRAM: {score_standalone.vram_used_mb:.0f} MiB)"
    )
    print(f"Total Value Score (Balanced): {score_standalone.total_score:.1f}")
    print(f"Verdict: {score_standalone.verdict}")
    print(f"Pros: {score_standalone.pros}")
    print(f"Cons: {score_standalone.cons}")

    evaluator.register_model(m_k2_7b)

    # 2. Speculative Pair: Prompt Lookup
    p_k2_lookup = evaluator.synthesize_pair(
        target_id=m_k2_7b.model_id,
        drafter_id=None,
        drafter_type="prompt_lookup",
        drafter_device="zero_flop",
        gamma=4,
        custom_alpha=0.52,
        notes="Zero-FLOP prompt lookup on K2-Horizon-7B",
    )
    score_lookup = evaluator.score_speculative_pair(p_k2_lookup)
    print("\n=== 2. K2-Horizon-7B + Prompt Lookup (Zero-FLOP) ===")
    print(
        f"Effective TPS: {score_lookup.effective_tps:.1f} (Speedup: {score_lookup.speedup_ratio:.2f}x)"
    )
    print(f"Total Value Score (Balanced): {score_lookup.total_score:.1f}")
    print(f"Verdict: {score_lookup.verdict}")
    print(f"Pros: {score_lookup.pros}")

    # 3. Speculative Pair: K2-Horizon 0.9B (Vocabulary Mismatch)
    m_k2_09b = ModelSpecs(
        model_id="k2-horizon-0.9b-q4",
        name="K2-Horizon-0.9B",
        family=ModelFamily.OTHER,
        parameter_count_b=0.95,
        quantization="Q4_K_M",
        vocab_size=64256,
        chat_template="k2_horizon",
        context_length=524288,
        weights_size_mb=720.0,
        runtime_vram_mb=1100.0,
        runtime_ram_mb=1400.0,
        device_allocation="cuda",
        standalone_tps=45.0,
        ttft_ms=65.0,
        mmlu=58.0,
        gsm8k=65.0,
        humaneval=52.0,
        arena_hard=48.0,
    )
    evaluator.register_model(m_k2_09b)
    p_k2_09b = evaluator.synthesize_pair(
        target_id=m_k2_7b.model_id,
        drafter_id=m_k2_09b.model_id,
        drafter_type="autoregressive_drafter",
        drafter_device="cpu",
        gamma=4,
        notes="Mismatched Vocab (64k vs 250k) - Degraded",
    )
    score_mismatched = evaluator.score_speculative_pair(p_k2_09b)
    print("\n=== 3. K2-Horizon-7B + K2-0.9B (Mismatched Vocab 64k vs 250k) ===")
    print(f"Alpha: {p_k2_09b.expected_alpha:.2f}")
    print(
        f"Speedup: {p_k2_09b.speedup_ratio:.2f}x (Effective TPS: {score_mismatched.effective_tps:.1f})"
    )
    print(f"Total Value Score: {score_mismatched.total_score:.1f}")
    print(f"Verdict: {score_mismatched.verdict}")
    print(f"Cons: {score_mismatched.cons}")

    # 4. Evaluate under Coding & Reasoning preset
    evaluator_coding = ModelEvaluator(
        weights=EvaluationWeights.get_preset("coding_reasoning")
    )
    score_k2_coding = evaluator_coding.score_standalone(m_k2_7b)
    score_lookup_coding = evaluator_coding.score_speculative_pair(p_k2_lookup)
    print(
        "\n=== 4. Scores under Coding & Reasoning Preset (High Weight on Intelligence) ==="
    )
    print(f"K2-Horizon-7B Standalone Value: {score_k2_coding.total_score:.1f}")
    print(f"K2-Horizon-7B + Prompt Lookup Value: {score_lookup_coding.total_score:.1f}")


if __name__ == "__main__":
    run_k2_evaluation()

"""Unit and integration tests for Model Evaluator and Speculative Synergy Framework."""

from __future__ import annotations

import pytest

from src.model_evaluator import (
    EvaluationScore,
    EvaluationWeights,
    ModelEvaluator,
    ModelFamily,
    ModelSpecs,
)


def test_model_specs_initialization() -> None:
    """Verify ModelSpecs dataclass creation and computed properties."""
    specs = ModelSpecs(
        model_id="test-model",
        name="Test Model 1B",
        family=ModelFamily.QWEN2_5,
        parameter_count_b=1.0,
        quantization="Q4_K_M",
        vocab_size=151936,
        chat_template="chatml",
        context_length=8192,
        weights_size_mb=600.0,
        runtime_vram_mb=1200.0,
        runtime_ram_mb=1500.0,
        device_allocation="cuda",
        standalone_tps=30.0,
        ttft_ms=100.0,
        mmlu=60.0,
        gsm8k=70.0,
        humaneval=50.0,
        arena_hard=60.0,
    )

    assert specs.model_id == "test-model"
    assert specs.composite_intelligence_raw == pytest.approx(60.0)
    assert specs.is_fine_tuned is False


def test_evaluation_weights_normalization() -> None:
    """Verify that EvaluationWeights auto-normalizes non-unit weights."""
    weights = EvaluationWeights(
        w_intel=2.0,
        w_speed=2.0,
        w_mem=2.0,
        w_spec=2.0,
        w_mmlu=1.0,
        w_gsm8k=1.0,
        w_humaneval=1.0,
        w_arena_hard=1.0,
    )

    assert weights.w_intel == pytest.approx(0.25)
    assert weights.w_speed == pytest.approx(0.25)
    assert weights.w_mem == pytest.approx(0.25)
    assert weights.w_spec == pytest.approx(0.25)

    assert weights.w_mmlu == pytest.approx(0.25)
    assert weights.w_gsm8k == pytest.approx(0.25)
    assert weights.w_humaneval == pytest.approx(0.25)
    assert weights.w_arena_hard == pytest.approx(0.25)


def test_evaluation_weights_zero_fallback() -> None:
    """Verify that zero weights fallback to reasonable defaults without division by zero."""
    weights = EvaluationWeights(
        w_intel=0.0,
        w_speed=0.0,
        w_mem=0.0,
        w_spec=0.0,
        w_mmlu=0.0,
        w_gsm8k=0.0,
        w_humaneval=0.0,
        w_arena_hard=0.0,
    )
    assert weights.w_intel == pytest.approx(0.35)
    assert weights.w_speed == pytest.approx(0.30)
    assert weights.w_mem == pytest.approx(0.20)
    assert weights.w_spec == pytest.approx(0.15)


def test_evaluation_weights_presets() -> None:
    """Test standard presets retrieval and validation."""
    balanced = EvaluationWeights.get_preset("balanced")
    assert balanced.w_intel == pytest.approx(0.35)

    coding = EvaluationWeights.get_preset("coding_reasoning")
    assert coding.w_intel == pytest.approx(0.50)
    assert coding.w_gsm8k >= 0.35
    assert coding.w_humaneval >= 0.35

    speed = EvaluationWeights.get_preset("high_throughput")
    assert speed.w_speed == pytest.approx(0.50)

    vram = EvaluationWeights.get_preset("low_vram")
    assert vram.w_mem == pytest.approx(0.40)

    with pytest.raises(ValueError, match="Unknown preset"):
        EvaluationWeights.get_preset("non_existent_preset")


def test_compute_intel_score() -> None:
    """Test intelligence scoring formula and quality retention degradation."""
    evaluator = ModelEvaluator()

    perfect = evaluator.compute_intel_score(100.0, 100.0, 100.0, 100.0)
    assert perfect == pytest.approx(100.0)

    zero = evaluator.compute_intel_score(0.0, 0.0, 0.0, 0.0)
    assert zero == pytest.approx(0.0)

    # Test quality retention penalty (e.g. cross-family draft drift)
    degraded = evaluator.compute_intel_score(
        100.0, 100.0, 100.0, 100.0, quality_retention=0.5
    )
    assert degraded == pytest.approx(50.0)


def test_compute_speed_score() -> None:
    """Test speed score with interactive thresholds and TTFT latency."""
    evaluator = ModelEvaluator()

    # Fast model with low TTFT
    fast_score = evaluator.compute_speed_score(tps=45.0, ttft_ms=50.0)
    assert fast_score >= 95.0

    # Slow model below interactive min
    slow_score = evaluator.compute_speed_score(tps=2.0, ttft_ms=800.0)
    assert slow_score < 20.0


def test_compute_mem_score() -> None:
    """Test memory evaluation against 4GB VRAM budget."""
    evaluator = ModelEvaluator()

    # Model comfortably within VRAM (< 3400 MiB)
    safe_mem, fits, headroom = evaluator.compute_mem_score(
        vram_used_mb=2000.0, ram_used_mb=3000.0
    )
    assert fits is True
    assert headroom == pytest.approx(2095.0)
    assert safe_mem >= 85.0

    # Model in tight VRAM zone (3400 - 4095 MiB)
    tight_mem, fits_tight, headroom_tight = evaluator.compute_mem_score(
        vram_used_mb=3800.0, ram_used_mb=4000.0
    )
    assert fits_tight is True
    assert headroom_tight == pytest.approx(295.0)
    assert 50.0 <= tight_mem <= 85.0

    # Model exceeding 4GB VRAM
    overflow_mem, fits_over, headroom_over = evaluator.compute_mem_score(
        vram_used_mb=5000.0, ram_used_mb=5000.0
    )
    assert fits_over is False
    assert headroom_over < 0.0
    assert overflow_mem < 45.0


def test_compute_spec_score() -> None:
    """Test speculative synergy score, alignment bonuses, and negative speedup penalties."""
    evaluator = ModelEvaluator()

    # Standalone baseline
    standalone = evaluator.compute_spec_score(
        is_pair=False,
        vocab_aligned=True,
        template_aligned=True,
        alpha=0.0,
        speedup=1.0,
        quality_retention=1.0,
    )
    assert standalone == pytest.approx(50.0)

    # High-synergy lossless pair
    high_synergy = evaluator.compute_spec_score(
        is_pair=True,
        vocab_aligned=True,
        template_aligned=True,
        alpha=0.75,
        speedup=1.5,
        quality_retention=1.0,
    )
    assert high_synergy >= 90.0

    # Mismatched vocabulary
    mismatched_vocab = evaluator.compute_spec_score(
        is_pair=True,
        vocab_aligned=False,
        template_aligned=True,
        alpha=0.50,
        speedup=1.0,
        quality_retention=0.6,
    )
    assert mismatched_vocab < 15.0

    # Negative speedup (slowdown) quadratic penalty
    slowdown = evaluator.compute_spec_score(
        is_pair=True,
        vocab_aligned=True,
        template_aligned=True,
        alpha=0.20,
        speedup=0.70,
        quality_retention=1.0,
    )
    # 100 * 0.20 * 1.0 * 1.0 * (0.7^2) = 20 * 0.49 = 9.8
    assert slowdown == pytest.approx(9.8, abs=0.5)


def test_evaluate_standalone_model() -> None:
    """Test end-to-end scoring of a standalone model."""
    evaluator = ModelEvaluator()
    model = evaluator.get_model("qwen2.5-1.5b-instruct-q4")
    score = evaluator.score_standalone(model)

    assert isinstance(score, EvaluationScore)
    assert score.is_speculative_pair is False
    assert score.fits_vram is True
    assert score.vram_headroom_mb > 1000.0
    assert score.effective_tps == pytest.approx(28.4)
    assert score.total_score > 65.0


def test_evaluate_speculative_pair() -> None:
    """Test end-to-end scoring of a high-synergy speculative pair."""
    evaluator = ModelEvaluator()
    pair = evaluator.get_pair("pair-qwen0.5b-cpu-qwen1.5b-cuda")
    score = evaluator.score_speculative_pair(pair)

    assert isinstance(score, EvaluationScore)
    assert score.is_speculative_pair is True
    assert score.fits_vram is True
    assert score.speedup_ratio == pytest.approx(1.46)
    assert score.effective_tps == pytest.approx(41.5)
    assert "High-Synergy" in score.verdict
    assert score.total_score > 80.0


def test_mismatched_pair_toxic_verdict() -> None:
    """Test that mismatched cross-family speculative pair gets flagged as degraded."""
    evaluator = ModelEvaluator()
    mismatched = evaluator.get_pair("pair-mismatched-qwen0.5b-llama3b")
    score = evaluator.score_speculative_pair(mismatched)

    assert "Degraded / Toxic" in score.verdict
    assert score.speedup_ratio < 1.0
    assert any("Vocabulary mismatch" in c for c in score.cons)
    assert any("Negative speedup" in c for c in score.cons)


def test_evaluate_all_catalog() -> None:
    """Test evaluating all models and speculative pairs in the catalog."""
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all()

    assert len(results) >= 15
    # Verify results are sorted descending by total_score
    for i in range(len(results) - 1):
        assert results[i].total_score >= results[i + 1].total_score
        assert results[i].rank == i + 1

    # Verify #1 rank is a high-synergy pair
    top = results[0]
    assert top.is_speculative_pair is True
    assert top.is_pareto_optimal is True


def test_pareto_frontier_logic() -> None:
    """Test Pareto optimality identification."""
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all()

    pareto_candidates = [s for s in results if s.is_pareto_optimal]
    assert len(pareto_candidates) >= 4

    # The fastest model (Qwen 0.5B standalone) must be Pareto optimal
    fastest = next(s for s in results if s.candidate_id == "qwen2.5-0.5b-instruct-q4")
    assert fastest.is_pareto_optimal is True

    # The top speculative pair must be Pareto optimal
    top_pair = results[0]
    assert top_pair.is_pareto_optimal is True


def test_explain_speculative_phenomenon() -> None:
    """Verify that explain_speculative_phenomenon returns rigorous theoretical structure."""
    theory = ModelEvaluator.explain_speculative_phenomenon()

    assert "lossless_distribution_theorem" in theory
    assert "Leviathan et al. (2023)" in theory["lossless_distribution_theorem"]
    assert "root_causes_of_output_degradation" in theory
    assert len(theory["root_causes_of_output_degradation"]) >= 4
    assert "hardware_mitigation_strategy" in theory
    assert "RTX 3050" in theory["hardware_mitigation_strategy"]


def test_markdown_and_json_export() -> None:
    """Verify Markdown and JSON export formatting."""
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all()

    md = evaluator.format_comparison_markdown(results, preset_name="balanced")
    assert "# Chimera-Inference" in md
    assert "RTX 3050" in md
    assert "| Rank |" in md

    json_dict = evaluator.export_json(results)
    assert "evaluation_weights" in json_dict
    assert "speculative_theory" in json_dict
    assert "rankings" in json_dict
    assert len(json_dict["rankings"]) == len(results)


def test_catalog_key_error_handling() -> None:
    """Verify proper exceptions on missing models or pairs."""
    evaluator = ModelEvaluator()

    with pytest.raises(KeyError, match="Model 'unknown-model' not found"):
        evaluator.get_model("unknown-model")

    with pytest.raises(KeyError, match="Pair 'unknown-pair' not found"):
        evaluator.get_pair("unknown-pair")


def test_speculative_math_expected_acceptance_length() -> None:
    """Verify expected acceptance length formula across edge cases."""
    # Boundary: gamma <= 0
    assert ModelEvaluator.calculate_expected_acceptance_length(0.7, 0) == pytest.approx(
        1.0
    )
    # Boundary: alpha = 0.0 -> exactly 1 token
    assert ModelEvaluator.calculate_expected_acceptance_length(0.0, 4) == pytest.approx(
        1.0
    )
    # Boundary: alpha = 1.0 -> exactly gamma + 1 tokens
    assert ModelEvaluator.calculate_expected_acceptance_length(1.0, 4) == pytest.approx(
        5.0
    )
    # General: alpha = 0.70, gamma = 4 -> (1 - 0.7^5) / 0.3 = 2.7731
    expected = (1.0 - 0.70**5) / (1.0 - 0.70)
    assert ModelEvaluator.calculate_expected_acceptance_length(
        0.70, 4
    ) == pytest.approx(expected)


def test_speculative_math_theoretical_speedup() -> None:
    """Verify speedup calculation for high-synergy and negative-speedup regimes."""
    # Zero-FLOP prompt lookup (cost_ratio ~ 0.005, alpha = 0.50, gamma = 3)
    sp_lookup = ModelEvaluator.calculate_theoretical_speedup(
        alpha=0.50, gamma=3, cost_ratio=0.005, verification_overhead=0.05
    )
    assert sp_lookup > 1.6

    # Severe mismatch / underpowered drafter (alpha = 0.15, gamma = 4, cost_ratio = 0.25)
    sp_negative = ModelEvaluator.calculate_theoretical_speedup(
        alpha=0.15, gamma=4, cost_ratio=0.25, verification_overhead=0.08
    )
    assert sp_negative < 1.0


def test_speculative_math_break_even_alpha() -> None:
    """Verify numerical solver for break-even alpha."""
    alpha_star = ModelEvaluator.calculate_break_even_alpha(
        gamma=4, cost_ratio=0.12, verification_overhead=0.08
    )
    sp_at_star = ModelEvaluator.calculate_theoretical_speedup(
        alpha=alpha_star, gamma=4, cost_ratio=0.12, verification_overhead=0.08
    )
    assert sp_at_star == pytest.approx(1.0, abs=0.02)


def test_speculative_math_optimal_gamma() -> None:
    """Verify lookahead search finds an optimal lookahead."""
    opt_gamma, opt_speedup = ModelEvaluator.calculate_optimal_gamma(
        alpha=0.75, cost_ratio=0.10, max_gamma=8
    )
    assert 2 <= opt_gamma <= 6
    assert opt_speedup > 1.4


def test_dynamic_pair_synthesis() -> None:
    """Test synthesize_pair method for automated alignment and memory computation."""
    evaluator = ModelEvaluator()

    # Same-family pair
    pair = evaluator.synthesize_pair(
        target_id="qwen2.5-1.5b-instruct-q4",
        drafter_id="qwen2.5-0.5b-instruct-q4",
        drafter_device="cpu",
        gamma=4,
    )
    assert pair.vocab_aligned is True
    assert pair.template_aligned is True
    assert pair.combined_vram_mb == pytest.approx(3030.0)
    assert pair.combined_ram_mb == pytest.approx(1800.0 + 1100.0)
    assert pair.speedup_ratio > 1.2
    assert pair.quality_retention == pytest.approx(1.0)

    # Prompt lookup synthesis
    p_lookup = evaluator.synthesize_pair(
        target_id="qwen2.5-3b-instruct-q4",
        drafter_type="prompt_lookup",
        gamma=3,
    )
    assert p_lookup.drafter_type == "prompt_lookup"
    assert p_lookup.drafter_device == "zero_flop"
    assert p_lookup.combined_vram_mb == pytest.approx(3650.0)
    assert p_lookup.speedup_ratio > 1.2

    # Reasoning target with base drafter
    p_reasoning = evaluator.synthesize_pair(
        target_id="deepseek-r1-distill-qwen-1.5b-q4",
        drafter_id="qwen2.5-0.5b-instruct-q4",
        drafter_device="cpu",
    )
    assert p_reasoning.expected_alpha < 0.65


def test_evaluate_all_filter_types() -> None:
    """Test filtering evaluate_all by standalone, speculative, and pareto."""
    evaluator = ModelEvaluator()

    all_scores = evaluator.evaluate_all(filter_type="all")
    standalone = evaluator.evaluate_all(filter_type="standalone")
    speculative = evaluator.evaluate_all(filter_type="speculative")
    pareto = evaluator.evaluate_all(filter_type="pareto")

    assert len(all_scores) == len(standalone) + len(speculative)
    assert all(not s.is_speculative_pair for s in standalone)
    assert all(s.is_speculative_pair for s in speculative)
    assert all(s.is_pareto_optimal for s in pareto)
    assert len(pareto) >= 5


def test_expanded_catalog_coverage() -> None:
    """Verify newly added SOTA models and pairs are properly registered."""
    evaluator = ModelEvaluator()

    # Check DeepSeek-R1 Llama 8B
    m_r1_llama = evaluator.get_model("deepseek-r1-distill-llama-8b-q4")
    assert m_r1_llama.gsm8k >= 89.0
    assert m_r1_llama.family == ModelFamily.DEEPSEEK_R1_LLAMA

    # Check Gemma pair
    p_gemma = evaluator.get_pair("pair-prompt-lookup-gemma2b")
    assert p_gemma.target.family == ModelFamily.GEMMA2
    assert p_gemma.effective_tps > p_gemma.target.standalone_tps

    # Check DeepSeek 1.5B reasoning pairs
    p_deepseek_lookup = evaluator.get_pair("pair-prompt-lookup-deepseek1.5b")
    assert p_deepseek_lookup.target.model_id == "deepseek-r1-distill-qwen-1.5b-q4"
    assert p_deepseek_lookup.speedup_ratio > 1.2

    # Check K2-Horizon 7B and pairs
    m_k2 = evaluator.get_model("k2-horizon-7b-q4")
    assert m_k2.family == ModelFamily.K2_HORIZON
    assert m_k2.context_length == 524288
    assert m_k2.gsm8k >= 90.0

    p_k2_lookup = evaluator.get_pair("pair-prompt-lookup-k2-7b")
    assert p_k2_lookup.speedup_ratio > 1.2
    assert p_k2_lookup.vocab_aligned is True

    p_k2_mismatched = evaluator.get_pair("pair-mismatched-k2-0.9b-7b")
    assert p_k2_mismatched.speedup_ratio < 1.0
    assert p_k2_mismatched.vocab_aligned is False


def test_quantization_spectrum_analysis() -> None:
    """Verify quantization spectrum evaluation computes correct pareto and allocation regimes."""
    evaluator = ModelEvaluator()
    scores = evaluator.evaluate_quantization_spectrum("qwen2.5-1.5b-instruct-q4")

    assert len(scores) == 8
    q4_score = next(s for s in scores if "Q4_K_M" in s.display_name)
    bf16_score = next(s for s in scores if "BF16" in s.display_name)

    # Q4_K_M must outperform BF16 on value score due to 4GB VRAM safety buffer
    assert q4_score.total_score > bf16_score.total_score
    assert q4_score.vram_used_mb < bf16_score.vram_used_mb
    assert q4_score.effective_tps > bf16_score.effective_tps

    # Test K2-Horizon-7B spectrum: BF16 must trigger OOM/RAM exhaustion while Q4_K_M is viable hybrid
    k2_scores = evaluator.evaluate_quantization_spectrum("k2-horizon-7b-q4")
    k2_bf16 = next(s for s in k2_scores if "BF16" in s.display_name)
    k2_q4 = next(s for s in k2_scores if "Q4_K_M" in s.display_name)

    assert k2_bf16.fits_vram is False
    assert "OOM" in k2_bf16.verdict
    assert k2_q4.fits_vram is True
    assert k2_q4.intel_score > 80.0

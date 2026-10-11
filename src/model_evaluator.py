"""Weighted Model Valuation and Speculative Synergy Framework.

Provides rigorous multi-criteria decision analysis (MCDA) to evaluate LLMs and
heterogeneous speculative drafting pairs on resource-constrained hardware
(e.g., NVIDIA RTX 3050 4GB VRAM + Intel i5-12500H P-core AVX2).

Integrates foundational speculative decoding mathematics (Leviathan et al. 2023,
Chen et al. 2023) to analyze token acceptance rates, negative speedup regimes,
and distribution retention across same-family vs cross-family configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelFamily(str, Enum):
    """Model architectural and tokenizer family."""

    QWEN2_5 = "qwen2.5"
    DEEPSEEK_R1_QWEN = "deepseek-r1-qwen"
    LLAMA3 = "llama3"
    DEEPSEEK_R1_LLAMA = "deepseek-r1-llama"
    GEMMA2 = "gemma2"
    PROMPT_LOOKUP = "prompt_lookup"
    TINYLLAMA = "tinyllama"
    K2_HORIZON = "k2_horizon"
    OTHER = "other"


class QuantizationType(str, Enum):
    """Standard LLM weight quantization formats."""

    BF16 = "BF16"
    FP16 = "FP16"
    FP8 = "FP8"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q5_K_M = "Q5_K_M"
    Q5_0 = "Q5_0"
    Q4_K_M = "Q4_K_M"
    Q4_0 = "Q4_0"
    Q3_K_M = "Q3_K_M"
    Q2_K = "Q2_K"


@dataclass(frozen=True)
class QuantizationProfile:
    """Empirical characteristics of a quantization format."""

    format: QuantizationType
    bits_per_weight: float
    quality_retention: float
    relative_size_ratio: float
    description: str


@dataclass(frozen=True)
class ModelSpecs:
    """Architectural specifications and empirical benchmark metrics for an LLM."""

    model_id: str
    name: str
    family: ModelFamily
    parameter_count_b: float
    quantization: str
    vocab_size: int
    chat_template: str
    context_length: int
    weights_size_mb: float
    runtime_vram_mb: float
    runtime_ram_mb: float
    device_allocation: str
    standalone_tps: float
    ttft_ms: float
    mmlu: float
    gsm8k: float
    humaneval: float
    arena_hard: float
    is_fine_tuned: bool = False
    specialization: str = "general"
    description: str = ""

    @property
    def composite_intelligence_raw(self) -> float:
        """Unweighted average intelligence benchmark score."""
        return (self.mmlu + self.gsm8k + self.humaneval + self.arena_hard) / 4.0


@dataclass(frozen=True)
class SpeculativePairSpecs:
    """Specifications for a speculative decoding draft-target configuration."""

    pair_id: str
    target: ModelSpecs
    drafter: ModelSpecs | None
    drafter_name: str
    drafter_type: str
    drafter_device: str
    combined_vram_mb: float
    combined_ram_mb: float
    vocab_aligned: bool
    template_aligned: bool
    expected_alpha: float
    gamma: int
    effective_tps: float
    speedup_ratio: float
    quality_retention: float
    distribution_mode: str
    notes: str = ""


@dataclass
class EvaluationWeights:
    """Configurable weights and constraints for multi-criteria valuation."""

    w_intel: float = 0.35
    w_speed: float = 0.30
    w_mem: float = 0.20
    w_spec: float = 0.15

    w_mmlu: float = 0.25
    w_gsm8k: float = 0.30
    w_humaneval: float = 0.30
    w_arena_hard: float = 0.15

    vram_budget_mb: float = 4095.0
    vram_safe_headroom_mb: float = 700.0
    ram_budget_mb: float = 16384.0
    tps_target: float = 35.0
    tps_min: float = 4.0
    ttft_target_ms: float = 80.0
    ttft_max_ms: float = 750.0

    def __post_init__(self) -> None:
        total_p = self.w_intel + self.w_speed + self.w_mem + self.w_spec
        if not (0.999 <= total_p <= 1.001):
            if total_p <= 0:
                self.w_intel, self.w_speed, self.w_mem, self.w_spec = (
                    0.35,
                    0.30,
                    0.20,
                    0.15,
                )
            else:
                self.w_intel /= total_p
                self.w_speed /= total_p
                self.w_mem /= total_p
                self.w_spec /= total_p

        total_sub = self.w_mmlu + self.w_gsm8k + self.w_humaneval + self.w_arena_hard
        if not (0.999 <= total_sub <= 1.001):
            if total_sub <= 0:
                self.w_mmlu, self.w_gsm8k, self.w_humaneval, self.w_arena_hard = (
                    0.25,
                    0.30,
                    0.30,
                    0.15,
                )
            else:
                self.w_mmlu /= total_sub
                self.w_gsm8k /= total_sub
                self.w_humaneval /= total_sub
                self.w_arena_hard /= total_sub

    @classmethod
    def get_preset(cls, preset_name: str) -> EvaluationWeights:
        """Retrieve standard evaluation profile preset."""
        p = preset_name.lower().strip()
        if p in ("balanced", "default"):
            return cls(w_intel=0.35, w_speed=0.30, w_mem=0.20, w_spec=0.15)
        if p in ("coding_reasoning", "coding", "reasoning"):
            return cls(
                w_intel=0.50,
                w_speed=0.20,
                w_mem=0.15,
                w_spec=0.15,
                w_mmlu=0.15,
                w_gsm8k=0.40,
                w_humaneval=0.40,
                w_arena_hard=0.05,
            )
        if p in ("high_throughput", "speed", "fast"):
            return cls(w_intel=0.20, w_speed=0.50, w_mem=0.15, w_spec=0.15)
        if p in ("low_vram", "memory", "vram_safe"):
            return cls(w_intel=0.25, w_speed=0.25, w_mem=0.40, w_spec=0.10)
        raise ValueError(
            f"Unknown preset: {preset_name}. Choose from: balanced, coding_reasoning, high_throughput, low_vram"
        )


@dataclass
class EvaluationScore:
    """Normalized scoring result for a single candidate or speculative pair."""

    candidate_id: str
    display_name: str
    is_speculative_pair: bool
    target_name: str
    drafter_name: str
    intel_score: float
    speed_score: float
    mem_score: float
    spec_score: float
    total_score: float
    fits_vram: bool
    vram_used_mb: float
    vram_headroom_mb: float
    effective_tps: float
    speedup_ratio: float
    quality_retention: float
    is_pareto_optimal: bool = False
    rank: int = 0
    verdict: str = ""
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)


class ModelEvaluator:
    """Production evaluation and benchmark engine for edge and laptop inference."""

    def __init__(self, weights: EvaluationWeights | None = None) -> None:
        self.weights = weights if weights is not None else EvaluationWeights()
        self._models: dict[str, ModelSpecs] = {}
        self._pairs: dict[str, SpeculativePairSpecs] = {}
        self._populate_default_catalog()

    def register_model(self, model: ModelSpecs) -> None:
        """Register an LLM into the evaluation catalog."""
        self._models[model.model_id] = model

    def register_pair(self, pair: SpeculativePairSpecs) -> None:
        """Register a speculative decoding draft-target configuration."""
        self._pairs[pair.pair_id] = pair

    def get_model(self, model_id: str) -> ModelSpecs:
        """Retrieve model specifications by identifier."""
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found in catalog.")
        return self._models[model_id]

    def get_pair(self, pair_id: str) -> SpeculativePairSpecs:
        """Retrieve speculative pair specifications by identifier."""
        if pair_id not in self._pairs:
            raise KeyError(f"Pair '{pair_id}' not found in catalog.")
        return self._pairs[pair_id]

    def compute_intel_score(
        self,
        mmlu: float,
        gsm8k: float,
        humaneval: float,
        arena_hard: float,
        quality_retention: float = 1.0,
    ) -> float:
        """Compute weighted intelligence score normalized to [0, 100]."""
        w = self.weights
        composite = (
            w.w_mmlu * mmlu
            + w.w_gsm8k * gsm8k
            + w.w_humaneval * humaneval
            + w.w_arena_hard * arena_hard
        )
        scaled = max(0.0, min(100.0, composite))
        return float(scaled * max(0.0, min(1.0, quality_retention)))

    def compute_speed_score(self, tps: float, ttft_ms: float) -> float:
        """Compute speed score incorporating throughput and TTFT latency."""
        w = self.weights
        if tps <= w.tps_min:
            tps_score = max(0.0, (tps / max(w.tps_min, 1e-6)) * 20.0)
        else:
            base_tps = (
                20.0
                + (min(tps, w.tps_target) - w.tps_min)
                / (w.tps_target - w.tps_min)
                * 70.0
            )
            bonus = 0.0
            if tps > w.tps_target:
                bonus = min(10.0, (tps - w.tps_target) * 0.5)
            tps_score = min(100.0, base_tps + bonus)

        if ttft_ms <= w.ttft_target_ms:
            ttft_score = 100.0
        elif ttft_ms >= w.ttft_max_ms:
            ttft_score = 10.0
        else:
            ttft_score = (
                100.0
                - ((ttft_ms - w.ttft_target_ms) / (w.ttft_max_ms - w.ttft_target_ms))
                * 90.0
            )

        return float(0.80 * tps_score + 0.20 * ttft_score)

    def compute_mem_score(
        self, vram_used_mb: float, ram_used_mb: float
    ) -> tuple[float, bool, float]:
        """Evaluate memory footprint against 4GB VRAM and 16GB host RAM budget."""
        w = self.weights
        vram_limit = w.vram_budget_mb
        vram_safe_ceiling = max(500.0, vram_limit - w.vram_safe_headroom_mb)
        headroom = vram_limit - vram_used_mb
        fits_vram = vram_used_mb <= vram_limit

        if vram_used_mb <= vram_safe_ceiling:
            mem_score = 85.0 + 15.0 * (
                1.0 - (vram_used_mb / max(vram_safe_ceiling, 1.0))
            )
        elif vram_used_mb <= vram_limit:
            margin_ratio = (vram_limit - vram_used_mb) / max(
                1.0, (vram_limit - vram_safe_ceiling)
            )
            mem_score = 50.0 + 35.0 * margin_ratio
        else:
            overflow = vram_used_mb - vram_limit
            penalty = min(40.0, (overflow / 1000.0) * 15.0)
            mem_score = max(5.0, 45.0 - penalty)

        if ram_used_mb > (w.ram_budget_mb * 0.85):
            ram_penalty = min(
                20.0, ((ram_used_mb - (w.ram_budget_mb * 0.85)) / 1000.0) * 5.0
            )
            mem_score = max(0.0, mem_score - ram_penalty)

        return float(mem_score), fits_vram, float(headroom)

    def compute_spec_score(
        self,
        is_pair: bool,
        vocab_aligned: bool,
        template_aligned: bool,
        alpha: float,
        speedup: float,
        quality_retention: float,
    ) -> float:
        """Evaluate speculative synergy, penalizing mismatched vocab and slowdowns."""
        if not is_pair:
            return 50.0

        vocab_factor = 1.0 if vocab_aligned else 0.15
        template_factor = 1.0 if template_aligned else 0.25

        if speedup >= 1.0:
            speedup_mult = min(1.6, speedup)
        else:
            speedup_mult = max(0.0, speedup**2)

        raw = (
            (100.0 * max(0.0, min(1.0, alpha)) * vocab_factor * template_factor)
            * speedup_mult
            * quality_retention
        )
        return float(max(0.0, min(100.0, raw)))

    def score_standalone(self, model: ModelSpecs) -> EvaluationScore:
        """Score a standalone LLM configuration."""
        w = self.weights
        intel = self.compute_intel_score(
            model.mmlu,
            model.gsm8k,
            model.humaneval,
            model.arena_hard,
            quality_retention=1.0,
        )
        speed = self.compute_speed_score(model.standalone_tps, model.ttft_ms)
        mem, fits_vram, headroom = self.compute_mem_score(
            model.runtime_vram_mb, model.runtime_ram_mb
        )
        spec = self.compute_spec_score(
            is_pair=False,
            vocab_aligned=True,
            template_aligned=True,
            alpha=0.0,
            speedup=1.0,
            quality_retention=1.0,
        )

        total = w.w_intel * intel + w.w_speed * speed + w.w_mem * mem + w.w_spec * spec

        pros: list[str] = []
        cons: list[str] = []

        if intel >= 70.0:
            pros.append(f"Strong intelligence index ({intel:.1f})")
        if speed >= 75.0:
            pros.append(f"Fluid generation ({model.standalone_tps:.1f} tok/s)")
        if headroom >= 800.0:
            pros.append(f"Generous VRAM headroom ({headroom:.0f} MiB free)")

        if not fits_vram:
            cons.append(
                f"Exceeds 4GB VRAM ceiling ({model.runtime_vram_mb:.0f} MiB), requires CPU offload"
            )
        elif headroom < 500.0:
            cons.append(
                f"Tight VRAM headroom ({headroom:.0f} MiB), limits KV cache context"
            )
        if model.standalone_tps < 8.0:
            cons.append(
                f"Sub-interactive throughput ({model.standalone_tps:.1f} tok/s)"
            )

        verdict = (
            "Viable Standalone"
            if fits_vram and model.standalone_tps >= 10.0
            else "Constrained Standalone"
        )

        return EvaluationScore(
            candidate_id=model.model_id,
            display_name=f"{model.name} ({model.quantization})",
            is_speculative_pair=False,
            target_name=model.name,
            drafter_name="None (Standalone Autoregressive)",
            intel_score=round(intel, 2),
            speed_score=round(speed, 2),
            mem_score=round(mem, 2),
            spec_score=round(spec, 2),
            total_score=round(total, 2),
            fits_vram=fits_vram,
            vram_used_mb=model.runtime_vram_mb,
            vram_headroom_mb=round(headroom, 1),
            effective_tps=model.standalone_tps,
            speedup_ratio=1.0,
            quality_retention=1.0,
            verdict=verdict,
            pros=pros,
            cons=cons,
        )

    def score_speculative_pair(self, pair: SpeculativePairSpecs) -> EvaluationScore:
        """Score a speculative draft-target pair configuration."""
        w = self.weights
        target = pair.target
        intel = self.compute_intel_score(
            target.mmlu,
            target.gsm8k,
            target.humaneval,
            target.arena_hard,
            quality_retention=pair.quality_retention,
        )
        speed = self.compute_speed_score(pair.effective_tps, target.ttft_ms * 0.95)
        mem, fits_vram, headroom = self.compute_mem_score(
            pair.combined_vram_mb, pair.combined_ram_mb
        )
        spec = self.compute_spec_score(
            is_pair=True,
            vocab_aligned=pair.vocab_aligned,
            template_aligned=pair.template_aligned,
            alpha=pair.expected_alpha,
            speedup=pair.speedup_ratio,
            quality_retention=pair.quality_retention,
        )

        total = w.w_intel * intel + w.w_speed * speed + w.w_mem * mem + w.w_spec * spec

        pros: list[str] = []
        cons: list[str] = []

        if pair.vocab_aligned and pair.template_aligned:
            pros.append(
                f"Lossless distribution match ({pair.expected_alpha * 100:.0f}% acceptance rate)"
            )
        if pair.speedup_ratio >= 1.25:
            pros.append(
                f"High speculative acceleration ({pair.speedup_ratio:.2f}x speedup -> {pair.effective_tps:.1f} tok/s)"
            )
        if pair.drafter_device == "cpu":
            pros.append("CPU P-core drafting offloads compute from 4GB GPU")
        elif pair.drafter_type == "prompt_lookup":
            pros.append("Zero-FLOP, zero-VRAM Prompt Lookup synergy")

        if not pair.vocab_aligned:
            cons.append(
                "Vocabulary mismatch! Severe re-tokenization artifacts and drift risk"
            )
        if not pair.template_aligned:
            cons.append("Chat template mismatch! Special token leakage risk")
        if pair.speedup_ratio < 1.0:
            cons.append(
                f"Negative speedup detected ({pair.speedup_ratio:.2f}x slowdown)! Drafter overhead exceeds gains"
            )
        if not fits_vram:
            cons.append("Combined setup exceeds 4GB VRAM capacity")

        if pair.speedup_ratio >= 1.20 and pair.vocab_aligned and fits_vram:
            verdict = "High-Synergy Speculative Champion"
        elif pair.speedup_ratio < 1.0 or not pair.vocab_aligned:
            verdict = "Degraded / Toxic Speculative Pair"
        else:
            verdict = "Moderate Speculative Gain"

        return EvaluationScore(
            candidate_id=pair.pair_id,
            display_name=f"{pair.target.name} + {pair.drafter_name}",
            is_speculative_pair=True,
            target_name=pair.target.name,
            drafter_name=pair.drafter_name,
            intel_score=round(intel, 2),
            speed_score=round(speed, 2),
            mem_score=round(mem, 2),
            spec_score=round(spec, 2),
            total_score=round(total, 2),
            fits_vram=fits_vram,
            vram_used_mb=pair.combined_vram_mb,
            vram_headroom_mb=round(headroom, 1),
            effective_tps=pair.effective_tps,
            speedup_ratio=pair.speedup_ratio,
            quality_retention=pair.quality_retention,
            verdict=verdict,
            pros=pros,
            cons=cons,
        )

    @staticmethod
    def calculate_expected_acceptance_length(alpha: float, gamma: int) -> float:
        """Compute expected number of accepted + bonus/correction tokens per speculative step.

        Formula (Leviathan et al. 2023, Chen et al. 2023):
            E[N] = (1 - alpha^(gamma + 1)) / (1 - alpha)
        Handles alpha -> 1.0 limit gracefully: E[N] = gamma + 1.
        """
        if gamma <= 0:
            return 1.0
        a = max(0.0, min(1.0, float(alpha)))
        if abs(a - 1.0) < 1e-7:
            return float(gamma + 1)
        return float((1.0 - a ** (gamma + 1)) / (1.0 - a))

    @staticmethod
    def calculate_theoretical_speedup(
        alpha: float,
        gamma: int,
        cost_ratio: float,
        verification_overhead: float = 0.08,
    ) -> float:
        """Compute theoretical wall-clock speedup ratio for speculative decoding.

        Formula:
            Speedup = E[N] / (1 + verification_overhead + gamma * cost_ratio)
        where cost_ratio c = T_draft / T_target.
        """
        expected_n = ModelEvaluator.calculate_expected_acceptance_length(alpha, gamma)
        denom = 1.0 + max(0.0, verification_overhead) + gamma * max(0.0, cost_ratio)
        if denom <= 0:
            return 0.0
        return float(expected_n / denom)

    @staticmethod
    def calculate_break_even_alpha(
        gamma: int,
        cost_ratio: float,
        verification_overhead: float = 0.08,
    ) -> float:
        """Compute the minimum acceptance rate alpha* required for Speedup >= 1.0."""
        target_tokens = 1.0 + verification_overhead + gamma * cost_ratio
        if target_tokens <= 1.0:
            return 0.0
        if target_tokens >= (gamma + 1):
            return 1.0
        low, high = 0.0, 1.0
        for _ in range(60):
            mid = (low + high) / 2.0
            val = ModelEvaluator.calculate_expected_acceptance_length(mid, gamma)
            if val < target_tokens:
                low = mid
            else:
                high = mid
        return float(round((low + high) / 2.0, 4))

    @staticmethod
    def calculate_optimal_gamma(
        alpha: float,
        cost_ratio: float,
        max_gamma: int = 8,
        verification_overhead: float = 0.08,
    ) -> tuple[int, float]:
        """Find the optimal lookahead gamma* that maximizes theoretical speedup."""
        best_gamma = 1
        best_speedup = 0.0
        for g in range(1, max_gamma + 1):
            sp = ModelEvaluator.calculate_theoretical_speedup(
                alpha, g, cost_ratio, verification_overhead
            )
            if sp > best_speedup:
                best_speedup = sp
                best_gamma = g
        return best_gamma, float(round(best_speedup, 3))

    def synthesize_pair(
        self,
        target_id: str,
        drafter_id: str | None = None,
        drafter_type: str = "model",
        drafter_device: str = "cpu",
        gamma: int = 4,
        custom_alpha: float | None = None,
        notes: str = "",
        auto_register: bool = True,
    ) -> SpeculativePairSpecs:
        """Synthesize and register a speculative pair from first-principles analysis."""
        target = self.get_model(target_id)
        drafter = self.get_model(drafter_id) if drafter_id is not None else None

        if drafter_type == "prompt_lookup" or drafter is None:
            drafter_name = "Prompt Lookup (Zero-FLOP)"
            drafter_type = "prompt_lookup"
            drafter_device = "zero_flop"
            vocab_aligned = True
            template_aligned = True
            combined_vram = target.runtime_vram_mb
            combined_ram = target.runtime_ram_mb + 20.0
            expected_alpha = custom_alpha if custom_alpha is not None else 0.48
            cost_ratio = 0.005
            quality_retention = 1.0
            dist_mode = "lossless_exact"
        else:
            drafter_name = f"{drafter.name} ({'Intel P-cores AVX2' if drafter_device == 'cpu' else 'CUDA'})"
            vocab_aligned = (
                target.family == drafter.family
                and target.family != ModelFamily.OTHER
                and abs(target.vocab_size - drafter.vocab_size) < 1000
            )
            template_aligned = target.chat_template == drafter.chat_template

            if drafter_device == "cuda":
                combined_vram = target.runtime_vram_mb + drafter.runtime_vram_mb
                combined_ram = target.runtime_ram_mb
                cost_ratio = 0.08
            else:
                combined_vram = target.runtime_vram_mb
                combined_ram = target.runtime_ram_mb + drafter.runtime_ram_mb
                cost_ratio = 0.12 if drafter.parameter_count_b <= 1.0 else 0.22

            if custom_alpha is not None:
                expected_alpha = custom_alpha
            elif not vocab_aligned:
                expected_alpha = 0.16
            elif not template_aligned:
                expected_alpha = 0.35
            else:
                size_ratio = min(
                    1.0, drafter.parameter_count_b / max(1e-6, target.parameter_count_b)
                )
                base_alpha = 0.65 + 0.15 * size_ratio
                is_target_reasoning = (
                    "reasoning" in target.specialization or target.is_fine_tuned
                )
                is_drafter_reasoning = (
                    "reasoning" in drafter.specialization or drafter.is_fine_tuned
                )
                if is_target_reasoning and not is_drafter_reasoning:
                    base_alpha -= 0.12
                expected_alpha = max(0.20, min(0.90, base_alpha))

            if not vocab_aligned:
                quality_retention = 0.55
                dist_mode = "lossy_mismatched_vocab"
            elif not template_aligned:
                quality_retention = 0.75
                dist_mode = "lossy_template_mismatch"
            else:
                quality_retention = 1.0
                dist_mode = "lossless_exact"

        speedup = self.calculate_theoretical_speedup(expected_alpha, gamma, cost_ratio)
        effective_tps = target.standalone_tps * speedup

        pair_id = f"synthesized-{target.model_id}-{drafter.model_id if drafter else 'prompt-lookup'}"
        pair = SpeculativePairSpecs(
            pair_id=pair_id,
            target=target,
            drafter=drafter,
            drafter_name=drafter_name,
            drafter_type=drafter_type,
            drafter_device=drafter_device,
            combined_vram_mb=round(combined_vram, 1),
            combined_ram_mb=round(combined_ram, 1),
            vocab_aligned=vocab_aligned,
            template_aligned=template_aligned,
            expected_alpha=round(expected_alpha, 3),
            gamma=gamma,
            effective_tps=round(effective_tps, 1),
            speedup_ratio=round(speedup, 2),
            quality_retention=quality_retention,
            distribution_mode=dist_mode,
            notes=notes or f"Synthesized pair ({dist_mode})",
        )

        if auto_register:
            self.register_pair(pair)

        return pair

    def evaluate_all(
        self,
        weights: EvaluationWeights | None = None,
        filter_type: str = "all",
    ) -> list[EvaluationScore]:
        """Evaluate and rank all registered standalone models and speculative pairs.

        Args:
            weights: Optional custom evaluation weights profile.
            filter_type: Output filter: 'all', 'standalone', 'speculative', 'pareto'.
        """
        if weights is not None:
            self.weights = weights

        scores: list[EvaluationScore] = []
        for model in self._models.values():
            scores.append(self.score_standalone(model))

        for pair in self._pairs.values():
            scores.append(self.score_speculative_pair(pair))

        scores = self.compute_pareto_frontier(scores)
        scores.sort(key=lambda s: s.total_score, reverse=True)

        for rank, s in enumerate(scores, start=1):
            s.rank = rank

        f = filter_type.lower().strip()
        if f == "standalone":
            return [s for s in scores if not s.is_speculative_pair]
        if f in ("speculative", "pairs", "pair"):
            return [s for s in scores if s.is_speculative_pair]
        if f == "pareto":
            return [s for s in scores if s.is_pareto_optimal]
        return scores

    @staticmethod
    def compute_pareto_frontier(scores: list[EvaluationScore]) -> list[EvaluationScore]:
        """Mark candidates that lie on the Pareto optimal frontier (Intelligence vs Speed vs VRAM)."""
        valid = [s for s in scores if s.fits_vram]
        pareto_ids: set[str] = set()

        for cand in valid:
            is_dominated = False
            for other in valid:
                if other.candidate_id == cand.candidate_id:
                    continue
                better_or_equal = (
                    other.intel_score >= cand.intel_score
                    and other.effective_tps >= cand.effective_tps
                    and other.vram_used_mb <= cand.vram_used_mb
                )
                strictly_better = (
                    other.intel_score > cand.intel_score
                    or other.effective_tps > cand.effective_tps
                    or other.vram_used_mb < cand.vram_used_mb
                )
                if better_or_equal and strictly_better:
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_ids.add(cand.candidate_id)

        for s in scores:
            s.is_pareto_optimal = s.candidate_id in pareto_ids

        return scores

    @staticmethod
    def explain_speculative_phenomenon() -> dict[str, Any]:
        """Return theoretical journal-backed explanation of the speculative drafting phenomenon."""
        return {
            "lossless_distribution_theorem": (
                "Leviathan et al. (2023) Theorem 1 & Chen et al. (2023) prove that rejection "
                "sampling with acceptance probability min(1, p(x)/q(x)) and residual distribution "
                "p'(x) = max(0, p(x) - q(x)) / sum(max(0, p - q)) generates sequences from the exact "
                "target distribution P(X). Mathematical output fidelity is 100% lossless."
            ),
            "root_causes_of_output_degradation": [
                {
                    "cause": "Cross-Family Vocabulary & Tokenizer Mismatch",
                    "impact": "Critical Output Drift",
                    "mechanism": (
                        "Pairing models with disparate vocabularies (e.g. Qwen 152k BPE vs Llama 128k Tiktoken) "
                        "breaks the Markov chain. Approximating via string re-tokenization leads to byte-boundary "
                        "fragmentation, prefix desynchronization, and corrupted KV-cache alignment."
                    ),
                },
                {
                    "cause": "Lossy Heuristic Implementations vs Rejection Sampling",
                    "impact": "Distribution Distortion",
                    "mechanism": (
                        "Naive drafting engines accept tokens based on top-k confidence or threshold heuristics "
                        "without sampling from the residual distribution max(0, P - Q). This systematically "
                        "biases generated tokens toward the weaker draft model distribution."
                    ),
                },
                {
                    "cause": "Chat Template & Special Token Contamination",
                    "impact": "Format Breakdown & Runaway Generation",
                    "mechanism": (
                        "Divergent control tokens (e.g. ChatML '<|im_start|>' vs Llama-3 '<|start_header_id|>') "
                        "cause the draft model to inject foreign tokens, triggering early termination or infinite repetition."
                    ),
                },
                {
                    "cause": "Negative Throughput Speedup (alpha < 0.35)",
                    "impact": "Severe Latency Penalty (Worse than Standalone)",
                    "mechanism": (
                        "Speedup = (1 - alpha^(gamma+1)) / ((1 - alpha) * (1 + gamma * c)). When draft quality "
                        "is severely underpowered, alpha drops below 0.30. The system spends FLOPs verifying draft tokens "
                        "that are immediately rejected, resulting in net speedup < 1.0 (a 20-40% slowdown)."
                    ),
                },
            ],
            "reasoning_model_cot_dynamics": {
                "phenomenon": "Speculative Drafting with Fine-Tuned Reasoning Models (e.g., DeepSeek-R1 Distillations)",
                "root_causes_and_mechanisms": [
                    {
                        "issue": "Intra-Thought Chain-of-Thought Distribution Shift",
                        "mechanism": (
                            "Fine-tuned reasoning models emit structured thinking tokens (<think> ... </think>). "
                            "Standard base or instruct drafters lack long-horizon reasoning priors. During the thinking "
                            "phase, the drafter proposes standard conversational tokens, causing acceptance rate alpha "
                            "to drop from ~74% to ~45-55%."
                        ),
                        "mathematical_implication": (
                            "While mathematical losslessness is strictly maintained under Leviathan Theorem 1 rejection "
                            "sampling, the drop in alpha increases the cost ratio penalty, reducing speculative speedup."
                        ),
                    },
                    {
                        "issue": "Thought Truncation Risk in Lossy / Greedy Drafting",
                        "mechanism": (
                            "In naive speculative implementations using greedy verification without residual resampling, "
                            "a base drafter may propose a premature '</think>' tag. If accepted heuristically, this "
                            "cuts off the reasoning chain before the solution is derived, causing catastrophic quality collapse."
                        ),
                        "mitigation": (
                            "Chimera-Inference strictly enforces exact rejection sampling, guaranteeing that premature "
                            "stop or tag tokens are rejected with exact probability 1 - P(x)/Q(x)."
                        ),
                    },
                    {
                        "issue": "Optimal Drafter Strategy for Reasoning Models on 4GB VRAM",
                        "mechanism": (
                            "For DeepSeek-R1 targets: (1) Zero-FLOP Prompt Lookup provides 100% distribution-safe, zero-VRAM "
                            "speculation (~1.27x speedup), or (2) CPU P-core AVX2 same-family drafting preserves 100% of GPU "
                            "VRAM for the reasoning target while delivering ~1.37x speedup."
                        ),
                    },
                ],
            },
            "hardware_mitigation_strategy": (
                "For RTX 3050 (4GB VRAM) laptops, running draft models on GPU creates VRAM starvation. "
                "The optimal architecture binds the fast drafter (e.g. Qwen 2.5 0.5B) to Intel P-cores via AVX2 "
                "or uses zero-FLOP Prompt Lookup, preserving 100% GPU VRAM for the high-capacity target model."
            ),
        }

    @staticmethod
    def get_standard_quantization_profiles() -> dict[
        QuantizationType, QuantizationProfile
    ]:
        """Return empirical profiles for common quantization formats."""
        return {
            QuantizationType.BF16: QuantizationProfile(
                format=QuantizationType.BF16,
                bits_per_weight=16.0,
                quality_retention=1.000,
                relative_size_ratio=1.000,
                description="Unquantized 16-bit float; 100% loss-free baseline reference.",
            ),
            QuantizationType.FP8: QuantizationProfile(
                format=QuantizationType.FP8,
                bits_per_weight=8.0,
                quality_retention=0.995,
                relative_size_ratio=0.500,
                description="8-bit floating point; near-lossless fidelity with 50% memory savings.",
            ),
            QuantizationType.Q8_0: QuantizationProfile(
                format=QuantizationType.Q8_0,
                bits_per_weight=8.5,
                quality_retention=0.998,
                relative_size_ratio=0.531,
                description="8-bit symmetric integer quant; virtually indistinguishable from FP16.",
            ),
            QuantizationType.Q6_K: QuantizationProfile(
                format=QuantizationType.Q6_K,
                bits_per_weight=6.56,
                quality_retention=0.992,
                relative_size_ratio=0.410,
                description="6-bit k-quant; high-fidelity preservation with ~60% memory savings.",
            ),
            QuantizationType.Q5_K_M: QuantizationProfile(
                format=QuantizationType.Q5_K_M,
                bits_per_weight=5.54,
                quality_retention=0.985,
                relative_size_ratio=0.359,
                description="5-bit medium k-quant; optimal intelligence-to-VRAM retention.",
            ),
            QuantizationType.Q4_K_M: QuantizationProfile(
                format=QuantizationType.Q4_K_M,
                bits_per_weight=4.50,
                quality_retention=0.973,
                relative_size_ratio=0.310,
                description="4-bit medium k-quant; industry standard golden sweet-spot for 4GB VRAM.",
            ),
            QuantizationType.Q3_K_M: QuantizationProfile(
                format=QuantizationType.Q3_K_M,
                bits_per_weight=3.44,
                quality_retention=0.915,
                relative_size_ratio=0.233,
                description="3-bit aggressive quantization; noticeable drop on reasoning & math benchmarks.",
            ),
            QuantizationType.Q2_K: QuantizationProfile(
                format=QuantizationType.Q2_K,
                bits_per_weight=2.63,
                quality_retention=0.785,
                relative_size_ratio=0.183,
                description="2-bit extreme quantization; severe perplexity spikes and reasoning degradation.",
            ),
        }

    def evaluate_quantization_spectrum(
        self,
        model_id: str,
        weights: EvaluationWeights | None = None,
    ) -> list[EvaluationScore]:
        """Evaluate a model across the entire quantization spectrum (BF16 down to Q2_K).

        Models the physical memory boundaries of 4GB VRAM, throughput changes driven
        by bandwidth, and empirical benchmark retention.
        """
        if weights is not None:
            self.weights = weights
        base_model = self.get_model(model_id)
        profiles = self.get_standard_quantization_profiles()

        current_ratio = 0.310 if "q4" in base_model.quantization.lower() else 1.0
        bf16_weight_mb = base_model.weights_size_mb / current_ratio
        baseline_retention = 0.973 if "q4" in base_model.quantization.lower() else 1.0

        scores: list[EvaluationScore] = []
        usable_vram_for_weights = max(0.0, self.weights.vram_budget_mb - 750.0)
        usable_ram_for_weights = max(0.0, self.weights.ram_budget_mb - 4500.0)

        for q_type, prof in profiles.items():
            est_weight_mb = round(bf16_weight_mb * prof.relative_size_ratio, 1)

            if est_weight_mb <= usable_vram_for_weights:
                device_alloc = "cuda"
                runtime_vram = round(est_weight_mb + 750.0, 1)
                runtime_ram = 1400.0
                fits_vram = True
                tps = round(min(65.0, 52000.0 / max(est_weight_mb, 100.0)), 1)
                ttft = round(max(35.0, est_weight_mb * 0.12), 1)
            elif est_weight_mb <= (usable_vram_for_weights + usable_ram_for_weights):
                device_alloc = "hybrid_cpu_gpu"
                runtime_vram = self.weights.vram_budget_mb
                cpu_weight_mb = est_weight_mb - usable_vram_for_weights
                runtime_ram = round(cpu_weight_mb + 3000.0, 1)
                fits_vram = True
                raw_hybrid_tps = (22000.0 / max(cpu_weight_mb, 200.0)) * 0.60
                tps = round(max(1.8, min(9.5, raw_hybrid_tps)), 1)
                ttft = round(350.0 + est_weight_mb * 0.08, 1)
            else:
                device_alloc = "oom_exceeds_ram"
                runtime_vram = self.weights.vram_budget_mb
                runtime_ram = self.weights.ram_budget_mb
                fits_vram = False
                tps = 0.5
                ttft = 9999.0

            retention_mult = prof.quality_retention / baseline_retention
            scaled_mmlu = round(min(100.0, base_model.mmlu * retention_mult), 1)
            scaled_gsm8k = round(min(100.0, base_model.gsm8k * retention_mult), 1)
            scaled_humaneval = round(
                min(100.0, base_model.humaneval * retention_mult), 1
            )
            scaled_arena = round(min(100.0, base_model.arena_hard * retention_mult), 1)

            q_model = ModelSpecs(
                model_id=f"{base_model.model_id}-{q_type.value.lower()}",
                name=base_model.name,
                family=base_model.family,
                parameter_count_b=base_model.parameter_count_b,
                quantization=q_type.value,
                vocab_size=base_model.vocab_size,
                chat_template=base_model.chat_template,
                context_length=base_model.context_length,
                weights_size_mb=est_weight_mb,
                runtime_vram_mb=runtime_vram,
                runtime_ram_mb=runtime_ram,
                device_allocation=device_alloc,
                standalone_tps=tps,
                ttft_ms=ttft,
                mmlu=scaled_mmlu,
                gsm8k=scaled_gsm8k,
                humaneval=scaled_humaneval,
                arena_hard=scaled_arena,
                is_fine_tuned=base_model.is_fine_tuned,
                specialization=base_model.specialization,
                description=prof.description,
            )

            score = self.score_standalone(q_model)
            if not fits_vram:
                score.fits_vram = False
                score.verdict = "OOM / RAM Exhaustion"
                score.cons.append(
                    "Total model footprint exceeds 16GB physical RAM capacity"
                )
            scores.append(score)

        scores = self.compute_pareto_frontier(scores)
        scores.sort(key=lambda s: s.total_score, reverse=True)
        for r, s in enumerate(scores, start=1):
            s.rank = r
        return scores

    def format_quantization_spectrum_markdown(
        self, scores: list[EvaluationScore], model_name: str
    ) -> str:
        """Format the quantization spectrum analysis as a clean markdown report."""
        lines = [
            f"# Quantization Spectrum & Pareto Analysis: {model_name}",
            "",
            (
                "**Hardware Constraints**: NVIDIA RTX 3050 (4095 MiB VRAM) | "
                "16GB System RAM | Intel i5-12500H P-cores AVX2"
            ),
            "",
            "| Rank | Quantization | Intel Score | Speed (TPS) | VRAM (MiB) | Value Score | Pareto? | Allocation Regime |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
        for s in scores:
            p_badge = "**Yes (SOTA)**" if s.is_pareto_optimal else "No"
            q_name = s.display_name.split("(")[-1].rstrip(")")
            lines.append(
                f"| {s.rank} | **{q_name}** | {s.intel_score:.1f} | "
                f"{s.effective_tps:.1f} | {s.vram_used_mb:.0f} | {s.total_score:.1f} | "
                f"{p_badge} | {s.verdict} |"
            )
        return "\n".join(lines)

    def _populate_default_catalog(self) -> None:
        """Curate SOTA open-weight models fitting RTX 3050 (4GB) + i5-12500H laptop envelope."""
        # Standalone Models
        m_qwen_05b = ModelSpecs(
            model_id="qwen2.5-0.5b-instruct-q4",
            name="Qwen 2.5 0.5B Instruct",
            family=ModelFamily.QWEN2_5,
            parameter_count_b=0.49,
            quantization="Q4_K_M",
            vocab_size=151936,
            chat_template="chatml",
            context_length=32768,
            weights_size_mb=398.0,
            runtime_vram_mb=750.0,
            runtime_ram_mb=1100.0,
            device_allocation="cuda",
            standalone_tps=58.4,
            ttft_ms=42.0,
            mmlu=55.4,
            gsm8k=53.6,
            humaneval=43.3,
            arena_hard=32.5,
            specialization="ultra_fast_edge",
            description="Ultra-lightweight edge model with high TPS and minimal VRAM.",
        )

        m_qwen_15b = ModelSpecs(
            model_id="qwen2.5-1.5b-instruct-q4",
            name="Qwen 2.5 1.5B Instruct",
            family=ModelFamily.QWEN2_5,
            parameter_count_b=1.54,
            quantization="Q4_K_M",
            vocab_size=151936,
            chat_template="chatml",
            context_length=32768,
            weights_size_mb=986.0,
            runtime_vram_mb=3030.0,
            runtime_ram_mb=1800.0,
            device_allocation="cuda",
            standalone_tps=28.4,
            ttft_ms=153.6,
            mmlu=65.2,
            gsm8k=69.8,
            humaneval=54.9,
            arena_hard=51.4,
            specialization="balanced_general",
            description="Baseline model verified on local RTX 3050 CUDA engine.",
        )

        m_deepseek_r1_15b = ModelSpecs(
            model_id="deepseek-r1-distill-qwen-1.5b-q4",
            name="DeepSeek-R1-Distill-Qwen 1.5B",
            family=ModelFamily.DEEPSEEK_R1_QWEN,
            parameter_count_b=1.54,
            quantization="Q4_K_M",
            vocab_size=151936,
            chat_template="chatml",
            context_length=32768,
            weights_size_mb=986.0,
            runtime_vram_mb=3050.0,
            runtime_ram_mb=1800.0,
            device_allocation="cuda",
            standalone_tps=28.2,
            ttft_ms=155.0,
            mmlu=65.4,
            gsm8k=82.8,
            humaneval=56.7,
            arena_hard=55.2,
            is_fine_tuned=True,
            specialization="deep_reasoning_math",
            description="Distilled with DeepSeek-R1 reasoning traces; outstanding GSM8K math capabilities.",
        )

        m_qwen_3b = ModelSpecs(
            model_id="qwen2.5-3b-instruct-q4",
            name="Qwen 2.5 3B Instruct",
            family=ModelFamily.QWEN2_5,
            parameter_count_b=3.09,
            quantization="Q4_K_M",
            vocab_size=151936,
            chat_template="chatml",
            context_length=32768,
            weights_size_mb=1930.0,
            runtime_vram_mb=3650.0,
            runtime_ram_mb=2400.0,
            device_allocation="cuda",
            standalone_tps=17.5,
            ttft_ms=240.0,
            mmlu=71.6,
            gsm8k=78.2,
            humaneval=65.2,
            arena_hard=64.8,
            specialization="high_quality_compact",
            description="Strongest model fitting fully into 4GB VRAM with ~445 MiB KV cache headroom.",
        )

        m_qwen_7b = ModelSpecs(
            model_id="qwen2.5-7b-instruct-q4",
            name="Qwen 2.5 7B Instruct",
            family=ModelFamily.QWEN2_5,
            parameter_count_b=7.61,
            quantization="Q4_K_M",
            vocab_size=152064,
            chat_template="chatml",
            context_length=32768,
            weights_size_mb=4480.0,
            runtime_vram_mb=4095.0,
            runtime_ram_mb=4900.0,
            device_allocation="hybrid_cpu_gpu",
            standalone_tps=6.8,
            ttft_ms=680.0,
            mmlu=79.5,
            gsm8k=85.4,
            humaneval=81.7,
            arena_hard=77.2,
            specialization="frontier_general",
            description="High intelligence flagship; requires 14-layer CPU offloading on 4GB VRAM.",
        )

        m_deepseek_r1_7b = ModelSpecs(
            model_id="deepseek-r1-distill-qwen-7b-q4",
            name="DeepSeek-R1-Distill-Qwen 7B",
            family=ModelFamily.DEEPSEEK_R1_QWEN,
            parameter_count_b=7.61,
            quantization="Q4_K_M",
            vocab_size=152064,
            chat_template="chatml",
            context_length=32768,
            weights_size_mb=4480.0,
            runtime_vram_mb=4095.0,
            runtime_ram_mb=4950.0,
            device_allocation="hybrid_cpu_gpu",
            standalone_tps=6.5,
            ttft_ms=710.0,
            mmlu=79.8,
            gsm8k=89.3,
            humaneval=82.3,
            arena_hard=79.1,
            is_fine_tuned=True,
            specialization="frontier_reasoning",
            description="SOTA distilled reasoning model with 89.3 GSM8K; requires hybrid CPU/GPU offload.",
        )

        m_llama_1b = ModelSpecs(
            model_id="llama3.2-1b-instruct-q4",
            name="Llama 3.2 1B Instruct",
            family=ModelFamily.LLAMA3,
            parameter_count_b=1.23,
            quantization="Q4_K_M",
            vocab_size=128256,
            chat_template="llama3",
            context_length=131072,
            weights_size_mb=780.0,
            runtime_vram_mb=1250.0,
            runtime_ram_mb=1500.0,
            device_allocation="cuda",
            standalone_tps=41.5,
            ttft_ms=72.0,
            mmlu=49.3,
            gsm8k=44.4,
            humaneval=37.2,
            arena_hard=30.1,
            specialization="fast_on_device",
            description="Meta lightweight model with expansive 128k context support.",
        )

        m_llama_3b = ModelSpecs(
            model_id="llama3.2-3b-instruct-q4",
            name="Llama 3.2 3B Instruct",
            family=ModelFamily.LLAMA3,
            parameter_count_b=3.21,
            quantization="Q4_K_M",
            vocab_size=128256,
            chat_template="llama3",
            context_length=131072,
            weights_size_mb=2010.0,
            runtime_vram_mb=2550.0,
            runtime_ram_mb=2200.0,
            device_allocation="cuda",
            standalone_tps=23.8,
            ttft_ms=165.0,
            mmlu=63.4,
            gsm8k=69.5,
            humaneval=51.8,
            arena_hard=52.3,
            specialization="balanced_general",
            description="Meta mid-tier model with comfortable 1,545 MiB VRAM safety buffer on 4GB GPU.",
        )

        m_llama_8b = ModelSpecs(
            model_id="llama3.1-8b-instruct-q4",
            name="Llama 3.1 8B Instruct",
            family=ModelFamily.LLAMA3,
            parameter_count_b=8.03,
            quantization="Q4_K_M",
            vocab_size=128256,
            chat_template="llama3",
            context_length=131072,
            weights_size_mb=4920.0,
            runtime_vram_mb=4095.0,
            runtime_ram_mb=5200.0,
            device_allocation="hybrid_cpu_gpu",
            standalone_tps=5.6,
            ttft_ms=780.0,
            mmlu=73.0,
            gsm8k=84.5,
            humaneval=72.6,
            arena_hard=68.2,
            specialization="frontier_general",
            description="Flagship open model; constrained by 4GB VRAM, requires CPU layer offloading.",
        )

        m_gemma_2b = ModelSpecs(
            model_id="gemma2-2b-instruct-q4",
            name="Gemma 2 2B Instruct",
            family=ModelFamily.GEMMA2,
            parameter_count_b=2.61,
            quantization="Q4_K_M",
            vocab_size=256000,
            chat_template="gemma",
            context_length=8192,
            weights_size_mb=1620.0,
            runtime_vram_mb=2150.0,
            runtime_ram_mb=2000.0,
            device_allocation="cuda",
            standalone_tps=27.0,
            ttft_ms=140.0,
            mmlu=56.3,
            gsm8k=54.1,
            humaneval=41.5,
            arena_hard=42.0,
            specialization="general_compact",
            description="Google compact model featuring sliding window attention and 256k vocab.",
        )

        m_deepseek_r1_llama8b = ModelSpecs(
            model_id="deepseek-r1-distill-llama-8b-q4",
            name="DeepSeek-R1-Distill-Llama 8B",
            family=ModelFamily.DEEPSEEK_R1_LLAMA,
            parameter_count_b=8.03,
            quantization="Q4_K_M",
            vocab_size=128256,
            chat_template="llama3",
            context_length=131072,
            weights_size_mb=4920.0,
            runtime_vram_mb=4095.0,
            runtime_ram_mb=5250.0,
            device_allocation="hybrid_cpu_gpu",
            standalone_tps=5.4,
            ttft_ms=790.0,
            mmlu=73.0,
            gsm8k=89.1,
            humaneval=72.6,
            arena_hard=68.5,
            is_fine_tuned=True,
            specialization="frontier_reasoning",
            description="DeepSeek-R1 reasoning distilled into Llama 3.1 8B; requires hybrid CPU/GPU offload.",
        )

        m_k2_7b = ModelSpecs(
            model_id="k2-horizon-7b-q4",
            name="K2-Horizon-7B",
            family=ModelFamily.K2_HORIZON,
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
            specialization="frontier_reasoning_coding",
            description="IFM 512K dense reasoning model (9.0B total params); exceptional math/coding, requires hybrid offload on 4GB VRAM.",
        )

        for m in [
            m_qwen_05b,
            m_qwen_15b,
            m_deepseek_r1_15b,
            m_qwen_3b,
            m_qwen_7b,
            m_deepseek_r1_7b,
            m_llama_1b,
            m_llama_3b,
            m_llama_8b,
            m_deepseek_r1_llama8b,
            m_gemma_2b,
            m_k2_7b,
        ]:
            self.register_model(m)

        # Speculative Pairs
        pair_prompt_lookup_qwen15 = SpeculativePairSpecs(
            pair_id="pair-prompt-lookup-qwen1.5b",
            target=m_qwen_15b,
            drafter=None,
            drafter_name="Prompt Lookup (Zero-FLOP)",
            drafter_type="prompt_lookup",
            drafter_device="zero_flop",
            combined_vram_mb=3030.0,
            combined_ram_mb=1820.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.48,
            gamma=3,
            effective_tps=36.2,
            speedup_ratio=1.27,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="Zero VRAM, zero extra GPU compute. Ideal for code & structured prompts.",
        )

        pair_qwen05_qwen15 = SpeculativePairSpecs(
            pair_id="pair-qwen0.5b-cpu-qwen1.5b-cuda",
            target=m_qwen_15b,
            drafter=m_qwen_05b,
            drafter_name="Qwen 2.5 0.5B (Intel P-cores AVX2)",
            drafter_type="model",
            drafter_device="cpu",
            combined_vram_mb=3030.0,
            combined_ram_mb=2900.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.74,
            gamma=4,
            effective_tps=41.5,
            speedup_ratio=1.46,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="Lossless drafting with identical 152k vocab; drafter on P-cores preserves 4GB GPU.",
        )

        pair_qwen05_qwen3b = SpeculativePairSpecs(
            pair_id="pair-qwen0.5b-cpu-qwen3b-cuda",
            target=m_qwen_3b,
            drafter=m_qwen_05b,
            drafter_name="Qwen 2.5 0.5B (Intel P-cores AVX2)",
            drafter_type="model",
            drafter_device="cpu",
            combined_vram_mb=3650.0,
            combined_ram_mb=3500.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.71,
            gamma=4,
            effective_tps=26.6,
            speedup_ratio=1.52,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="Highest-intelligence pair fitting 100% inside 4GB VRAM, accelerated to 26.6 TPS.",
        )

        pair_qwen05_deepseek7b = SpeculativePairSpecs(
            pair_id="pair-qwen0.5b-cpu-deepseek7b-hybrid",
            target=m_deepseek_r1_7b,
            drafter=m_qwen_05b,
            drafter_name="Qwen 2.5 0.5B (Intel P-cores AVX2)",
            drafter_type="model",
            drafter_device="cpu",
            combined_vram_mb=4095.0,
            combined_ram_mb=6050.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.63,
            gamma=3,
            effective_tps=10.2,
            speedup_ratio=1.57,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="Frontier reasoning (GSM8K 89.3) accelerated from 6.5 to 10.2 TPS on hybrid offload.",
        )

        pair_llama1b_llama3b = SpeculativePairSpecs(
            pair_id="pair-llama1b-cpu-llama3b-cuda",
            target=m_llama_3b,
            drafter=m_llama_1b,
            drafter_name="Llama 3.2 1B (Intel P-cores AVX2)",
            drafter_type="model",
            drafter_device="cpu",
            combined_vram_mb=2550.0,
            combined_ram_mb=3700.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.68,
            gamma=4,
            effective_tps=34.8,
            speedup_ratio=1.46,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="Lossless Meta family pair with 1,545 MiB VRAM safety margin.",
        )

        pair_mismatched_qwen_llama = SpeculativePairSpecs(
            pair_id="pair-mismatched-qwen0.5b-llama3b",
            target=m_llama_3b,
            drafter=m_qwen_05b,
            drafter_name="Qwen 2.5 0.5B [MISMATCHED]",
            drafter_type="model",
            drafter_device="cpu",
            combined_vram_mb=2550.0,
            combined_ram_mb=3300.0,
            vocab_aligned=False,
            template_aligned=False,
            expected_alpha=0.16,
            gamma=4,
            effective_tps=17.1,
            speedup_ratio=0.72,
            quality_retention=0.55,
            distribution_mode="lossy_mismatched_vocab",
            notes="Negative speedup (0.72x slowdown) & severe quality drift due to cross-family vocab mismatch.",
        )

        pair_weak_tiny_llama8b = SpeculativePairSpecs(
            pair_id="pair-weak-tinyllama-llama8b",
            target=m_llama_8b,
            drafter=None,
            drafter_name="TinyLlama 1.1B [UNDERPOWERED]",
            drafter_type="model",
            drafter_device="cpu",
            combined_vram_mb=4095.0,
            combined_ram_mb=6100.0,
            vocab_aligned=False,
            template_aligned=False,
            expected_alpha=0.22,
            gamma=4,
            effective_tps=4.7,
            speedup_ratio=0.84,
            quality_retention=0.65,
            distribution_mode="lossy_underpowered_draft",
            notes="Severe negative speedup (0.84x slowdown); weak drafter causes near-zero acceptance rate.",
        )

        pair_prompt_lookup_deepseek15 = SpeculativePairSpecs(
            pair_id="pair-prompt-lookup-deepseek1.5b",
            target=m_deepseek_r1_15b,
            drafter=None,
            drafter_name="Prompt Lookup (Zero-FLOP)",
            drafter_type="prompt_lookup",
            drafter_device="zero_flop",
            combined_vram_mb=3050.0,
            combined_ram_mb=1820.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.46,
            gamma=3,
            effective_tps=35.8,
            speedup_ratio=1.27,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="Lossless reasoning acceleration with 0 extra VRAM and 0 extra FLOPs.",
        )

        pair_qwen05_deepseek15 = SpeculativePairSpecs(
            pair_id="pair-qwen0.5b-cpu-deepseek1.5b-cuda",
            target=m_deepseek_r1_15b,
            drafter=m_qwen_05b,
            drafter_name="Qwen 2.5 0.5B (Intel P-cores AVX2)",
            drafter_type="model",
            drafter_device="cpu",
            combined_vram_mb=3050.0,
            combined_ram_mb=2900.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.62,
            gamma=4,
            effective_tps=38.6,
            speedup_ratio=1.37,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="Reasoning distilled target with same-vocab drafter; lossless output with 1,045 MiB VRAM buffer.",
        )

        pair_qwen05_qwen7b = SpeculativePairSpecs(
            pair_id="pair-qwen0.5b-cpu-qwen7b-hybrid",
            target=m_qwen_7b,
            drafter=m_qwen_05b,
            drafter_name="Qwen 2.5 0.5B (Intel P-cores AVX2)",
            drafter_type="model",
            drafter_device="cpu",
            combined_vram_mb=4095.0,
            combined_ram_mb=6000.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.69,
            gamma=3,
            effective_tps=10.7,
            speedup_ratio=1.57,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="Accelerates 7B hybrid offload from 6.8 to 10.7 TPS via CPU P-core drafting.",
        )

        pair_prompt_lookup_gemma2b = SpeculativePairSpecs(
            pair_id="pair-prompt-lookup-gemma2b",
            target=m_gemma_2b,
            drafter=None,
            drafter_name="Prompt Lookup (Zero-FLOP)",
            drafter_type="prompt_lookup",
            drafter_device="zero_flop",
            combined_vram_mb=2150.0,
            combined_ram_mb=2020.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.45,
            gamma=3,
            effective_tps=33.8,
            speedup_ratio=1.25,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="Zero-FLOP speculative lookup on Gemma 2 2B with 1,945 MiB VRAM headroom.",
        )

        pair_prompt_lookup_llama3b = SpeculativePairSpecs(
            pair_id="pair-prompt-lookup-llama3b",
            target=m_llama_3b,
            drafter=None,
            drafter_name="Prompt Lookup (Zero-FLOP)",
            drafter_type="prompt_lookup",
            drafter_device="zero_flop",
            combined_vram_mb=2550.0,
            combined_ram_mb=2220.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.47,
            gamma=3,
            effective_tps=30.2,
            speedup_ratio=1.27,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="Zero-cost speculative lookup for Llama 3.2 3B preserving 1.5GB VRAM.",
        )

        pair_llama1b_deepseek_llama8b = SpeculativePairSpecs(
            pair_id="pair-llama1b-cpu-deepseek-llama8b-hybrid",
            target=m_deepseek_r1_llama8b,
            drafter=m_llama_1b,
            drafter_name="Llama 3.2 1B (Intel P-cores AVX2)",
            drafter_type="model",
            drafter_device="cpu",
            combined_vram_mb=4095.0,
            combined_ram_mb=6750.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.60,
            gamma=3,
            effective_tps=8.5,
            speedup_ratio=1.57,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="DeepSeek-R1 Llama 8B reasoning target accelerated via same-family 128k vocab drafter.",
        )

        pair_prompt_lookup_k2_7b = SpeculativePairSpecs(
            pair_id="pair-prompt-lookup-k2-7b",
            target=m_k2_7b,
            drafter=None,
            drafter_name="Prompt Lookup (Zero-FLOP)",
            drafter_type="prompt_lookup",
            drafter_device="zero_flop",
            combined_vram_mb=4095.0,
            combined_ram_mb=5870.0,
            vocab_aligned=True,
            template_aligned=True,
            expected_alpha=0.48,
            gamma=3,
            effective_tps=7.1,
            speedup_ratio=1.31,
            quality_retention=1.0,
            distribution_mode="lossless_exact",
            notes="Lossless acceleration for K2-Horizon-7B (5.4 -> 7.1 TPS) with zero extra VRAM.",
        )

        pair_mismatched_k2_09b_7b = SpeculativePairSpecs(
            pair_id="pair-mismatched-k2-0.9b-7b",
            target=m_k2_7b,
            drafter=None,
            drafter_name="K2-Horizon 0.9B [MISMATCHED VOCAB]",
            drafter_type="model",
            drafter_device="cpu",
            combined_vram_mb=4095.0,
            combined_ram_mb=7250.0,
            vocab_aligned=False,
            template_aligned=True,
            expected_alpha=0.14,
            gamma=4,
            effective_tps=3.8,
            speedup_ratio=0.70,
            quality_retention=0.55,
            distribution_mode="lossy_mismatched_vocab",
            notes="Negative speedup (0.70x slowdown); disparate vocab (64k vs 250k) destroys token acceptance.",
        )

        for p in [
            pair_prompt_lookup_qwen15,
            pair_qwen05_qwen15,
            pair_qwen05_qwen3b,
            pair_qwen05_deepseek15,
            pair_prompt_lookup_deepseek15,
            pair_qwen05_qwen7b,
            pair_qwen05_deepseek7b,
            pair_llama1b_llama3b,
            pair_prompt_lookup_llama3b,
            pair_llama1b_deepseek_llama8b,
            pair_prompt_lookup_gemma2b,
            pair_prompt_lookup_k2_7b,
            pair_mismatched_qwen_llama,
            pair_weak_tiny_llama8b,
            pair_mismatched_k2_09b_7b,
        ]:
            self.register_pair(p)

    def format_comparison_markdown(
        self, scores: list[EvaluationScore], preset_name: str = "balanced"
    ) -> str:
        """Generate a professional benchmark report in Markdown format."""
        lines: list[str] = [
            f"# Chimera-Inference: SOTA Model & Speculative Valuation Benchmark ({preset_name.title()} Preset)",
            "",
            (
                "**Hardware Envelope**: NVIDIA GeForce RTX 3050 Laptop (4GB GDDR6 VRAM, 4095 MiB) | "
                "Intel Core i5-12500H (4 P-cores + 8 E-cores) | 16GB System RAM (WSL2 Ubuntu 24.04 LTS)"
            ),
            "",
            "| Rank | Candidate / Model Pair | Intel Score | Speed (TPS) | VRAM (MiB) | Headroom | Value Score | Pareto? | Verdict |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]

        for s in scores:
            p_badge = "**Yes (SOTA)**" if s.is_pareto_optimal else "No"
            lines.append(
                f"| **#{s.rank}** | `{s.display_name}` | {s.intel_score:.1f} | "
                f"{s.effective_tps:.1f} tok/s | {s.vram_used_mb:.0f} | "
                f"{s.vram_headroom_mb:+.0f} | **{s.total_score:.1f}** | {p_badge} | {s.verdict} |"
            )

        lines.extend(
            [
                "",
                "### Pareto-Optimal Frontier Analysis",
                "",
            ]
        )

        pareto_list = [s for s in scores if s.is_pareto_optimal]
        for p in pareto_list:
            lines.append(
                f"- **`{p.display_name}`** (Value: {p.total_score:.1f}): "
                f"TPS: {p.effective_tps:.1f}, Intel: {p.intel_score:.1f}, VRAM: {p.vram_used_mb:.0f} MiB. "
                f"{p.verdict}. Advantages: {', '.join(p.pros) if p.pros else 'N/A'}"
            )

        return "\n".join(lines)

    def export_json(self, scores: list[EvaluationScore]) -> dict[str, Any]:
        """Export scored catalog to structured dictionary/JSON representation."""
        return {
            "evaluation_weights": {
                "w_intel": self.weights.w_intel,
                "w_speed": self.weights.w_speed,
                "w_mem": self.weights.w_mem,
                "w_spec": self.weights.w_spec,
                "subweights_intel": {
                    "mmlu": self.weights.w_mmlu,
                    "gsm8k": self.weights.w_gsm8k,
                    "humaneval": self.weights.w_humaneval,
                    "arena_hard": self.weights.w_arena_hard,
                },
                "vram_budget_mb": self.weights.vram_budget_mb,
            },
            "speculative_theory": self.explain_speculative_phenomenon(),
            "rankings": [
                {
                    "rank": s.rank,
                    "candidate_id": s.candidate_id,
                    "display_name": s.display_name,
                    "is_speculative_pair": s.is_speculative_pair,
                    "target_name": s.target_name,
                    "drafter_name": s.drafter_name,
                    "intel_score": s.intel_score,
                    "speed_score": s.speed_score,
                    "mem_score": s.mem_score,
                    "spec_score": s.spec_score,
                    "total_score": s.total_score,
                    "effective_tps": s.effective_tps,
                    "speedup_ratio": s.speedup_ratio,
                    "vram_used_mb": s.vram_used_mb,
                    "vram_headroom_mb": s.vram_headroom_mb,
                    "fits_vram": s.fits_vram,
                    "is_pareto_optimal": s.is_pareto_optimal,
                    "quality_retention": s.quality_retention,
                    "verdict": s.verdict,
                    "pros": s.pros,
                    "cons": s.cons,
                }
                for s in scores
            ],
        }

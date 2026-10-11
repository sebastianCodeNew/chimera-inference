"""Unit tests for Advanced SOTA Sampling Engine."""

import numpy as np
import pytest

from src.sampling import Sampler, SamplingParams

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def test_sampling_params_validation():
    # Valid default
    p = SamplingParams()
    assert p.is_greedy is False
    assert p.has_penalties is False

    # Greedy when temperature == 0.0
    p_greedy = SamplingParams(temperature=0.0)
    assert p_greedy.is_greedy is True

    # Invalid validations
    with pytest.raises(ValueError, match="temperature must be >= 0.0"):
        SamplingParams(temperature=-0.5)

    with pytest.raises(ValueError, match="top_k must be >= 0"):
        SamplingParams(top_k=-1)

    with pytest.raises(ValueError, match="top_p must be in"):
        SamplingParams(top_p=0.0)

    with pytest.raises(ValueError, match="top_p must be in"):
        SamplingParams(top_p=1.5)

    with pytest.raises(ValueError, match="min_p must be in"):
        SamplingParams(min_p=-0.1)

    with pytest.raises(ValueError, match="min_p must be in"):
        SamplingParams(min_p=1.2)

    with pytest.raises(ValueError, match="repetition_penalty must be >= 0.0"):
        SamplingParams(repetition_penalty=-1.0)


def test_greedy_sampling_numpy():
    logits = np.array([1.0, 5.0, 2.0, -3.0], dtype=np.float32)
    params = SamplingParams(temperature=0.0)
    tok = Sampler.sample(logits, params)
    assert tok == 1


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_greedy_sampling_torch():
    logits = torch.tensor([1.0, 2.0, 8.0, 0.5], dtype=torch.float32)
    params = SamplingParams(temperature=0.0)
    tok = Sampler.sample(logits, params)
    assert tok == 2


def test_repetition_penalty_numpy():
    logits = np.array([4.0, 4.0, 1.0], dtype=np.float32)
    # Repeated token 0 should have logit reduced
    penalized = Sampler.apply_penalties(
        logits, context_tokens=[0], repetition_penalty=2.0
    )
    assert penalized[0] == 2.0
    assert penalized[1] == 4.0
    assert penalized[2] == 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_repetition_penalty_torch():
    logits = torch.tensor([5.0, 5.0, -2.0], dtype=torch.float32)
    # Token 0 positive -> divided by 2.0 = 2.5
    # Token 2 negative -> multiplied by 2.0 = -4.0
    penalized = Sampler.apply_penalties(
        logits, context_tokens=[0, 2], repetition_penalty=2.0
    )
    assert float(penalized[0]) == 2.5
    assert float(penalized[1]) == 5.0
    assert float(penalized[2]) == -4.0


def test_frequency_and_presence_penalty():
    logits = np.array([10.0, 10.0], dtype=np.float32)
    # Token 0 appears 3 times, token 1 appears 0 times
    context = [0, 0, 0]
    penalized = Sampler.apply_penalties(
        logits,
        context_tokens=context,
        frequency_penalty=1.5,
        presence_penalty=0.5,
    )
    # Token 0: 10.0 - 1.5*3 - 0.5 = 5.0
    assert np.isclose(penalized[0], 5.0)
    assert penalized[1] == 10.0


def test_top_k_filtering():
    logits = np.array([1.0, 10.0, 5.0, 2.0, 8.0], dtype=np.float32)
    # Top 2 are indices 1 (10.0) and 4 (8.0)
    filtered = Sampler.apply_top_k(logits, top_k=2)
    assert filtered[1] == 10.0
    assert filtered[4] == 8.0
    assert np.isneginf(filtered[0])
    assert np.isneginf(filtered[2])
    assert np.isneginf(filtered[3])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_top_k_filtering_torch():
    logits = torch.tensor([1.0, 10.0, 5.0, 2.0, 8.0], dtype=torch.float32)
    filtered = Sampler.apply_top_k(logits, top_k=2)
    assert filtered[1] == 10.0
    assert filtered[4] == 8.0
    assert torch.isneginf(filtered[0])
    assert torch.isneginf(filtered[2])
    assert torch.isneginf(filtered[3])


def test_min_p_filtering():
    # Construct logits where token 0 is dominant
    logits = np.array([10.0, 7.0, 1.0, 0.0], dtype=np.float32)
    # min_p = 0.2 means any token with prob < 0.2 * max_prob is discarded
    filtered = Sampler.apply_min_p(logits, min_p=0.2)
    assert not np.isneginf(filtered[0])
    # Low logit tokens (1.0 and 0.0) will definitely have prob < 0.2 * max_prob
    assert np.isneginf(filtered[2])
    assert np.isneginf(filtered[3])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_min_p_filtering_torch():
    logits = torch.tensor([10.0, 7.0, 1.0, 0.0], dtype=torch.float32)
    filtered = Sampler.apply_min_p(logits, min_p=0.2)
    assert not torch.isneginf(filtered[0])
    assert torch.isneginf(filtered[2])
    assert torch.isneginf(filtered[3])


def test_top_p_filtering():
    logits = np.array([10.0, 1.0, 0.5, 0.1], dtype=np.float32)
    filtered = Sampler.apply_top_p(logits, top_p=0.9)
    # The dominant token represents >99% probability, others should be filtered
    assert filtered[0] == 10.0
    assert np.isneginf(filtered[1])
    assert np.isneginf(filtered[2])
    assert np.isneginf(filtered[3])


def test_sampling_determinism_with_seed():
    logits = np.array([2.0, 2.1, 1.9, 2.05], dtype=np.float32)
    params1 = SamplingParams(temperature=0.8, seed=42)
    params2 = SamplingParams(temperature=0.8, seed=42)

    samples1 = [Sampler.sample(logits, params1) for _ in range(5)]
    samples2 = [Sampler.sample(logits, params2) for _ in range(5)]
    assert samples1 == samples2


def test_sample_with_probs():
    logits = np.array([10.0, 0.0], dtype=np.float32)
    params = SamplingParams(temperature=1.0)
    tok, prob = Sampler.sample_with_probs(logits, params)
    assert tok == 0
    assert prob > 0.99


def test_sampling_persistent_generator_advances():
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    params = SamplingParams(temperature=1.0)
    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)

    seq1 = [Sampler.sample(logits, params, generator=rng1) for _ in range(10)]
    seq2 = [Sampler.sample(logits, params, generator=rng2) for _ in range(10)]

    assert seq1 == seq2
    # Ensure they are not all identical (i.e. PRNG actually advanced)
    assert len(set(seq1)) > 1


def test_min_p_and_top_p_with_neginf_inputs():
    logits = np.array([10.0, float("-inf"), 5.0, float("-inf")], dtype=np.float32)
    min_p_out = Sampler.apply_min_p(logits, min_p=0.1)
    assert not np.isnan(min_p_out).any()
    assert min_p_out[0] == 10.0

    top_p_out = Sampler.apply_top_p(logits, top_p=0.9)
    assert not np.isnan(top_p_out).any()
    assert top_p_out[0] == 10.0

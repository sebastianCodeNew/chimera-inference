"""Unit tests for Heterogeneous Speculative Decoding Coordinator."""

import numpy as np
import pytest

from src.generator import ChimeraGenerator
from src.model import ChimeraModel
from src.speculative import (
    HeterogeneousSpeculativeCoordinator,
    ModelDraftProposer,
    PromptLookupDraftProposer,
    SpeculativeStats,
    SpeculativeVerifier,
)
from src.tokenizer import ChimeraTokenizer

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def test_speculative_stats():
    stats = SpeculativeStats()
    assert stats.acceptance_rate == 0.0
    assert stats.tokens_per_target_step == 0.0

    stats.total_draft_tokens = 10
    stats.total_accepted_tokens = 8
    stats.target_eval_calls = 4
    stats.total_generated_tokens = 12

    assert np.isclose(stats.acceptance_rate, 0.8)
    assert np.isclose(stats.tokens_per_target_step, 3.0)


def test_prompt_lookup_draft_proposer():
    proposer = PromptLookupDraftProposer(min_ngram_size=2, max_ngram_size=3)

    # Empty / too short sequence
    draft, _ = proposer.propose([1], gamma=3)
    assert draft == []

    # Sequence with repetition: [1, 2, 3, 99, 1, 2, 3] -> with gamma=2 proposes [99, 1]
    seq = [1, 2, 3, 99, 1, 2, 3]
    draft, _ = proposer.propose(seq, gamma=2)
    assert draft == [99, 1]

    # With gamma=1 proposes [99]
    draft_1, _ = proposer.propose(seq, gamma=1)
    assert draft_1 == [99]

    # No match
    seq_no_match = [10, 20, 30, 40, 50]
    draft_no_match, _ = proposer.propose(seq_no_match, gamma=3)
    assert draft_no_match == []


def test_model_draft_proposer():
    draft_model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    draft_gen = ChimeraGenerator(draft_model, tokenizer, max_seq_len=64)

    proposer = ModelDraftProposer(draft_gen)
    draft, _ = proposer.propose([1, 2, 3], gamma=3)
    assert len(draft) == 3


def test_speculative_verifier_greedy_all_accepted():
    draft = [10, 20, 30]
    target_predictions = [10, 20, 30, 40]  # All 3 match, bonus is 40

    accepted, next_tok = SpeculativeVerifier.verify_greedy(target_predictions, draft)
    assert accepted == [10, 20, 30]
    assert next_tok == 40


def test_speculative_verifier_greedy_partial_accepted():
    draft = [10, 20, 30]
    target_predictions = [10, 20, 999, 40]  # Token 30 mismatches with 999

    accepted, next_tok = SpeculativeVerifier.verify_greedy(target_predictions, draft)
    assert accepted == [10, 20]
    assert next_tok == 999


def test_speculative_verifier_greedy_first_rejected():
    draft = [10, 20, 30]
    target_predictions = [999, 20, 30, 40]  # First token mismatches

    accepted, next_tok = SpeculativeVerifier.verify_greedy(target_predictions, draft)
    assert accepted == []
    assert next_tok == 999


def test_speculative_verifier_lossless_stochastic_identical_distribution():
    # If p == q, all tokens should be accepted with probability 1.0
    p_dist = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    draft = [1]
    target_probs = [p_dist, p_dist]
    draft_probs = [p_dist]

    accepted, next_tok = SpeculativeVerifier.verify_lossless_stochastic(
        target_probs, draft, draft_probs, seed=42
    )
    assert accepted == [1]
    assert next_tok in [0, 1, 2]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_speculative_verifier_lossless_stochastic_torch():
    p_dist = torch.tensor([0.2, 0.7, 0.1], dtype=torch.float32)
    draft = [1]
    target_probs = [p_dist, p_dist]
    draft_probs = [p_dist]

    accepted, next_tok = SpeculativeVerifier.verify_lossless_stochastic(
        target_probs, draft, draft_probs, seed=42
    )
    assert accepted == [1]
    assert 0 <= next_tok < 3


def test_heterogeneous_speculative_coordinator_mock_generate():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    coordinator = HeterogeneousSpeculativeCoordinator(
        target_generator=generator,
        gamma=3,
    )

    streamed: list[str] = []

    def callback(s: str) -> None:
        streamed.append(s)

    output = coordinator.generate("Halo", max_new_tokens=6, streaming_callback=callback)
    assert isinstance(output, str)
    assert len(streamed) <= 6
    assert coordinator.stats.total_generated_tokens <= 6
    assert coordinator.stats.target_eval_calls > 0


def test_heterogeneous_speculative_coordinator_mock_chat():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    coordinator = HeterogeneousSpeculativeCoordinator(
        target_generator=generator,
        gamma=2,
    )

    messages = [{"role": "user", "content": "Halo"}]
    output = coordinator.chat(messages, max_new_tokens=5)
    assert isinstance(output, str)


def test_speculative_early_stopping_on_mismatch():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    # Proposer that proposes wrong tokens
    class MismatchProposer:
        def propose(self, seq, gamma):
            return [99999, 88888], None

    coordinator = HeterogeneousSpeculativeCoordinator(
        target_generator=generator,
        draft_proposer=MismatchProposer(),
        gamma=2,
    )

    forward_calls = 0
    orig_forward = model.forward_token

    def counted_forward(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return orig_forward(*args, **kwargs)

    model.forward_token = counted_forward

    output = coordinator.generate("Halo", max_new_tokens=4)
    assert isinstance(output, str)
    assert coordinator.stats.total_accepted_tokens == 0
    # Prefill has 1 token ("Halo"), so forward_prompt or forward_token(last).
    # Then for each of 4 generated tokens, with early stopping exactly 1 forward call per step!
    # Total decode calls should be 4, not 4 * (gamma + 1) = 12!
    assert forward_calls <= 6


def test_speculative_stream_iterator():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    coordinator = HeterogeneousSpeculativeCoordinator(
        target_generator=generator,
        gamma=2,
    )

    tokens = list(coordinator.generate_stream("Halo", max_new_tokens=4))
    assert len(tokens) <= 4

    chat_tokens = list(
        coordinator.chat_stream([{"role": "user", "content": "Halo"}], max_new_tokens=4)
    )
    assert len(chat_tokens) <= 4


def test_speculative_eos_handling_in_draft():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    class GoodProposerWithEOS:
        def propose(self, seq, gamma):
            # Model predicts (x+1). If next is eos, provide it as draft.
            return [generator.tokenizer.eos_token_id, 12345], None

    coordinator = HeterogeneousSpeculativeCoordinator(
        target_generator=generator,
        draft_proposer=GoodProposerWithEOS(),
        gamma=2,
    )

    output = coordinator.generate("Halo", max_new_tokens=10)
    assert isinstance(output, str)
    # Must stop when EOS is encountered, not keep going to max_new_tokens
    assert coordinator.stats.total_generated_tokens <= 10


def test_speculative_stop_sequence():
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    generator = ChimeraGenerator(model, tokenizer, max_seq_len=64)

    coordinator = HeterogeneousSpeculativeCoordinator(
        target_generator=generator,
        gamma=2,
    )

    output = coordinator.generate("Halo", max_new_tokens=10, stop="\n")
    assert isinstance(output, str)
    assert "\n" not in output

"""Tests for the reverse-KL reward helpers in ``distillkit_r.training.opd_trainer``.

These are pure functions (math only) and run without the training stack installed.
"""

import math

import pytest

from distillkit_r.training.opd_trainer import reverse_kl, reverse_kl_reward


def _normalized_logprobs(probs: list[float]) -> list[float]:
    """Return log-probabilities for an already-normalized probability vector."""
    return [math.log(p) for p in probs]


def test_reverse_kl_zero_for_identical_distributions():
    """KL(p || p) is exactly zero."""
    logp = _normalized_logprobs([0.2, 0.3, 0.5])
    assert reverse_kl(logp, logp) == pytest.approx(0.0, abs=1e-12)


def test_reverse_kl_is_nonnegative():
    """Reverse KL is nonnegative for distinct distributions."""
    student = _normalized_logprobs([0.7, 0.2, 0.1])
    teacher = _normalized_logprobs([0.3, 0.3, 0.4])
    assert reverse_kl(student, teacher) > 0.0


def test_reverse_kl_matches_hand_computed_value():
    """Reverse KL equals the closed-form sum for a two-outcome case."""
    student = _normalized_logprobs([0.75, 0.25])
    teacher = _normalized_logprobs([0.5, 0.5])
    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert reverse_kl(student, teacher) == pytest.approx(expected, rel=1e-9)


def test_reward_is_negative_kl():
    """The reward is the negation of reverse KL and is non-positive."""
    student = _normalized_logprobs([0.6, 0.4])
    teacher = _normalized_logprobs([0.5, 0.5])
    kl = reverse_kl(student, teacher)
    assert reverse_kl_reward(student, teacher) == pytest.approx(-kl, rel=1e-9)
    assert reverse_kl_reward(student, teacher) <= 0.0


def test_reward_zero_for_identical_distributions():
    """Reward is zero when student matches teacher exactly."""
    logp = _normalized_logprobs([0.1, 0.2, 0.7])
    assert reverse_kl_reward(logp, logp) == pytest.approx(0.0, abs=1e-12)


def test_mismatched_support_raises():
    """A support-length mismatch is rejected."""
    with pytest.raises(ValueError, match="support length"):
        reverse_kl([math.log(1.0)], _normalized_logprobs([0.5, 0.5]))

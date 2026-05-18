"""Scoring helpers for deterministic and probabilistic forecasts."""

from __future__ import annotations

import math


def bias(actual: list[float], predicted: list[float]) -> float:
    """Return mean signed forecast error, predicted minus actual."""
    _validate_pairs(actual, predicted)
    return sum(predicted[i] - actual[i] for i in range(len(actual))) / len(actual)


def interval_coverage(actual: list[float], lower: list[float], upper: list[float]) -> float:
    """Return share of actual values inside inclusive intervals."""
    _validate_pairs(actual, lower)
    _validate_pairs(actual, upper)
    covered = sum(lower[i] <= actual[i] <= upper[i] for i in range(len(actual)))
    return covered / len(actual)


def interval_score(
    actual: list[float], lower: list[float], upper: list[float], *, alpha: float = 0.2
) -> float:
    """Return mean central prediction interval score."""
    _validate_pairs(actual, lower)
    _validate_pairs(actual, upper)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    total = 0.0
    for i in range(len(actual)):
        width = upper[i] - lower[i]
        below = max(lower[i] - actual[i], 0.0)
        above = max(actual[i] - upper[i], 0.0)
        total += width + (2 / alpha) * below + (2 / alpha) * above
    return total / len(actual)


def probability_for_actual_bin(bin_probs: dict[int, float], actual_high_f: float) -> float:
    """Return probability assigned to the rounded observed high bin."""
    observed_bin = int(math.floor(actual_high_f + 0.5))
    return float(bin_probs.get(observed_bin, 0.0))


def brier_score_for_bin(
    bin_probs: dict[int, float], actual_high_f: float, candidate_bin: int
) -> float:
    """Return one-vs-rest Brier score for a candidate temperature bin."""
    observed_bin = int(math.floor(actual_high_f + 0.5))
    outcome = 1.0 if observed_bin == candidate_bin else 0.0
    probability = float(bin_probs.get(candidate_bin, 0.0))
    return (probability - outcome) ** 2


def _validate_pairs(left: list[float], right: list[float]) -> None:
    if not left:
        raise ValueError("inputs must not be empty")
    if len(left) != len(right):
        raise ValueError("inputs must have the same length")

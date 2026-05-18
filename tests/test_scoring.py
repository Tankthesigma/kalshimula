import pytest

from src.models.scoring import (
    bias,
    brier_score_for_bin,
    interval_coverage,
    interval_score,
    probability_for_actual_bin,
)


def test_bias_returns_predicted_minus_actual_mean() -> None:
    assert bias([70, 80], [72, 79]) == pytest.approx(0.5)


def test_interval_coverage_counts_inclusive_hits() -> None:
    assert interval_coverage([70, 80, 90], [69, 81, 89], [71, 85, 89]) == pytest.approx(
        1 / 3
    )


def test_interval_score_penalizes_misses() -> None:
    score = interval_score([70], [72], [74], alpha=0.2)

    assert score == pytest.approx(22.0)


def test_interval_score_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        interval_score([70], [69], [71], alpha=1.0)


def test_probability_for_actual_bin_uses_half_up_rounding() -> None:
    assert probability_for_actual_bin({70: 0.2, 71: 0.8}, 70.5) == pytest.approx(0.8)


def test_brier_score_for_bin_scores_one_vs_rest() -> None:
    assert brier_score_for_bin({71: 0.8}, 70.5, 71) == pytest.approx(0.04)
    assert brier_score_for_bin({70: 0.2}, 70.5, 70) == pytest.approx(0.04)

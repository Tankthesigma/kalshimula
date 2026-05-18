import pytest

from src.models.baselines import (
    climatology_prediction,
    mean_absolute_error,
    root_mean_squared_error,
)


def test_mean_absolute_error() -> None:
    assert mean_absolute_error([70, 72, 74], [71, 70, 74]) == pytest.approx(1.0)


def test_root_mean_squared_error() -> None:
    assert root_mean_squared_error([70, 72], [73, 72]) == pytest.approx(2.1213203436)


def test_climatology_prediction_uses_mean() -> None:
    assert climatology_prediction([60, 70, 80]) == pytest.approx(70)


def test_baseline_helpers_reject_empty_inputs() -> None:
    with pytest.raises(ValueError):
        mean_absolute_error([], [])
    with pytest.raises(ValueError):
        climatology_prediction([])


def test_baseline_helpers_reject_length_mismatch() -> None:
    with pytest.raises(ValueError):
        mean_absolute_error([1], [1, 2])

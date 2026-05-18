"""Small baseline scoring helpers."""

from __future__ import annotations

import math


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    """Return mean absolute error for paired actual/predicted values."""
    _validate_pairs(actual, predicted)
    return sum(abs(actual[i] - predicted[i]) for i in range(len(actual))) / len(actual)


def root_mean_squared_error(actual: list[float], predicted: list[float]) -> float:
    """Return root mean squared error for paired actual/predicted values."""
    _validate_pairs(actual, predicted)
    mse = sum((actual[i] - predicted[i]) ** 2 for i in range(len(actual))) / len(actual)
    return math.sqrt(mse)


def climatology_prediction(history: list[float]) -> float:
    """Use the historical mean as a tiny climatology baseline."""
    if not history:
        raise ValueError("history must not be empty")
    return sum(history) / len(history)


def _validate_pairs(actual: list[float], predicted: list[float]) -> None:
    if not actual:
        raise ValueError("actual must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")

"""Forecast/actual joining helpers for supervised datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.datasets.training import TrainingExample, make_training_example


@dataclass(frozen=True)
class ForecastRecord:
    """One point forecast record ready to join with an actual."""

    city: str
    target_date: date
    source: str
    forecast_high_f: float | None


@dataclass(frozen=True)
class ActualRecord:
    """One observed high record ready to join with forecasts."""

    city: str
    target_date: date
    actual_high_f: float | None


def join_forecasts_to_actuals(
    forecasts: list[ForecastRecord], actuals: list[ActualRecord]
) -> list[TrainingExample]:
    """Create training examples where forecast and actual values are present."""
    actual_by_key = {
        (actual.city, actual.target_date): actual.actual_high_f for actual in actuals
    }
    examples: list[TrainingExample] = []
    for forecast in forecasts:
        actual = actual_by_key.get((forecast.city, forecast.target_date))
        if forecast.forecast_high_f is None or actual is None:
            continue
        examples.append(
            make_training_example(
                city=forecast.city,
                target_date=forecast.target_date,
                source=forecast.source,
                forecast_high_f=forecast.forecast_high_f,
                actual_high_f=actual,
            )
        )
    return examples

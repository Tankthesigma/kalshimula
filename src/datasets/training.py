"""Training dataset primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class TrainingExample:
    """One supervised row comparing a forecast against the observed high."""

    city: str
    target_date: date
    source: str
    forecast_high_f: float
    actual_high_f: float
    error_f: float


def make_training_example(
    *,
    city: str,
    target_date: date,
    source: str,
    forecast_high_f: float,
    actual_high_f: float,
) -> TrainingExample:
    """Create a supervised example with signed forecast error."""
    forecast = float(forecast_high_f)
    actual = float(actual_high_f)
    return TrainingExample(
        city=city,
        target_date=target_date,
        source=source,
        forecast_high_f=forecast,
        actual_high_f=actual,
        error_f=forecast - actual,
    )


def examples_to_dataframe(examples: list[TrainingExample]) -> pd.DataFrame:
    """Convert training examples into a stable tabular shape."""
    columns = [
        "city",
        "target_date",
        "source",
        "forecast_high_f",
        "actual_high_f",
        "error_f",
    ]
    rows = [asdict(example) for example in examples]
    return pd.DataFrame(rows, columns=columns)

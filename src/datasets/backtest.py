"""Backtest dataset helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class BacktestRow:
    """One backtest row for a forecast distribution and realized high."""

    city: str
    target_date: date
    source: str
    point_f: float
    actual_high_f: float
    absolute_error_f: float


def make_backtest_row(
    *,
    city: str,
    target_date: date,
    source: str,
    point_f: float,
    actual_high_f: float,
) -> BacktestRow:
    """Create a backtest row with absolute point forecast error."""
    point = float(point_f)
    actual = float(actual_high_f)
    return BacktestRow(
        city=city,
        target_date=target_date,
        source=source,
        point_f=point,
        actual_high_f=actual,
        absolute_error_f=abs(point - actual),
    )


def backtest_rows_to_dataframe(rows: list[BacktestRow]) -> pd.DataFrame:
    """Convert backtest rows into a stable tabular shape."""
    columns = [
        "city",
        "target_date",
        "source",
        "point_f",
        "actual_high_f",
        "absolute_error_f",
    ]
    return pd.DataFrame([asdict(row) for row in rows], columns=columns)

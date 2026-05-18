"""Backtest summary helpers."""

from __future__ import annotations

import pandas as pd

from src.models.baselines import mean_absolute_error, root_mean_squared_error
from src.models.scoring import bias

SUMMARY_COLUMNS = ["city", "source", "n", "mae", "rmse", "bias"]


def summarize_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize point forecast performance by city and source."""
    required = {"city", "source", "actual_high_f", "point_f"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows = []
    for (city, source), group in df.groupby(["city", "source"], sort=True):
        actual = group["actual_high_f"].astype(float).tolist()
        predicted = group["point_f"].astype(float).tolist()
        rows.append(
            {
                "city": city,
                "source": source,
                "n": len(group),
                "mae": mean_absolute_error(actual, predicted),
                "rmse": root_mean_squared_error(actual, predicted),
                "bias": bias(actual, predicted),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

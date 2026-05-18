"""Empirical interval calibration helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INTERVAL_COLUMNS = [
    "city",
    "source",
    "n",
    "lower_error_f",
    "upper_error_f",
    "alpha",
]


def fit_empirical_intervals(rows: pd.DataFrame, *, alpha: float = 0.2) -> pd.DataFrame:
    """Fit empirical central error intervals by city/source."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    required = {"city", "source", "point_f", "actual_high_f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if rows.empty:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)

    df = rows.copy()
    df["error_f"] = df["actual_high_f"].astype(float) - df["point_f"].astype(float)
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    records = []
    for (city, source), group in df.groupby(["city", "source"], sort=True):
        errors = group["error_f"].astype(float)
        records.append(
            {
                "city": city,
                "source": source,
                "n": len(group),
                "lower_error_f": errors.quantile(lower_q),
                "upper_error_f": errors.quantile(upper_q),
                "alpha": alpha,
            }
        )
    return pd.DataFrame(records, columns=INTERVAL_COLUMNS)


def apply_empirical_intervals(rows: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Add calibrated lower/upper point forecast bounds."""
    required = {"city", "source", "point_f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    interval_required = {"city", "source", "lower_error_f", "upper_error_f"}
    interval_missing = interval_required - set(intervals.columns)
    if interval_missing:
        raise ValueError(f"missing interval columns: {sorted(interval_missing)}")

    merged = rows.merge(
        intervals[["city", "source", "lower_error_f", "upper_error_f"]],
        on=["city", "source"],
        how="left",
    )
    merged["lower_error_f"] = merged["lower_error_f"].fillna(0.0)
    merged["upper_error_f"] = merged["upper_error_f"].fillna(0.0)
    merged["interval_lower_f"] = merged["point_f"].astype(float) + merged["lower_error_f"]
    merged["interval_upper_f"] = merged["point_f"].astype(float) + merged["upper_error_f"]
    return merged


def write_interval_table(input_path: Path, output_path: Path, *, alpha: float = 0.2) -> pd.DataFrame:
    """Fit and write empirical intervals from collected rows CSV."""
    rows = _read_rows(input_path)
    intervals = fit_empirical_intervals(rows, alpha=alpha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(output_path, index=False)
    return intervals


def _read_rows(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    parse_dates = ["target_date"] if "target_date" in columns else None
    return pd.read_csv(path, parse_dates=parse_dates)

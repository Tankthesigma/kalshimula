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
    """Fit empirical central residual intervals by city/source.

    ``lower_error_f`` and ``upper_error_f`` are quantiles of
    ``actual_high_f - point_f`` and are added to a point forecast to produce
    bounds.
    """
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
    """Add calibrated lower/upper forecast bounds.

    Existing ``interval_lower_f`` / ``interval_upper_f`` columns are preserved
    as raw point aliases for compatibility. Explicit ``*_raw_f`` and
    ``*_corrected_f`` columns make the centering contract visible to consumers.
    """
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
    point = merged["point_f"].astype(float)
    lower_error = merged["lower_error_f"].astype(float)
    upper_error = merged["upper_error_f"].astype(float)
    merged["interval_lower_raw_f"] = point + lower_error
    merged["interval_upper_raw_f"] = point + upper_error
    merged["interval_lower_f"] = merged["interval_lower_raw_f"]
    merged["interval_upper_f"] = merged["interval_upper_raw_f"]
    if "corrected_point_f" in merged.columns:
        corrected = merged["corrected_point_f"].astype(float)
        correction = corrected - point
        merged["interval_lower_corrected_f"] = corrected + lower_error - correction
        merged["interval_upper_corrected_f"] = corrected + upper_error - correction
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

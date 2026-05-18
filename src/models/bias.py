"""Bias correction helpers for point forecasts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BIAS_COLUMNS = ["city", "source", "n", "mean_error_f", "bias_correction_f"]


def fit_bias_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Fit mean-error bias corrections by city/source."""
    required = {"city", "source", "point_f", "actual_high_f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if rows.empty:
        return pd.DataFrame(columns=BIAS_COLUMNS)

    df = rows.copy()
    df["error_f"] = df["point_f"].astype(float) - df["actual_high_f"].astype(float)
    grouped = df.groupby(["city", "source"], sort=True)["error_f"]
    out = grouped.agg(n="size", mean_error_f="mean").reset_index()
    out["bias_correction_f"] = -out["mean_error_f"]
    return out[BIAS_COLUMNS]


def apply_bias_correction(rows: pd.DataFrame, bias_table: pd.DataFrame) -> pd.DataFrame:
    """Add corrected_point_f using city/source bias corrections."""
    required = {"city", "source", "point_f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    correction_required = {"city", "source", "bias_correction_f"}
    correction_missing = correction_required - set(bias_table.columns)
    if correction_missing:
        raise ValueError(f"missing bias columns: {sorted(correction_missing)}")

    merged = rows.merge(
        bias_table[["city", "source", "bias_correction_f"]],
        on=["city", "source"],
        how="left",
    )
    merged["bias_correction_f"] = merged["bias_correction_f"].fillna(0.0)
    merged["corrected_point_f"] = (
        merged["point_f"].astype(float) + merged["bias_correction_f"].astype(float)
    )
    return merged


def write_bias_table(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Fit and write a bias table from collected rows CSV."""
    rows = _read_rows(input_path)
    table = fit_bias_table(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def _read_rows(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    parse_dates = ["target_date"] if "target_date" in columns else None
    return pd.read_csv(path, parse_dates=parse_dates)

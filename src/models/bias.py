"""Bias correction helpers for point forecasts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BIAS_COLUMNS = ["city", "source", "n", "mean_error_f", "bias_correction_f"]
SEASONAL_BIAS_COLUMNS = [
    "city",
    "source",
    "month",
    "n",
    "mean_error_f",
    "bias_correction_f",
]


def fit_bias_table(
    rows: pd.DataFrame, *, group_month: bool = False, include_fallback: bool = True
) -> pd.DataFrame:
    """Fit mean-error bias corrections.

    Default behavior groups by city/source. Internally ``mean_error_f`` is
    ``point_f - actual_high_f``; ``bias_correction_f`` is the amount to add to
    the point forecast. With ``group_month=True`` the table
    includes city/source/month rows plus optional city/source fallback rows where
    ``month`` is missing. The fallback keeps future months usable when the train
    split does not contain every calendar month.
    """
    required = {"city", "source", "point_f", "actual_high_f"}
    if group_month:
        required.add("target_date")
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if rows.empty:
        return pd.DataFrame(columns=SEASONAL_BIAS_COLUMNS if group_month else BIAS_COLUMNS)

    df = rows.copy()
    df["error_f"] = df["point_f"].astype(float) - df["actual_high_f"].astype(float)
    if not group_month:
        return _fit_grouped_bias(df, ["city", "source"], BIAS_COLUMNS)

    df["month"] = pd.to_datetime(df["target_date"]).dt.month.astype("Int64")
    monthly = _fit_grouped_bias(df, ["city", "source", "month"], SEASONAL_BIAS_COLUMNS)
    if not include_fallback:
        return monthly

    fallback = _fit_grouped_bias(df, ["city", "source"], BIAS_COLUMNS)
    fallback["month"] = pd.NA
    fallback = fallback[SEASONAL_BIAS_COLUMNS]
    return pd.concat([monthly, fallback], ignore_index=True)


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

    if "month" in bias_table.columns:
        return _apply_seasonal_bias_correction(rows, bias_table)

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


def _fit_grouped_bias(df: pd.DataFrame, group_cols: list[str], columns: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, sort=True)["error_f"]
    out = grouped.agg(n="size", mean_error_f="mean").reset_index()
    out["bias_correction_f"] = -out["mean_error_f"]
    return out[columns]


def _apply_seasonal_bias_correction(rows: pd.DataFrame, bias_table: pd.DataFrame) -> pd.DataFrame:
    if "target_date" not in rows.columns:
        raise ValueError("rows must include target_date for seasonal bias correction")

    merged = rows.copy()
    merged["_bias_month"] = pd.to_datetime(merged["target_date"]).dt.month.astype("Int64")
    seasonal = bias_table[bias_table["month"].notna()].copy()
    seasonal["month"] = seasonal["month"].astype("Int64")
    fallback = bias_table[bias_table["month"].isna()].copy()

    merged = merged.merge(
        seasonal[["city", "source", "month", "bias_correction_f"]].rename(
            columns={"month": "_bias_month", "bias_correction_f": "_seasonal_bias_correction_f"}
        ),
        on=["city", "source", "_bias_month"],
        how="left",
    )
    if not fallback.empty:
        merged = merged.merge(
            fallback[["city", "source", "bias_correction_f"]].rename(
                columns={"bias_correction_f": "_fallback_bias_correction_f"}
            ),
            on=["city", "source"],
            how="left",
        )
    else:
        merged["_fallback_bias_correction_f"] = pd.NA

    merged["bias_correction_f"] = (
        merged["_seasonal_bias_correction_f"]
        .combine_first(merged["_fallback_bias_correction_f"])
        .fillna(0.0)
    )
    merged["corrected_point_f"] = (
        merged["point_f"].astype(float) + merged["bias_correction_f"].astype(float)
    )
    return merged.drop(
        columns=[
            "_bias_month",
            "_seasonal_bias_correction_f",
            "_fallback_bias_correction_f",
        ]
    )

"""Residual diagnostics for model evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import pandas as pd

RESIDUAL_SUMMARY_COLUMNS = [
    "city",
    "source",
    "month",
    "n",
    "mae_raw",
    "rmse_raw",
    "bias_raw",
    "residual_std_raw",
    "mae_corrected",
    "rmse_corrected",
    "bias_corrected",
    "residual_std_corrected",
]


@dataclass(frozen=True)
class ResidualDiagnostics:
    """Residual summaries for source and monthly model checks."""

    source_summary: pd.DataFrame
    monthly_summary: pd.DataFrame


def summarize_residuals(rows: pd.DataFrame, *, group_month: bool = False) -> pd.DataFrame:
    """Summarize point forecast residuals by city/source and optionally month.

    Residuals use the same public sign as evaluation bias: predicted minus actual,
    so positive values mean the forecast is too high.
    """
    required = {"city", "source", "actual_high_f", "point_f"}
    if group_month:
        required.add("target_date")
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    if rows.empty:
        return pd.DataFrame(columns=RESIDUAL_SUMMARY_COLUMNS)

    df = rows.copy()
    if group_month:
        df["month"] = pd.to_datetime(df["target_date"]).dt.month.astype("Int64")
    else:
        df["month"] = pd.NA

    records = []
    group_keys = ["city", "source", "month"] if group_month else ["city", "source"]
    for key, group in df.groupby(group_keys, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        key_values = dict(zip(group_keys, key, strict=True))
        city = key_values["city"]
        source = key_values["source"]
        month = key_values.get("month", pd.NA)
        records.append(
            {
                "city": city,
                "source": source,
                "month": month,
                **_residual_metrics(group),
            }
        )

    return pd.DataFrame(records, columns=RESIDUAL_SUMMARY_COLUMNS)


def build_residual_diagnostics(rows: pd.DataFrame) -> ResidualDiagnostics:
    """Build source-level and month-level residual diagnostic tables."""
    monthly = (
        summarize_residuals(rows, group_month=True)
        if "target_date" in rows.columns
        else pd.DataFrame(columns=RESIDUAL_SUMMARY_COLUMNS)
    )
    return ResidualDiagnostics(
        source_summary=summarize_residuals(rows),
        monthly_summary=monthly,
    )


def write_residual_diagnostics(*, input_path: Path, output_dir: Path) -> ResidualDiagnostics:
    """Read model rows and write residual diagnostic CSVs."""
    rows = _read_rows(input_path)
    diagnostics = build_residual_diagnostics(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.source_summary.to_csv(output_dir / "source_residuals.csv", index=False)
    diagnostics.monthly_summary.to_csv(output_dir / "monthly_residuals.csv", index=False)
    return diagnostics


def _residual_metrics(group: pd.DataFrame) -> dict[str, object]:
    valid = group[["actual_high_f", "point_f"]].dropna()
    raw = _metrics(valid["point_f"] - valid["actual_high_f"]) if not valid.empty else _empty_metrics()

    corrected = _empty_metrics()
    if "corrected_point_f" in group.columns:
        corrected_rows = group[["actual_high_f", "corrected_point_f"]].dropna()
        if not corrected_rows.empty:
            corrected = _metrics(
                corrected_rows["corrected_point_f"] - corrected_rows["actual_high_f"]
            )

    return {
        "n": int(raw["n"]),
        "mae_raw": raw["mae"],
        "rmse_raw": raw["rmse"],
        "bias_raw": raw["bias"],
        "residual_std_raw": raw["std"],
        "mae_corrected": corrected["mae"],
        "rmse_corrected": corrected["rmse"],
        "bias_corrected": corrected["bias"],
        "residual_std_corrected": corrected["std"],
    }


def _metrics(residual: pd.Series) -> dict[str, object]:
    values = residual.astype(float)
    return {
        "n": len(values),
        "mae": float(values.abs().mean()),
        "rmse": float(sqrt((values**2).mean())),
        "bias": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }


def _empty_metrics() -> dict[str, object]:
    return {"n": 0, "mae": pd.NA, "rmse": pd.NA, "bias": pd.NA, "std": pd.NA}


def _read_rows(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    parse_dates = ["target_date"] if "target_date" in columns else None
    return pd.read_csv(path, parse_dates=parse_dates)

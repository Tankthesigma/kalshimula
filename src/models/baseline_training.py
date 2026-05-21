"""Dependency-free baseline training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.backtest import summarize_backtest
from src.models.baselines import mean_absolute_error, root_mean_squared_error
from src.models.bias import apply_bias_correction, fit_bias_table
from src.models.scoring import bias, interval_coverage

EVALUATION_COLUMNS = [
    "city",
    "source",
    "n",
    "mae_raw",
    "rmse_raw",
    "bias_raw",
    "mae_corrected",
    "rmse_corrected",
    "bias_corrected",
    "interval_coverage_raw",
    "interval_width_raw",
    "interval_coverage_corrected",
    "interval_width_corrected",
]


@dataclass(frozen=True)
class BaselineTrainingResult:
    """Artifacts from fitting the baseline bias-correction model."""

    bias_table: pd.DataFrame
    evaluation: pd.DataFrame


def train_bias_baseline(rows: pd.DataFrame) -> BaselineTrainingResult:
    """Fit city/source bias correction and evaluate raw vs corrected predictions."""
    bias_table = fit_bias_table(rows)
    corrected = apply_bias_correction(rows, bias_table)
    evaluation = evaluate_corrected_predictions(corrected)
    return BaselineTrainingResult(bias_table=bias_table, evaluation=evaluation)


def evaluate_corrected_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    """Evaluate raw point forecasts against corrected point forecasts.

    Interval coverage/width columns are populated only when the corresponding
    interval columns are present. This keeps bias-only baseline training usable
    while making train/eval interval calibration visible in the same summary.
    """
    required = {"city", "source", "actual_high_f", "point_f", "corrected_point_f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if rows.empty:
        return pd.DataFrame(columns=EVALUATION_COLUMNS)

    records = []
    for (city, source), group in rows.groupby(["city", "source"], sort=True):
        actual = group["actual_high_f"].astype(float).tolist()
        raw = group["point_f"].astype(float).tolist()
        corrected = group["corrected_point_f"].astype(float).tolist()
        raw_interval = _interval_metrics(
            group,
            lower_col="interval_lower_raw_f"
            if "interval_lower_raw_f" in group.columns
            else "interval_lower_f",
            upper_col="interval_upper_raw_f"
            if "interval_upper_raw_f" in group.columns
            else "interval_upper_f",
        )
        corrected_interval = _interval_metrics(
            group,
            lower_col="interval_lower_corrected_f",
            upper_col="interval_upper_corrected_f",
        )
        records.append(
            {
                "city": city,
                "source": source,
                "n": len(group),
                "mae_raw": mean_absolute_error(actual, raw),
                "rmse_raw": root_mean_squared_error(actual, raw),
                "bias_raw": bias(actual, raw),
                "mae_corrected": mean_absolute_error(actual, corrected),
                "rmse_corrected": root_mean_squared_error(actual, corrected),
                "bias_corrected": bias(actual, corrected),
                "interval_coverage_raw": raw_interval["coverage"],
                "interval_width_raw": raw_interval["width"],
                "interval_coverage_corrected": corrected_interval["coverage"],
                "interval_width_corrected": corrected_interval["width"],
            }
        )
    return pd.DataFrame(records, columns=EVALUATION_COLUMNS)


def write_baseline_training_outputs(
    *, input_path: Path, bias_out: Path, evaluation_out: Path
) -> BaselineTrainingResult:
    """Train baseline artifacts from CSV and write outputs."""
    rows = _read_rows(input_path)
    result = train_bias_baseline(rows)
    bias_out.parent.mkdir(parents=True, exist_ok=True)
    evaluation_out.parent.mkdir(parents=True, exist_ok=True)
    result.bias_table.to_csv(bias_out, index=False)
    result.evaluation.to_csv(evaluation_out, index=False)
    return result


def summarize_raw_backtest(rows: pd.DataFrame) -> pd.DataFrame:
    """Expose raw backtest summary for baseline comparisons."""
    return summarize_backtest(rows)


def _read_rows(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    parse_dates = ["target_date"] if "target_date" in columns else None
    return pd.read_csv(path, parse_dates=parse_dates)


def _interval_metrics(group: pd.DataFrame, *, lower_col: str, upper_col: str) -> dict[str, object]:
    if lower_col not in group.columns or upper_col not in group.columns:
        return {"coverage": pd.NA, "width": pd.NA}

    interval_rows = group[["actual_high_f", lower_col, upper_col]].dropna()
    if interval_rows.empty:
        return {"coverage": 0.0, "width": 0.0}

    actual = interval_rows["actual_high_f"].astype(float).tolist()
    lower = interval_rows[lower_col].astype(float).tolist()
    upper = interval_rows[upper_col].astype(float).tolist()
    width = (interval_rows[upper_col].astype(float) - interval_rows[lower_col].astype(float)).mean()
    return {
        "coverage": interval_coverage(actual, lower, upper),
        "width": float(width),
    }

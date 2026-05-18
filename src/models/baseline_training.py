"""Dependency-free baseline training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.backtest import summarize_backtest
from src.models.baselines import mean_absolute_error, root_mean_squared_error
from src.models.bias import apply_bias_correction, fit_bias_table
from src.models.scoring import bias

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
    """Evaluate raw point forecasts against corrected point forecasts."""
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


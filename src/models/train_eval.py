"""Leakage-safe train/test evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.baseline_training import evaluate_corrected_predictions
from src.models.bias import apply_bias_correction, fit_bias_table
from src.models.diagnostics import build_residual_diagnostics
from src.models.intervals import apply_empirical_intervals, fit_empirical_intervals


@dataclass(frozen=True)
class TrainEvalResult:
    """Artifacts from fitting on train rows and evaluating on test rows."""

    train_rows: pd.DataFrame
    test_rows: pd.DataFrame
    bias_table: pd.DataFrame
    interval_table: pd.DataFrame
    corrected_test_rows: pd.DataFrame
    evaluation: pd.DataFrame
    source_residuals: pd.DataFrame
    monthly_residuals: pd.DataFrame


def split_rows_by_date(
    rows: pd.DataFrame, *, test_start: str | pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train/test by target_date, with test_start inclusive."""
    if "target_date" not in rows.columns:
        raise ValueError("rows must include target_date")
    df = rows.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    cutoff = pd.Timestamp(test_start)
    train = df[df["target_date"] < cutoff].copy()
    test = df[df["target_date"] >= cutoff].copy()
    return train, test


def train_eval_split(
    rows: pd.DataFrame, *, test_start: str | pd.Timestamp, alpha: float = 0.2
) -> TrainEvalResult:
    """Fit bias/intervals on train rows and evaluate corrected forecasts on test rows."""
    train, test = split_rows_by_date(rows, test_start=test_start)
    if train.empty:
        raise ValueError("train split is empty")
    if test.empty:
        raise ValueError("test split is empty")

    bias_table = fit_bias_table(train, group_month=True)
    interval_table = fit_empirical_intervals(train, alpha=alpha)
    corrected = apply_bias_correction(test, bias_table)
    corrected = apply_empirical_intervals(corrected, interval_table)
    evaluation = evaluate_corrected_predictions(corrected)
    residuals = build_residual_diagnostics(corrected)
    return TrainEvalResult(
        train_rows=train,
        test_rows=test,
        bias_table=bias_table,
        interval_table=interval_table,
        corrected_test_rows=corrected,
        evaluation=evaluation,
        source_residuals=residuals.source_summary,
        monthly_residuals=residuals.monthly_summary,
    )


def write_train_eval_outputs(
    *, input_path: Path, output_dir: Path, test_start: str, alpha: float = 0.2
) -> TrainEvalResult:
    """Run leakage-safe train/test evaluation and write CSV artifacts."""
    rows = pd.read_csv(input_path, parse_dates=["target_date"])
    result = train_eval_split(rows, test_start=test_start, alpha=alpha)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.train_rows.to_csv(output_dir / "train_rows.csv", index=False)
    result.test_rows.to_csv(output_dir / "test_rows.csv", index=False)
    result.bias_table.to_csv(output_dir / "bias_table.csv", index=False)
    result.interval_table.to_csv(output_dir / "interval_table.csv", index=False)
    result.corrected_test_rows.to_csv(output_dir / "corrected_test_rows.csv", index=False)
    result.evaluation.to_csv(output_dir / "evaluation.csv", index=False)
    result.source_residuals.to_csv(output_dir / "source_residuals.csv", index=False)
    result.monthly_residuals.to_csv(output_dir / "monthly_residuals.csv", index=False)
    return result

"""Leakage-safe train/test evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
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


def split_rows_by_month_stratified(
    rows: pd.DataFrame, *, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each city/source/month into train/test rows by date order.

    This is a diagnostic split for measuring month-aware calibration on short
    windows. It is not a substitute for the default chronological split.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    required = {"city", "source", "target_date"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = rows.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["_month"] = df["target_date"].dt.month.astype("Int64")
    df["_row_order"] = range(len(df))

    train_parts = []
    test_parts = []
    for _, group in df.sort_values(["target_date", "_row_order"]).groupby(
        ["city", "source", "_month"], sort=True
    ):
        if len(group) < 2:
            train_parts.append(group)
            continue
        test_count = min(max(1, ceil(len(group) * test_fraction)), len(group) - 1)
        train_parts.append(group.iloc[:-test_count])
        test_parts.append(group.iloc[-test_count:])

    train = pd.concat(train_parts, ignore_index=False) if train_parts else df.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=False) if test_parts else df.iloc[0:0]
    drop_cols = ["_month", "_row_order"]
    return (
        train.drop(columns=drop_cols).sort_values("target_date").copy(),
        test.drop(columns=drop_cols).sort_values("target_date").copy(),
    )


def train_eval_split(
    rows: pd.DataFrame,
    *,
    test_start: str | pd.Timestamp | None = None,
    alpha: float = 0.2,
    split_strategy: str = "date",
    test_fraction: float = 0.2,
    bias_strategy: str = "seasonal",
    bias_recent_days: int | None = None,
) -> TrainEvalResult:
    """Fit bias/intervals on train rows and evaluate corrected forecasts on test rows."""
    normalized_strategy = split_strategy.replace("-", "_")
    if normalized_strategy == "date":
        if test_start is None:
            raise ValueError("test_start is required for date split")
        train, test = split_rows_by_date(rows, test_start=test_start)
    elif normalized_strategy == "month_stratified":
        train, test = split_rows_by_month_stratified(rows, test_fraction=test_fraction)
    else:
        raise ValueError(f"unknown split_strategy: {split_strategy}")
    if train.empty:
        raise ValueError("train split is empty")
    if test.empty:
        raise ValueError("test split is empty")

    bias_table = _fit_bias_for_strategy(
        train,
        bias_strategy=bias_strategy,
        bias_recent_days=bias_recent_days,
    )
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
    *,
    input_path: Path,
    output_dir: Path,
    test_start: str | None = None,
    alpha: float = 0.2,
    split_strategy: str = "date",
    test_fraction: float = 0.2,
    bias_strategy: str = "seasonal",
    bias_recent_days: int | None = None,
) -> TrainEvalResult:
    """Run leakage-safe train/test evaluation and write CSV artifacts."""
    rows = pd.read_csv(input_path, parse_dates=["target_date"])
    result = train_eval_split(
        rows,
        test_start=test_start,
        alpha=alpha,
        split_strategy=split_strategy,
        test_fraction=test_fraction,
        bias_strategy=bias_strategy,
        bias_recent_days=bias_recent_days,
    )
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


def _fit_bias_for_strategy(
    train: pd.DataFrame,
    *,
    bias_strategy: str,
    bias_recent_days: int | None,
) -> pd.DataFrame:
    normalized_strategy = bias_strategy.replace("-", "_")
    if normalized_strategy == "seasonal":
        return fit_bias_table(train, group_month=True)
    if normalized_strategy == "global":
        return fit_bias_table(train)
    if normalized_strategy == "recent":
        recent = _recent_train_rows(train, bias_recent_days=bias_recent_days)
        return fit_bias_table(recent)
    raise ValueError(f"unknown bias_strategy: {bias_strategy}")


def _recent_train_rows(train: pd.DataFrame, *, bias_recent_days: int | None) -> pd.DataFrame:
    if bias_recent_days is None:
        raise ValueError("bias_recent_days is required when bias_strategy='recent'")
    if bias_recent_days < 1:
        raise ValueError("bias_recent_days must be at least 1")
    if "target_date" not in train.columns:
        raise ValueError("train rows must include target_date for recent bias correction")

    df = train.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    cutoff = df["target_date"].max() - pd.Timedelta(days=bias_recent_days - 1)
    recent = df[df["target_date"] >= cutoff].copy()
    if recent.empty:
        raise ValueError("recent bias training split is empty")
    return recent

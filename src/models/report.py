"""Model report assembly for collected backtest rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.backtest import summarize_backtest
from src.models.baseline_training import train_bias_baseline
from src.models.intervals import fit_empirical_intervals


@dataclass(frozen=True)
class ModelReport:
    """Report tables for a collected backtest dataset."""

    raw_summary: pd.DataFrame
    bias_table: pd.DataFrame
    corrected_evaluation: pd.DataFrame
    intervals: pd.DataFrame


def build_model_report(rows: pd.DataFrame, *, alpha: float = 0.2) -> ModelReport:
    """Build raw, corrected, and interval report tables."""
    raw_summary = summarize_backtest(rows)
    baseline = train_bias_baseline(rows)
    intervals = fit_empirical_intervals(rows, alpha=alpha)
    return ModelReport(
        raw_summary=raw_summary,
        bias_table=baseline.bias_table,
        corrected_evaluation=baseline.evaluation,
        intervals=intervals,
    )


def write_model_report(
    *, input_path: Path, output_dir: Path, alpha: float = 0.2
) -> ModelReport:
    """Build and write model report CSVs into an output directory."""
    rows = _read_rows(input_path)
    report = build_model_report(rows, alpha=alpha)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.raw_summary.to_csv(output_dir / "raw_summary.csv", index=False)
    report.bias_table.to_csv(output_dir / "bias_table.csv", index=False)
    report.corrected_evaluation.to_csv(
        output_dir / "corrected_evaluation.csv", index=False
    )
    report.intervals.to_csv(output_dir / "intervals.csv", index=False)
    return report


def _read_rows(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    parse_dates = ["target_date"] if "target_date" in columns else None
    return pd.read_csv(path, parse_dates=parse_dates)

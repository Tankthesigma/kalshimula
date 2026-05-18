"""Historical run orchestration for collection and model reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.batch_collect_cli import collect_many_cities, write_batch_outputs
from src.models.report import write_model_report
from src.models.train_eval import write_train_eval_outputs


@dataclass(frozen=True)
class HistoricalRunResult:
    """Paths and row counts produced by a historical run."""

    rows_path: Path
    summary_path: Path
    report_dir: Path
    train_eval_dir: Path
    n_rows: int
    n_summary_rows: int


def run_historical_pipeline(
    *,
    cities: list[str],
    start: date,
    end: date,
    test_start: date,
    out_dir: Path,
    cache_root: Path,
    alpha: float = 0.2,
) -> HistoricalRunResult:
    """Collect rows and write all model/report artifacts for a historical run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.csv"
    summary_path = out_dir / "summary.csv"
    report_dir = out_dir / "model_report"
    train_eval_dir = out_dir / "train_eval"

    rows = collect_many_cities(
        cities=cities,
        start=start,
        end=end,
        cache_root=cache_root,
    )
    summary = write_batch_outputs(rows, rows_path, summary_path)
    if rows.empty:
        _write_empty_artifact_dirs(report_dir, train_eval_dir)
    else:
        write_model_report(input_path=rows_path, output_dir=report_dir, alpha=alpha)
        write_train_eval_outputs(
            input_path=rows_path,
            output_dir=train_eval_dir,
            test_start=test_start.isoformat(),
            alpha=alpha,
        )

    return HistoricalRunResult(
        rows_path=rows_path,
        summary_path=summary_path,
        report_dir=report_dir,
        train_eval_dir=train_eval_dir,
        n_rows=len(rows),
        n_summary_rows=len(summary),
    )


def _write_empty_artifact_dirs(report_dir: Path, train_eval_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    train_eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(report_dir / "raw_summary.csv", index=False)
    pd.DataFrame().to_csv(report_dir / "bias_table.csv", index=False)
    pd.DataFrame().to_csv(report_dir / "corrected_evaluation.csv", index=False)
    pd.DataFrame().to_csv(report_dir / "intervals.csv", index=False)
    pd.DataFrame().to_csv(train_eval_dir / "evaluation.csv", index=False)

"""CSV persistence helpers for dataset foundations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.datasets.backtest import BacktestRow, backtest_rows_to_dataframe
from src.datasets.training import TrainingExample, examples_to_dataframe


def save_training_examples(path: Path, examples: list[TrainingExample]) -> None:
    """Save training examples to CSV with stable columns."""
    _write_csv(path, examples_to_dataframe(examples))


def load_training_examples(path: Path) -> pd.DataFrame:
    """Load training examples from CSV."""
    return pd.read_csv(path, parse_dates=["target_date"])


def save_backtest_rows(path: Path, rows: list[BacktestRow]) -> None:
    """Save backtest rows to CSV with stable columns."""
    _write_csv(path, backtest_rows_to_dataframe(rows))


def load_backtest_rows(path: Path) -> pd.DataFrame:
    """Load backtest rows from CSV."""
    return pd.read_csv(path, parse_dates=["target_date"])


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

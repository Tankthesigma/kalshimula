"""CLI for summarizing collected backtest CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models.backtest import summarize_backtest


def summarize_backtest_csv(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Read collected backtest rows, summarize them, and write CSV output."""
    df = pd.read_csv(input_path, parse_dates=["target_date"])
    summary = summarize_backtest(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="Summarize collected weather backtest rows.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    summary = summarize_backtest_csv(args.input, args.out)
    print(f"Wrote {len(summary)} summary rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

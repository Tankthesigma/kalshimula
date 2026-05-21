"""Batch CLI for collecting and summarizing multiple cities."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.collect import collect_backtest_rows
from src.datasets.backtest import backtest_rows_to_dataframe
from src.models.backtest import summarize_backtest


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_cities(value: str) -> list[str]:
    cities = [city.strip() for city in value.split(",") if city.strip()]
    if not cities:
        raise argparse.ArgumentTypeError("at least one city is required")
    return cities


def collect_many_cities(
    *,
    cities: list[str],
    start,
    end,
    cache_root: Path,
    openmeteo_mode: str = "naive",
) -> pd.DataFrame:
    """Collect backtest rows for multiple cities into one dataframe."""
    rows = []
    for city in cities:
        result = collect_backtest_rows(
            city=city,
            start=start,
            end=end,
            cache_root=cache_root,
            openmeteo_mode=openmeteo_mode,
        )
        rows.extend(result.rows)
    return backtest_rows_to_dataframe(rows)


def write_batch_outputs(df: pd.DataFrame, rows_path: Path, summary_path: Path) -> pd.DataFrame:
    """Write collected rows and grouped summary CSVs."""
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(rows_path, index=False)
    summary = summarize_backtest(df)
    summary.to_csv(summary_path, index=False)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="batch_collect",
        description="Collect and summarize cache-backed weather rows for multiple cities.",
    )
    parser.add_argument("--cities", required=True, type=_parse_cities)
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument("--rows-out", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--cache", default=Path(".cache/weather"), type=Path)
    parser.add_argument(
        "--openmeteo-mode",
        choices=["naive", "sources", "both"],
        default="naive",
        help=(
            "Historical Open-Meteo rows to collect: pooled naive baseline, "
            "individual model sources, or both."
        ),
    )
    args = parser.parse_args(argv)

    rows = collect_many_cities(
        cities=args.cities,
        start=args.start,
        end=args.end,
        cache_root=args.cache,
        openmeteo_mode=args.openmeteo_mode,
    )
    summary = write_batch_outputs(rows, args.rows_out, args.summary_out)
    print(
        f"Wrote {len(rows)} rows to {args.rows_out} and "
        f"{len(summary)} summary rows to {args.summary_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

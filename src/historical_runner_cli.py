"""CLI for running the historical collection/report pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.historical_runner import run_historical_pipeline


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_cities(value: str) -> list[str]:
    cities = [city.strip() for city in value.split(",") if city.strip()]
    if not cities:
        raise argparse.ArgumentTypeError("at least one city is required")
    return cities


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="historical_runner",
        description="Run historical collection, reports, and train/test evaluation.",
    )
    parser.add_argument("--cities", required=True, type=_parse_cities)
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument("--test-start", required=True, type=_parse_date)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cache", default=Path(".cache/weather"), type=Path)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--chunk-days", default=1, type=int)
    args = parser.parse_args(argv)

    result = run_historical_pipeline(
        cities=args.cities,
        start=args.start,
        end=args.end,
        test_start=args.test_start,
        out_dir=args.out_dir,
        cache_root=args.cache,
        alpha=args.alpha,
        progress=print,
        workers=args.workers,
        chunk_days=args.chunk_days,
    )
    print(
        f"Wrote {result.n_rows} rows and {result.n_summary_rows} summary rows "
        f"({result.n_skipped} skipped, {result.n_errors} errors) "
        f"under {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

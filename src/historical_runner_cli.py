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
    parser.add_argument("--validation-start", type=_parse_date)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cache", default=Path(".cache/weather"), type=Path)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument(
        "--bias-strategy",
        choices=["seasonal", "global", "recent"],
        default="seasonal",
        help="Bias correction strategy for train/test evaluation.",
    )
    parser.add_argument(
        "--bias-recent-days",
        type=int,
        help="Number of trailing train days to use when --bias-strategy=recent.",
    )
    parser.add_argument(
        "--openmeteo-mode",
        choices=["naive", "sources", "both"],
        default="naive",
        help=(
            "Historical Open-Meteo rows to collect: pooled naive baseline, "
            "individual model sources, or both."
        ),
    )
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--chunk-days", default=1, type=int)
    args = parser.parse_args(argv)
    if args.bias_strategy == "recent" and args.bias_recent_days is None:
        parser.error("--bias-recent-days is required when --bias-strategy=recent")

    result = run_historical_pipeline(
        cities=args.cities,
        start=args.start,
        end=args.end,
        test_start=args.test_start,
        validation_start=args.validation_start,
        out_dir=args.out_dir,
        cache_root=args.cache,
        alpha=args.alpha,
        bias_strategy=args.bias_strategy,
        bias_recent_days=args.bias_recent_days,
        openmeteo_mode=args.openmeteo_mode,
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

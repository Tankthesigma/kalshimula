"""CLI for collecting cache-backed weather backtest rows."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.collect import collect_backtest_rows, write_collection_csv


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="collect",
        description="Collect cache-backed weather backtest rows for a city/date range.",
    )
    parser.add_argument("--city", required=True)
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache", default=Path(".cache/weather"), type=Path)
    args = parser.parse_args(argv)

    result = collect_backtest_rows(
        city=args.city,
        start=args.start,
        end=args.end,
        cache_root=args.cache,
    )
    write_collection_csv(result, args.out)
    print(f"Wrote {len(result.rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

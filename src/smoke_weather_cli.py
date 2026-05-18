"""CLI wrapper around :mod:`src.smoke_weather`.

Usage::

    python -m src.smoke_weather_cli --date 2025-01-01
    python -m src.smoke_weather_cli --cities denver,nyc --date 2025-01-01 --out out.csv

Exit code is ``0`` iff every probed source returned without exception.
Failures (``ok=False`` rows) cause exit ``1`` so the script is usable from
shell or CI.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.config import load_stations
from src.smoke_weather import (
    SMOKE_COLUMNS,
    smoke_cities,
    smoke_results_to_dataframe,
)


def _parse_cities(value: str | None) -> list[str]:
    if value is None:
        return sorted(load_stations().keys())
    return [chunk.strip() for chunk in value.split(",") if chunk.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smoke_weather_cli",
        description="Probe weather sources for one or more cities and one date.",
    )
    parser.add_argument(
        "--cities",
        default=None,
        help="Comma-separated city slugs. Defaults to every slug in stations.yaml.",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Target date in YYYY-MM-DD form.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional CSV path to write smoke results.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    target = date.fromisoformat(args.date)
    cities = _parse_cities(args.cities)

    results = smoke_cities(cities, target)
    df = smoke_results_to_dataframe(results)

    header = " ".join(SMOKE_COLUMNS)
    print(header)
    for row in df.itertuples(index=False):
        print(
            f"{row.city} {row.target_date} {row.source} {row.ok} "
            f"{row.high_f if row.high_f is not None else '-'} "
            f"{row.error or '-'}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    all_ok = bool(df["ok"].all()) if not df.empty else True
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

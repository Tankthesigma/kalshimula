"""CLI for normalized NWS forecast guidance rows."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from src.models.nws_guidance import write_nws_guidance_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cities", help="Comma-separated city slugs. Defaults to all configured cities.")
    parser.add_argument(
        "--market-type",
        choices=["high", "low", "both"],
        default="high",
        help="Forecast market type to normalize. Defaults to high.",
    )
    parser.add_argument("--fetched-at", help="UTC ISO timestamp for deterministic test/backfill rows.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cities = _parse_cities(args.cities)
    fetched_at = _parse_fetched_at(args.fetched_at) if args.fetched_at else None
    rows = write_nws_guidance_rows(
        output_path=args.out,
        target=date.fromisoformat(args.date),
        cities=cities,
        market_types=_market_types(args.market_type),
        fetched_at=fetched_at,
    )
    print(f"Wrote {len(rows)} NWS guidance rows to {args.out}")
    return 0


def _parse_cities(value: str | None) -> list[str] | None:
    if value is None:
        return None
    cities = [city.strip().lower() for city in value.split(",") if city.strip()]
    return cities or None


def _market_types(value: str) -> list[str]:
    if value == "both":
        return ["high", "low"]
    return [value]


def _parse_fetched_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

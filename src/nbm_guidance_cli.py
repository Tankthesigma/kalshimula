"""Fetch public NBM text guidance into normalized guidance rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from src import predict
from src.models.nbm_guidance import NOMADS_BLEND_BASE_URL, write_nbm_guidance_rows
from src.models.station_rules import DEFAULT_STATION_RULES_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Target date, YYYY-MM-DD or shortcut.")
    parser.add_argument("--as-of", required=True, help="UTC ISO timestamp.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cities", help="Optional comma-separated city slugs.")
    parser.add_argument("--station-rules", type=Path, default=DEFAULT_STATION_RULES_PATH)
    parser.add_argument(
        "--base-url",
        default=NOMADS_BLEND_BASE_URL,
        help="NBM text product base URL. Use NOAA AWS S3 for historical archive probes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = write_nbm_guidance_rows(
        output_path=args.out,
        target=predict._parse_date(args.date),
        as_of_ts=args.as_of,
        station_rules_path=args.station_rules,
        cities=_split_csv(args.cities),
        base_url=args.base_url,
    )
    print(f"Wrote {len(rows)} NBM guidance rows: {args.out}")
    return 0


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip().lower() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

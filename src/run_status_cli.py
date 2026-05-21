"""CLI for inspecting a historical run directory while it is still running."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import load_stations


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_cities(value: str) -> list[str]:
    cities = [city.strip() for city in value.split(",") if city.strip()]
    if not cities:
        raise argparse.ArgumentTypeError("at least one city is required")
    return cities


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _expected_rows(*, cities: list[str], start, end, sources_per_day: int) -> int:
    days = (end - start).days + 1
    return len(cities) * days * sources_per_day


def _source_counts(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "source" not in rows.columns:
        return pd.DataFrame(columns=["source", "rows"])
    return (
        rows.groupby("source", sort=True)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", "source"], ascending=[False, True])
    )


def _city_counts(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "city" not in rows.columns:
        return pd.DataFrame(columns=["city", "rows"])
    return (
        rows.groupby("city", sort=True)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", "city"], ascending=[False, True])
    )


def _latest_row(rows: pd.DataFrame) -> dict | None:
    if rows.empty:
        return None
    return rows.iloc[-1].to_dict()


def _format_table(table: pd.DataFrame, *, limit: int = 12) -> str:
    if table.empty:
        return "  (none)"
    return table.head(limit).to_string(index=False)


def build_status(
    *,
    run_dir: Path,
    cities: list[str],
    start,
    end,
    sources_per_day: int,
) -> str:
    rows = _read_csv_if_exists(run_dir / "rows.csv")
    errors = _read_csv_if_exists(run_dir / "errors.csv")
    expected = _expected_rows(
        cities=cities, start=start, end=end, sources_per_day=sources_per_day
    )
    n_rows = len(rows)
    percent = (n_rows / expected * 100) if expected else 0.0
    latest = _latest_row(rows)

    lines = [
        f"Run: {run_dir}",
        f"Rows: {n_rows:,} / {expected:,} theoretical ({percent:.1f}%)",
        f"Errors: {len(errors):,}",
    ]
    if latest is not None:
        latest_bits = [
            str(latest.get("city", "")),
            str(latest.get("target_date", "")),
            str(latest.get("source", "")),
        ]
        lines.append(f"Latest row: {', '.join(bit for bit in latest_bits if bit)}")

    lines.extend(["", "Rows by city:", _format_table(_city_counts(rows))])
    lines.extend(["", "Rows by source:", _format_table(_source_counts(rows))])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_status",
        description="Show progress for a historical run directory.",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--cities",
        type=_parse_cities,
        default=",".join(load_stations().keys()),
    )
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument(
        "--sources-per-day",
        default=1,
        type=int,
        help="Theoretical rows per city/date. Use 8 for --openmeteo-mode both.",
    )
    args = parser.parse_args(argv)

    print(
        build_status(
            run_dir=args.run_dir,
            cities=args.cities,
            start=args.start,
            end=args.end,
            sources_per_day=args.sources_per_day,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

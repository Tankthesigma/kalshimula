"""CLI for weather-only ASOS/METAR nowcast features."""

from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from src.models.nowcast_features import write_nowcast_features
from src.models.station_rules import DEFAULT_STATION_RULES_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nowcast_features",
        description="Build weather-only point-in-time nowcast features.",
    )
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--as-of", required=True, help="UTC ISO timestamp")
    parser.add_argument("--decision-time-label", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--station-rules", type=Path, default=DEFAULT_STATION_RULES_PATH)
    parser.add_argument("--observations-csv", type=Path)
    parser.add_argument("--fetch-live", action="store_true")
    args = parser.parse_args(argv)

    observations = (
        pd.read_csv(args.observations_csv)
        if args.observations_csv is not None
        else None
    )
    result = write_nowcast_features(
        output_dir=args.out_dir,
        target_date=date.fromisoformat(args.target_date),
        as_of_ts=_parse_as_of(args.as_of),
        decision_time_label=args.decision_time_label,
        observations=observations,
        station_rules_path=args.station_rules,
        fetch_live=args.fetch_live,
        git_commit=_git_commit(),
    )
    print(
        f"Wrote nowcast features to {args.out_dir}: "
        f"{len(result.observations)} observations, {len(result.features)} feature rows"
    )
    return 0


def _parse_as_of(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())

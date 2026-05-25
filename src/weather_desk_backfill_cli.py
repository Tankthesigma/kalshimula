"""Backfill weather-desk scheduled packets across a date range.

This is a thin, mainline-safe wrapper around ``weather_desk_schedule_cli``. It
does not add market data. Its purpose is reproducible adjusted-vs-raw nowcast
testing over many settled days without hand-written shell loops.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from src import weather_desk_refresh_cli, weather_desk_schedule_cli
from src.config import load_stations
from src.models.station_rules import DEFAULT_STATION_RULES_PATH


@dataclass(frozen=True)
class BackfillRun:
    target_date: str
    out_dir: str
    exit_code: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run-dir", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--cities",
        default=",".join(load_stations().keys()),
        help="Comma-separated city slugs. Defaults to all configured cities.",
    )
    parser.add_argument("--decision-hours", default=weather_desk_schedule_cli.DEFAULT_DECISION_HOURS)
    parser.add_argument("--threshold-offsets", default="-2,0,2")
    parser.add_argument("--multi-source-mode", default="single")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--station-rules", type=Path, default=DEFAULT_STATION_RULES_PATH)
    parser.add_argument("--market-type", choices=["high", "low"], default="high")
    parser.add_argument("--observations-csv", type=Path)
    parser.add_argument("--observation-store", type=Path)
    parser.add_argument("--update-observation-store", action="store_true")
    parser.add_argument("--fetch-live", action="store_true")
    parser.add_argument("--include-nws-guidance", action="store_true")
    parser.add_argument("--no-require-gate", action="store_true")
    parser.add_argument("--model-version", default="mainline-nowcast-v1")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue later dates if a scheduled packet exits nonzero.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(
        weather_desk_refresh_cli._normalize_threshold_offsets(
            list(argv if argv is not None else sys.argv[1:])
        )
    )
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")

    runs: list[BackfillRun] = []
    for target in _date_range(start, end):
        date_out = args.out_dir / target.isoformat()
        schedule_args = [
            "--model-run-dir",
            str(args.model_run_dir),
            "--cities",
            args.cities,
            "--date",
            target.isoformat(),
            "--decision-hours",
            args.decision_hours,
            f"--threshold-offsets={args.threshold_offsets}",
            "--multi-source-mode",
            args.multi_source_mode,
            "--out-dir",
            str(date_out),
            "--station-rules",
            str(args.station_rules),
            "--market-type",
            args.market_type,
            "--model-version",
            args.model_version,
        ]
        if args.observations_csv is not None:
            schedule_args.extend(["--observations-csv", str(args.observations_csv)])
        if args.observation_store is not None:
            schedule_args.extend(["--observation-store", str(args.observation_store)])
        if args.update_observation_store:
            schedule_args.append("--update-observation-store")
        if args.fetch_live:
            schedule_args.append("--fetch-live")
        if args.include_nws_guidance:
            schedule_args.append("--include-nws-guidance")
        if args.no_require_gate:
            schedule_args.append("--no-require-gate")

        code = weather_desk_schedule_cli.main(schedule_args)
        runs.append(
            BackfillRun(
                target_date=target.isoformat(),
                out_dir=str(date_out),
                exit_code=code,
            )
        )
        if code != 0 and not args.continue_on_error:
            break

    exit_code = 0 if all(run.exit_code == 0 for run in runs) else 1
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "cities": args.cities,
        "market_type": args.market_type,
        "decision_hours": args.decision_hours,
        "continue_on_error": bool(args.continue_on_error),
        "packet_layout": "one schedule directory per date",
        "runs": [asdict(run) for run in runs],
        "exit_code": exit_code,
        "notes": [
            "Mainline weather-only backfill. No market prices, order books, private PnL labels, or trade instructions.",
            "Delegates each date to weather_desk_schedule_cli, preserving per-city local as-of cutoffs.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "weather_desk_backfill_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote weather desk backfill manifest: {manifest_path}")
    return exit_code


def _date_range(start: date, end: date) -> list[date]:
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

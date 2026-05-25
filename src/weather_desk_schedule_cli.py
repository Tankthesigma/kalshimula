"""Run weather-desk refreshes at local decision-time slices.

The ordinary weather desk refresh accepts one UTC ``--as-of`` timestamp. That
is correct for one packet, but a daily market schedule such as 04/07/10/13/15
local needs per-city UTC timestamps because the supported cities span multiple
time zones. This command writes one packet directory per city and local slice.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from src import predict, weather_desk_refresh_cli
from src.config import load_stations
from src.models.station_rules import DEFAULT_STATION_RULES_PATH, station_rule_by_key

DEFAULT_DECISION_HOURS = "04,07,10,13,15"
DEFAULT_DECISION_MINUTE = 20


@dataclass(frozen=True)
class ScheduledRun:
    city: str
    market_type: str
    decision_time_label: str
    local_hour: int
    local_minute: int
    timezone: str
    as_of_ts_utc: str
    out_dir: str
    exit_code: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run-dir", required=True, type=Path)
    parser.add_argument(
        "--cities",
        default=",".join(load_stations().keys()),
        help="Comma-separated city slugs. Defaults to all configured cities.",
    )
    parser.add_argument("--date", default="tomorrow")
    parser.add_argument("--decision-hours", default=DEFAULT_DECISION_HOURS)
    parser.add_argument(
        "--decision-minute",
        type=int,
        default=DEFAULT_DECISION_MINUTE,
        help=(
            "Local minute for each scheduled decision hour. Defaults to 20 so "
            "hourly ASOS observations near :54 clear the 10-minute availability lag."
        ),
    )
    parser.add_argument("--threshold-offsets", default="-2,0,2")
    parser.add_argument("--multi-source-mode", default="single")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="weather_desk")
    parser.add_argument("--station-rules", type=Path, default=DEFAULT_STATION_RULES_PATH)
    parser.add_argument("--market-type", choices=["high"], default="high")
    parser.add_argument("--observations-csv", type=Path)
    parser.add_argument("--observation-store", type=Path)
    parser.add_argument("--update-observation-store", action="store_true")
    parser.add_argument("--fetch-live", action="store_true")
    parser.add_argument("--include-nws-guidance", action="store_true")
    parser.add_argument("--include-nbm-guidance", action="store_true")
    parser.add_argument("--no-require-gate", action="store_true")
    parser.add_argument("--model-version", default="mainline-nowcast-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(
        weather_desk_refresh_cli._normalize_threshold_offsets(
            list(argv if argv is not None else sys.argv[1:])
        )
    )
    target = predict._parse_date(args.date)
    cities = _split_csv(args.cities)
    labels = _parse_decision_hours(args.decision_hours)
    decision_minute = _parse_decision_minute(args.decision_minute)
    runs: list[ScheduledRun] = []
    for label, hour in labels:
        for city in cities:
            rule = station_rule_by_key(
                city=city,
                market_type=args.market_type,
                path=args.station_rules,
            )
            local_dt = datetime.combine(
                target,
                time(hour=hour, minute=decision_minute),
                tzinfo=ZoneInfo(rule.timezone),
            )
            as_of_utc = local_dt.astimezone(UTC).isoformat()
            packet_dir = args.out_dir / f"{label}_local" / city
            refresh_args = [
                "--model-run-dir",
                str(args.model_run_dir),
                "--cities",
                city,
                "--date",
                args.date,
                f"--threshold-offsets={args.threshold_offsets}",
                "--multi-source-mode",
                args.multi_source_mode,
                "--out-dir",
                str(packet_dir),
                "--prefix",
                args.prefix,
                "--station-rules",
                str(args.station_rules),
                "--market-type",
                args.market_type,
                "--as-of",
                as_of_utc,
                "--decision-time-label",
                label,
                "--model-version",
                args.model_version,
            ]
            if args.observations_csv is not None:
                refresh_args.extend(["--observations-csv", str(args.observations_csv)])
            if args.observation_store is not None:
                refresh_args.extend(["--observation-store", str(args.observation_store)])
            if args.update_observation_store:
                refresh_args.append("--update-observation-store")
            if args.fetch_live:
                refresh_args.append("--fetch-live")
            if args.include_nws_guidance:
                refresh_args.append("--include-nws-guidance")
            if args.include_nbm_guidance:
                refresh_args.append("--include-nbm-guidance")
            if args.no_require_gate:
                refresh_args.append("--no-require-gate")
            code = weather_desk_refresh_cli.main(refresh_args)
            runs.append(
                ScheduledRun(
                    city=city,
                    market_type=args.market_type,
                    decision_time_label=label,
                    local_hour=hour,
                    local_minute=decision_minute,
                    timezone=rule.timezone,
                    as_of_ts_utc=as_of_utc,
                    out_dir=str(packet_dir),
                    exit_code=code,
                )
            )
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "target_date": target.isoformat(),
        "cities": cities,
        "market_type": args.market_type,
        "decision_time_labels": [label for label, _ in labels],
        "decision_minute": decision_minute,
        "include_nws_guidance": bool(args.include_nws_guidance),
        "include_nbm_guidance": bool(args.include_nbm_guidance),
        "packet_layout": "one directory per decision_time_label/city",
        "runs": [asdict(run) for run in runs],
        "exit_code": 0 if all(run.exit_code == 0 for run in runs) else 1,
        "notes": [
            "Mainline weather-only schedule. No market prices, order books, private PnL labels, or trade instructions.",
            "Each city uses its own local decision time converted to UTC for no-leak ASOS feature filtering.",
            "Default :20 local as-of keeps the 10-minute observation availability lag while reducing round-hour ASOS staleness.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "weather_desk_schedule_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote weather desk schedule manifest: {manifest_path}")
    return int(manifest["exit_code"])


def _split_csv(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _parse_decision_hours(value: str) -> list[tuple[str, int]]:
    labels: list[tuple[str, int]] = []
    for raw in value.split(","):
        text = raw.strip()
        if not text:
            continue
        hour = int(text)
        if not 0 <= hour <= 23:
            raise ValueError(f"decision hour must be 0-23: {text}")
        labels.append((f"{hour:02d}", hour))
    if not labels:
        raise ValueError("at least one decision hour is required")
    return labels


def _parse_decision_minute(value: int) -> int:
    if not 0 <= value <= 59:
        raise ValueError(f"decision minute must be 0-59: {value}")
    return value


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

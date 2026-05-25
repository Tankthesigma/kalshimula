"""One-command weather-only prediction plus weather desk packet refresh."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from src import predict, predict_batch_cli, weather_desk_cli
from src.config import load_stations
from src.models.nbm_guidance import NOMADS_BLEND_BASE_URL
from src.models.station_rules import DEFAULT_STATION_RULES_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run-dir", required=True, type=Path)
    parser.add_argument(
        "--cities",
        default=",".join(load_stations().keys()),
        help="Comma-separated city slugs. Defaults to all configured cities.",
    )
    parser.add_argument("--date", default="tomorrow")
    parser.add_argument("--threshold-offsets", default="-2,0,2")
    parser.add_argument("--multi-source-mode", default="single")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="weather_desk")
    parser.add_argument("--station-rules", type=Path, default=DEFAULT_STATION_RULES_PATH)
    parser.add_argument("--market-type", choices=["high"], default="high")
    parser.add_argument("--as-of", help="UTC ISO timestamp. Defaults to now.")
    parser.add_argument("--decision-time-label", default="morning")
    parser.add_argument("--observations-csv", type=Path)
    parser.add_argument("--observation-store", type=Path)
    parser.add_argument("--update-observation-store", action="store_true")
    parser.add_argument("--fetch-live", action="store_true")
    parser.add_argument("--include-nws-guidance", action="store_true")
    parser.add_argument("--include-nbm-guidance", action="store_true")
    parser.add_argument(
        "--nbm-base-url",
        default=NOMADS_BLEND_BASE_URL,
        help="NBM text product base URL passed through to weather_desk_cli.",
    )
    parser.add_argument("--no-require-gate", action="store_true")
    parser.add_argument("--model-version", default="mainline-nowcast-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(_normalize_threshold_offsets(list(argv or sys.argv[1:])))
    out_dir = args.out_dir
    prediction_path = out_dir / f"{args.prefix}_predictions.json"
    desk_dir = out_dir / args.prefix
    target_date = predict._parse_date(args.date).isoformat()
    as_of = _parse_as_of(args.as_of).isoformat()

    batch_args = [
        "--cities",
        args.cities,
        "--date",
        args.date,
        "--model-run-dir",
        str(args.model_run_dir),
        f"--threshold-offsets={args.threshold_offsets}",
        "--multi-source-mode",
        args.multi_source_mode,
        "--out",
        str(prediction_path),
    ]
    if not args.no_require_gate:
        batch_args.append("--require-gate")

    batch_code = predict_batch_cli.main(batch_args)
    desk_args = [
        "--predictions-json",
        str(prediction_path),
        "--target-date",
        target_date,
        "--as-of",
        as_of,
        "--decision-time-label",
        args.decision_time_label,
        "--station-rules",
        str(args.station_rules),
        "--cities",
        args.cities,
        "--market-type",
        args.market_type,
        "--model-version",
        args.model_version,
        "--out-dir",
        str(desk_dir),
    ]
    if args.observations_csv is not None:
        desk_args.extend(["--observations-csv", str(args.observations_csv)])
    if args.observation_store is not None:
        desk_args.extend(["--observation-store", str(args.observation_store)])
    if args.update_observation_store:
        desk_args.append("--update-observation-store")
    if args.fetch_live:
        desk_args.append("--fetch-live")
    if args.include_nws_guidance:
        desk_args.append("--include-nws-guidance")
    if args.include_nbm_guidance:
        desk_args.append("--include-nbm-guidance")
        desk_args.extend(["--nbm-base-url", args.nbm_base_url])

    desk_code = weather_desk_cli.main(desk_args) if prediction_path.exists() else 1
    manifest_path = out_dir / f"{args.prefix}_refresh_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "target_date": target_date,
        "as_of_ts_utc": as_of,
        "cities": args.cities,
        "market_type": args.market_type,
        "decision_time_label": args.decision_time_label,
        "include_nws_guidance": bool(args.include_nws_guidance),
        "include_nbm_guidance": bool(args.include_nbm_guidance),
        "nbm_base_url": args.nbm_base_url if args.include_nbm_guidance else None,
        "exit_code": 0 if batch_code == 0 and desk_code == 0 else 1,
        "steps": {
            "predict_batch": {"exit_code": batch_code},
            "weather_desk": {"exit_code": desk_code},
        },
        "artifacts": {
            "prediction_json": str(prediction_path),
            "weather_desk_dir": str(desk_dir),
            "weather_analyst_packet": str(
                desk_dir / "weather_analyst" / "weather_analyst_packet.md"
            ),
            "manifest": str(manifest_path),
        },
        "notes": [
            "Mainline weather-only refresh. No market prices, order books, private PnL labels, or trade instructions.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote prediction JSON: {prediction_path}")
    print(f"Wrote weather desk packet: {desk_dir}")
    print(f"Wrote weather desk refresh manifest: {manifest_path}")
    return int(manifest["exit_code"])


def _normalize_threshold_offsets(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if (
            arg == "--threshold-offsets"
            and index + 1 < len(argv)
            and argv[index + 1].startswith("-")
            and not argv[index + 1].startswith("--")
        ):
            normalized.append(f"--threshold-offsets={argv[index + 1]}")
            index += 2
            continue
        normalized.append(arg)
        index += 1
    return normalized


def _parse_as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


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

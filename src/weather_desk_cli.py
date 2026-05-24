"""Mainline weather-desk pipeline.

This command stitches together the weather-only nowcast stack:
features -> frozen prediction export -> weather-adjusted export -> report.
It does not fetch or use market data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from src.models.nowcast_adjustment import write_nowcast_adjusted_predictions
from src.models.nowcast_features import write_nowcast_features
from src.models.nowcast_predictions import write_nowcast_predictions
from src.models.nowcast_report import write_nowcast_report
from src.models.station_rules import DEFAULT_STATION_RULES_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--as-of", required=True, help="UTC ISO timestamp")
    parser.add_argument("--decision-time-label", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--station-rules", type=Path, default=DEFAULT_STATION_RULES_PATH)
    parser.add_argument("--observations-csv", type=Path)
    parser.add_argument("--observation-store", type=Path)
    parser.add_argument("--update-observation-store", action="store_true")
    parser.add_argument("--fetch-live", action="store_true")
    parser.add_argument("--model-version", default="mainline-nowcast-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir
    git_commit = _git_commit()
    observations = (
        pd.read_csv(args.observations_csv)
        if args.observations_csv is not None
        else None
    )
    feature_result = write_nowcast_features(
        output_dir=out_dir / "nowcast_features",
        target_date=date.fromisoformat(args.target_date),
        as_of_ts=_parse_as_of(args.as_of),
        decision_time_label=args.decision_time_label,
        observations=observations,
        observation_store_path=args.observation_store,
        update_observation_store=args.update_observation_store,
        station_rules_path=args.station_rules,
        fetch_live=args.fetch_live,
        git_commit=git_commit,
    )
    raw_result = write_nowcast_predictions(
        predictions_json_path=args.predictions_json,
        output_dir=out_dir / "predictions_nowcast_raw",
        decision_time_label=args.decision_time_label,
        nowcast_features_path=out_dir / "nowcast_features" / "nowcast_features.csv",
        station_rules_path=args.station_rules,
        as_of_ts_utc=args.as_of,
        model_version=args.model_version,
        git_commit=git_commit,
    )
    adjusted_result = write_nowcast_adjusted_predictions(
        predictions_path=out_dir / "predictions_nowcast_raw" / "predictions_nowcast.csv",
        features_path=out_dir / "nowcast_features" / "nowcast_features.csv",
        output_dir=out_dir / "predictions_nowcast_adjusted",
        git_commit=git_commit,
    )
    report_result = write_nowcast_report(
        predictions_path=out_dir / "predictions_nowcast_adjusted" / "predictions_nowcast.csv",
        output_dir=out_dir / "nowcast_report",
        git_commit=git_commit,
    )
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "target_date": args.target_date,
        "as_of_ts_utc": _parse_as_of(args.as_of).isoformat(),
        "decision_time_label": args.decision_time_label,
        "artifacts": {
            "nowcast_features": "nowcast_features/nowcast_features.csv",
            "predictions_nowcast_raw": "predictions_nowcast_raw/predictions_nowcast.csv",
            "predictions_nowcast_adjusted": (
                "predictions_nowcast_adjusted/predictions_nowcast.csv"
            ),
            "nowcast_report": "nowcast_report/nowcast_report.md",
        },
        "row_counts": {
            "observations": int(len(feature_result.observations)),
            "features": int(len(feature_result.features)),
            "raw_prediction_rows": int(len(raw_result.predictions)),
            "adjusted_prediction_rows": int(len(adjusted_result.predictions)),
            "report_rows": int(len(report_result.summary)),
        },
        "notes": [
            "Mainline weather-only pipeline. No market prices, order books, private PnL labels, or trade instructions.",
            "Bobby/private audit may consume predictions_nowcast_adjusted as a separate model mode.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weather_desk_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote weather desk packet to {out_dir}: "
        f"{len(adjusted_result.predictions)} adjusted prediction rows, "
        f"{len(report_result.summary)} report rows"
    )
    return 0


def _parse_as_of(value: str) -> datetime:
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
    raise SystemExit(main(sys.argv[1:]))

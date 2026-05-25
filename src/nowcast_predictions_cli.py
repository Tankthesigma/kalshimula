"""CLI for exporting frozen nowcast prediction rows for private audit joins."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.models.nowcast_predictions import write_nowcast_predictions
from src.models.station_rules import DEFAULT_STATION_RULES_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--decision-time-label", required=True)
    parser.add_argument("--nowcast-features", type=Path, default=None)
    parser.add_argument("--station-rules", type=Path, default=DEFAULT_STATION_RULES_PATH)
    parser.add_argument("--market-type", choices=["high"], default="high")
    parser.add_argument("--as-of", default=None, help="UTC timestamp fallback when no feature file is provided")
    parser.add_argument("--model-version", default="mainline-nowcast-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = write_nowcast_predictions(
        predictions_json_path=args.predictions_json,
        output_dir=args.out_dir,
        decision_time_label=args.decision_time_label,
        nowcast_features_path=args.nowcast_features,
        station_rules_path=args.station_rules,
        as_of_ts_utc=args.as_of,
        market_type=args.market_type,
        model_version=args.model_version,
        git_commit=_git_commit(),
    )
    print(
        f"Wrote {len(result.predictions)} nowcast prediction rows to "
        f"{args.out_dir / 'predictions_nowcast.csv'}"
    )
    return 0


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

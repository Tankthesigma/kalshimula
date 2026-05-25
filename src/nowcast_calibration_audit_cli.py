"""Run the market-free nowcast probability calibration audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from src.models.nowcast_calibration_audit import (
    build_calibration_audit,
    discover_prediction_files,
    fetch_ncei_actuals_for_predictions,
    read_actuals_csv,
    write_calibration_audit,
)
from src.models.station_rules import DEFAULT_STATION_RULES_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root",
        action="append",
        type=Path,
        default=[],
        help="Root to scan for predictions_nowcast_*/predictions_nowcast.csv.",
    )
    parser.add_argument(
        "--prediction-file",
        action="append",
        type=Path,
        default=[],
        help="Explicit predictions_nowcast.csv file. Can be repeated.",
    )
    parser.add_argument(
        "--actuals-csv",
        type=Path,
        help="Observed highs CSV with city,target_date,actual_high_f[,actual_source].",
    )
    parser.add_argument(
        "--fetch-ncei",
        action="store_true",
        help="Fetch missing official NCEI TMAX actuals for prediction city/date pairs.",
    )
    parser.add_argument("--station-rules", type=Path, default=DEFAULT_STATION_RULES_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/model_quality"))
    parser.add_argument("--n-buckets", type=int, default=10)
    parser.add_argument(
        "--min-statistical-n",
        type=int,
        default=30,
        help="Below this scored-group count, reports are labeled smoke only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prediction_files = sorted(
        set(args.prediction_file) | set(discover_prediction_files(args.prediction_root))
    )
    if not prediction_files:
        raise SystemExit("No prediction files found. Pass --prediction-root or --prediction-file.")

    actuals_frames = []
    if args.actuals_csv is not None:
        actuals_frames.append(read_actuals_csv(args.actuals_csv))
    if args.fetch_ncei:
        actuals_frames.append(
            fetch_ncei_actuals_for_predictions(
                prediction_files,
                station_rules_path=args.station_rules,
            )
        )
    if not actuals_frames:
        raise SystemExit("Provide --actuals-csv, --fetch-ncei, or both.")
    actuals = pd.concat(actuals_frames, ignore_index=True).drop_duplicates(
        subset=["city", "target_date"],
        keep="first",
    )

    result = build_calibration_audit(
        prediction_files,
        actuals=actuals,
        n_buckets=args.n_buckets,
        min_statistical_n=args.min_statistical_n,
        station_rules_path=args.station_rules,
        git_commit=_git_commit(),
    )
    write_calibration_audit(result, args.out_dir)
    print(json.dumps(result.manifest, indent=2, sort_keys=True))
    print(f"Wrote nowcast calibration audit: {args.out_dir}")
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
    raise SystemExit(main())

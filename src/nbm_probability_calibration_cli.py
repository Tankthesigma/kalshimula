"""Fit and apply market-free NBM probability calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.models.nbm_probability_calibration import (
    DEFAULT_TARGET_COVERAGE,
    apply_calibration_to_root,
    fit_temperature_scale,
    write_calibration_params,
)
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
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--apply-start", required=True)
    parser.add_argument("--apply-end", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--actuals-csv", type=Path)
    parser.add_argument("--fetch-ncei", action="store_true")
    parser.add_argument("--station-rules", type=Path, default=DEFAULT_STATION_RULES_PATH)
    parser.add_argument("--objective", choices=["nll", "coverage_then_nll"], default="nll")
    parser.add_argument("--target-coverage", type=float, default=DEFAULT_TARGET_COVERAGE)
    parser.add_argument("--min-temperature", type=float, default=1.0)
    parser.add_argument("--max-temperature", type=float, default=4.0)
    parser.add_argument("--temperature-step", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prediction_files = [
        path
        for path in discover_prediction_files([args.prediction_root])
        if path.parent.name == "predictions_nowcast_nbm"
    ]
    if not prediction_files:
        raise ValueError("no predictions_nowcast_nbm files found")
    if args.actuals_csv is not None:
        actuals = read_actuals_csv(args.actuals_csv)
    elif args.fetch_ncei:
        actuals = fetch_ncei_actuals_for_predictions(
            prediction_files,
            station_rules_path=args.station_rules,
        )
    else:
        raise ValueError("provide --actuals-csv or --fetch-ncei")

    base_audit = build_calibration_audit(
        prediction_files,
        actuals=actuals,
        station_rules_path=args.station_rules,
        git_commit=_git_commit(),
    )
    params, grid = fit_temperature_scale(
        base_audit.scored_rows,
        prediction_root=args.prediction_root,
        train_start=args.train_start,
        train_end=args.train_end,
        target_coverage=args.target_coverage,
        objective=args.objective,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
        step=args.temperature_step,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    params_path = args.out_dir / "nbm_calibration_params.json"
    write_calibration_params(params, params_path)
    grid.to_csv(args.out_dir / "nbm_calibration_temperature_grid.csv", index=False)
    written = apply_calibration_to_root(
        prediction_root=args.prediction_root,
        calibration_params_path=params_path,
        apply_start=args.apply_start,
        apply_end=args.apply_end,
        git_commit=_git_commit(),
    )
    calibrated_audit = build_calibration_audit(
        written,
        actuals=actuals,
        station_rules_path=args.station_rules,
        git_commit=_git_commit(),
    )
    write_calibration_audit(calibrated_audit, args.out_dir / "heldout_calibrated_audit")
    manifest = {
        "schema_version": "1.0",
        "git_commit": _git_commit(),
        "prediction_root": str(args.prediction_root),
        "train_window": {"start": args.train_start, "end": args.train_end},
        "apply_window": {"start": args.apply_start, "end": args.apply_end},
        "calibration_params": params.to_dict(),
        "written_files": [str(path) for path in written],
        "notes": [
            "Market-free NBM calibration. The apply window is disjoint from the fit window.",
            "No market prices, order books, private PnL labels, or trade instructions are used.",
        ],
    }
    (args.out_dir / "nbm_probability_calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote NBM calibration params: {params_path}")
    print(f"Wrote {len(written)} calibrated heldout packet files")
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

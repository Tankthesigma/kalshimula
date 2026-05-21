"""One-command daily model refresh for prediction and readiness review output."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src import (
    model_gate_cli,
    model_policy_report_cli,
    predict_batch_cli,
    prediction_review_cli,
)
from src.config import load_stations


@dataclass(frozen=True)
class RefreshPaths:
    json_out: Path
    review_out: Path
    gate_out: Path
    policy_out: Path
    manifest_out: Path


def build_refresh_paths(
    *,
    model_run_dir: Path,
    out_dir: Path | None,
    prefix: str,
) -> RefreshPaths:
    """Return stable output paths for the daily refresh artifacts."""
    directory = out_dir or model_run_dir
    return RefreshPaths(
        json_out=directory / f"{prefix}.json",
        review_out=directory / f"{prefix}.txt",
        gate_out=directory / f"{prefix}_gate.txt",
        policy_out=directory / f"{prefix}_model_policy.txt",
        manifest_out=directory / f"{prefix}_manifest.json",
    )


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


def _write_gate_report(*, run_dir: Path, out_path: Path) -> int:
    try:
        checks = model_gate_cli.build_gate_checks(
            run_dir=run_dir,
            max_test_mae=model_gate_cli.DEFAULT_MAX_TEST_MAE,
            min_interval_coverage=model_gate_cli.DEFAULT_MIN_INTERVAL_COVERAGE,
            max_interval_width=model_gate_cli.DEFAULT_MAX_INTERVAL_WIDTH,
            max_recalibrated_brier=model_gate_cli.DEFAULT_MAX_RECALIBRATED_BRIER,
            max_recalibrated_ece=model_gate_cli.DEFAULT_MAX_RECALIBRATED_ECE,
            min_brier_improvement=model_gate_cli.DEFAULT_MIN_BRIER_IMPROVEMENT,
            min_ece_improvement=model_gate_cli.DEFAULT_MIN_ECE_IMPROVEMENT,
            expected_source=model_gate_cli.DEFAULT_EXPECTED_SOURCE,
        )
        report = model_gate_cli.render_gate_report(checks)
        code = 0 if all(check.passed for check in checks) else 1
    except ValueError as error:
        report = f"Model readiness gate:\n  FAIL artifact_error: {error}\nOutcome: FAIL"
        code = 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    return code


def _write_policy_report(*, run_dir: Path, out_path: Path) -> None:
    report = model_policy_report_cli.build_model_policy_report(run_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")


def _write_manifest(
    *,
    out_path: Path,
    model_run_dir: Path,
    cities: str,
    target_date: str,
    threshold_offsets: str,
    require_gate: bool,
    paths: RefreshPaths,
    batch_code: int,
    review_code: int,
    gate_code: int,
) -> int:
    exit_code = next(
        (code for code in (batch_code, review_code, gate_code) if code != 0),
        0,
    )
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_run_dir": str(model_run_dir),
        "cities": cities,
        "target_date": target_date,
        "threshold_offsets": threshold_offsets,
        "require_gate": require_gate,
        "exit_code": exit_code,
        "steps": {
            "batch_predictions": {"exit_code": batch_code},
            "prediction_review": {"exit_code": review_code},
            "model_gate_report": {"exit_code": gate_code},
            "model_policy_report": {"exit_code": 0},
        },
        "artifacts": {
            "prediction_json": str(paths.json_out),
            "prediction_review": str(paths.review_out),
            "model_gate_report": str(paths.gate_out),
            "model_policy_report": str(paths.policy_out),
            "manifest": str(paths.manifest_out),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="daily_model_refresh",
        description="Write gated batch prediction JSON and a text review report.",
    )
    parser.add_argument("--model-run-dir", required=True, type=Path)
    parser.add_argument(
        "--cities",
        default=",".join(load_stations().keys()),
        help="Comma-separated city slugs. Defaults to all configured cities.",
    )
    parser.add_argument("--date", default="tomorrow")
    parser.add_argument("--threshold-offsets", default="-2,0,2")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--prefix", default="latest_predictions")
    parser.add_argument(
        "--no-require-gate",
        action="store_true",
        help="Diagnostic mode: do not require model readiness gate before predictions.",
    )
    args = parser.parse_args(_normalize_threshold_offsets(list(argv or sys.argv[1:])))

    paths = build_refresh_paths(
        model_run_dir=args.model_run_dir,
        out_dir=args.out_dir,
        prefix=args.prefix,
    )
    batch_args = [
        "--cities",
        args.cities,
        "--date",
        args.date,
        "--model-run-dir",
        str(args.model_run_dir),
        f"--threshold-offsets={args.threshold_offsets}",
        "--out",
        str(paths.json_out),
    ]
    if not args.no_require_gate:
        batch_args.append("--require-gate")

    batch_code = predict_batch_cli.main(batch_args)
    review_code = prediction_review_cli.main(
        ["--input", str(paths.json_out), "--out", str(paths.review_out)]
    )
    gate_code = _write_gate_report(run_dir=args.model_run_dir, out_path=paths.gate_out)
    _write_policy_report(run_dir=args.model_run_dir, out_path=paths.policy_out)
    print(f"Wrote prediction JSON: {paths.json_out}")
    print(f"Wrote prediction review: {paths.review_out}")
    print(f"Wrote model gate report: {paths.gate_out}")
    print(f"Wrote model policy report: {paths.policy_out}")
    exit_code = _write_manifest(
        out_path=paths.manifest_out,
        model_run_dir=args.model_run_dir,
        cities=args.cities,
        target_date=args.date,
        threshold_offsets=args.threshold_offsets,
        require_gate=not args.no_require_gate,
        paths=paths,
        batch_code=batch_code,
        review_code=review_code,
        gate_code=gate_code,
    )
    print(f"Wrote packet manifest: {paths.manifest_out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

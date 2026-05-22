"""One-command daily model refresh for prediction and readiness review output."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src import (
    daily_packet_check_cli,
    forward_test_gate_cli,
    forward_test_settle_cli,
    model_gate_cli,
    model_policy_report_cli,
    predict_batch_cli,
    prediction_review_cli,
)
from src.config import load_stations

DEFAULT_MAX_PACKET_AGE_HOURS = 24.0


@dataclass(frozen=True)
class RefreshPaths:
    json_out: Path
    review_out: Path
    gate_out: Path
    gate_json_out: Path
    policy_out: Path
    manifest_out: Path
    check_out: Path


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
        gate_json_out=directory / f"{prefix}_gate.json",
        policy_out=directory / f"{prefix}_model_policy.txt",
        manifest_out=directory / f"{prefix}_manifest.json",
        check_out=directory / f"{prefix}_check.json",
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


def _write_gate_report(*, run_dir: Path, out_path: Path, json_out_path: Path | None = None) -> int:
    try:
        checks = model_gate_cli.build_gate_checks(
            run_dir=run_dir,
            min_rows=model_gate_cli.DEFAULT_MIN_ROWS,
            min_cities=model_gate_cli.DEFAULT_MIN_CITIES,
            min_sources=model_gate_cli.DEFAULT_MIN_SOURCES,
            min_target_dates=model_gate_cli.DEFAULT_MIN_TARGET_DATES,
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
        payload = model_gate_cli.build_gate_check_payload(run_dir, checks)
        code = 0 if all(check.passed for check in checks) else 1
    except ValueError as error:
        report = f"Model readiness gate:\n  FAIL artifact_error: {error}\nOutcome: FAIL"
        payload = model_gate_cli.build_artifact_error_payload(run_dir, error)
        code = 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    if json_out_path is not None:
        model_gate_cli.write_gate_payload(payload, json_out_path)
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
    require_selected_source_applied: bool,
    max_packet_age_hours: float | None,
    paths: RefreshPaths,
    batch_code: int,
    review_code: int,
    gate_code: int,
    check_code: int | None = None,
    settlement_code: int | None = None,
    forward_test_gate_code: int | None = None,
    settlement_artifacts: dict[str, str] | None = None,
    forward_test_gate_artifacts: dict[str, str] | None = None,
) -> int:
    exit_code = next(
        (
            code
            for code in (
                batch_code,
                review_code,
                gate_code,
                check_code,
                settlement_code,
                forward_test_gate_code,
            )
            if code not in {None, 0}
        ),
        0,
    )
    steps = {
        "batch_predictions": {"exit_code": batch_code},
        "prediction_review": {"exit_code": review_code},
        "model_gate_report": {"exit_code": gate_code},
        "model_gate_json": {"exit_code": gate_code},
        "model_policy_report": {"exit_code": 0},
    }
    if check_code is not None:
        steps["packet_check"] = {"exit_code": check_code}
    if settlement_code is not None:
        steps["forward_test_settlement"] = {"exit_code": settlement_code}
    if forward_test_gate_code is not None:
        steps["forward_test_gate"] = {"exit_code": forward_test_gate_code}

    artifacts = {
        "prediction_json": str(paths.json_out),
        "prediction_review": str(paths.review_out),
        "model_gate_report": str(paths.gate_out),
        "model_gate_json": str(paths.gate_json_out),
        "model_policy_report": str(paths.policy_out),
        "manifest": str(paths.manifest_out),
        "packet_check": str(paths.check_out),
    }
    if settlement_artifacts:
        artifacts.update(settlement_artifacts)
    if forward_test_gate_artifacts:
        artifacts.update(forward_test_gate_artifacts)

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_run_dir": str(model_run_dir),
        "cities": cities,
        "target_date": target_date,
        "threshold_offsets": threshold_offsets,
        "require_gate": require_gate,
        "require_selected_source_applied": require_selected_source_applied,
        "max_packet_age_hours": max_packet_age_hours,
        "exit_code": exit_code,
        "steps": steps,
        "artifacts": artifacts,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return exit_code


def _packet_target_date(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload.get("target_date")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"prediction packet missing target_date: {path}")
    return target


def _settlement_artifacts(
    *,
    out_dir: Path,
    target_date: str,
    settlement_out: Path | None,
    settlement_history: Path | None,
    settlement_report_out: Path | None,
    settlement_no_report: bool,
) -> dict[str, str]:
    history = settlement_history or out_dir / "history.csv"
    artifacts = {
        "settlement_json": str(
            settlement_out or out_dir / f"{target_date}_settlement.json"
        ),
        "settlement_history": str(history),
    }
    if not settlement_no_report:
        artifacts["settlement_report"] = str(
            settlement_report_out or history.with_name("report.json")
        )
    return artifacts


def _run_settlement(
    *,
    packet_path: Path,
    target_date: str,
    actuals_csv: Path | None,
    out_dir: Path,
    settlement_out: Path | None,
    settlement_history: Path | None,
    settlement_report_out: Path | None,
    settlement_no_report: bool,
) -> int:
    args = [
        "--packet",
        str(packet_path),
        "--target-date",
        target_date,
        "--out-dir",
        str(out_dir),
    ]
    if actuals_csv is not None:
        args.extend(["--actuals-csv", str(actuals_csv)])
    if settlement_out is not None:
        args.extend(["--out", str(settlement_out)])
    if settlement_history is not None:
        args.extend(["--history", str(settlement_history)])
    if settlement_report_out is not None:
        args.extend(["--report-out", str(settlement_report_out)])
    if settlement_no_report:
        args.append("--no-report")
    return forward_test_settle_cli.main(args)


def _run_forward_test_gate(
    *,
    report_path: Path,
    out_path: Path,
    min_target_dates: int,
    min_predictions: int,
    min_threshold_events: int,
    max_mae: float,
    max_abs_bias: float,
    min_interval_coverage: float,
    max_threshold_brier: float,
    max_threshold_ece: float,
) -> int:
    return forward_test_gate_cli.main(
        [
            "--report",
            str(report_path),
            "--out",
            str(out_path),
            "--min-target-dates",
            str(min_target_dates),
            "--min-predictions",
            str(min_predictions),
            "--min-threshold-events",
            str(min_threshold_events),
            "--max-mae",
            str(max_mae),
            "--max-abs-bias",
            str(max_abs_bias),
            "--min-interval-coverage",
            str(min_interval_coverage),
            "--max-threshold-brier",
            str(max_threshold_brier),
            "--max-threshold-ece",
            str(max_threshold_ece),
        ]
    )


def _run_packet_check(paths: RefreshPaths) -> int:
    return daily_packet_check_cli.main(
        [
            "--manifest",
            str(paths.manifest_out),
            "--json",
            "--out",
            str(paths.check_out),
        ]
    )


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
        "--max-packet-age-hours",
        default=DEFAULT_MAX_PACKET_AGE_HOURS,
        type=float,
        help="Maximum age accepted by daily_packet_check when this packet is verified.",
    )
    parser.add_argument(
        "--no-max-packet-age",
        action="store_true",
        help="Diagnostic mode: omit packet freshness expiry from the manifest.",
    )
    parser.add_argument(
        "--no-require-gate",
        action="store_true",
        help="Diagnostic mode: do not require model readiness gate before predictions.",
    )
    parser.add_argument(
        "--allow-source-fallback",
        action="store_true",
        help=(
            "Diagnostic mode: allow a prediction packet even if the selected "
            "source policy could not be applied for one or more cities."
        ),
    )
    parser.add_argument(
        "--settle",
        action="store_true",
        help="After packet check, settle the packet and write forward-test artifacts.",
    )
    parser.add_argument(
        "--settle-target-date",
        help="Target date to settle. Defaults to the packet target_date.",
    )
    parser.add_argument(
        "--settle-actuals-csv",
        type=Path,
        help="Optional offline actuals CSV for settlement.",
    )
    parser.add_argument(
        "--settlement-out-dir",
        type=Path,
        help="Directory for settlement JSON/history/report. Defaults to <out>/forward_test.",
    )
    parser.add_argument("--settlement-out", type=Path)
    parser.add_argument("--settlement-history", type=Path)
    parser.add_argument("--settlement-report-out", type=Path)
    parser.add_argument(
        "--settlement-no-report",
        action="store_true",
        help="Pass --no-report through to forward_test_settle.",
    )
    parser.add_argument(
        "--forward-test-gate",
        action="store_true",
        help="Run forward_test_gate after packet check and optional settlement.",
    )
    parser.add_argument(
        "--forward-test-gate-report",
        type=Path,
        help="Forward-test report JSON to gate. Defaults to the settlement report path.",
    )
    parser.add_argument(
        "--forward-test-gate-out",
        type=Path,
        help="Forward-test gate JSON output path.",
    )
    parser.add_argument(
        "--forward-test-min-target-dates",
        default=forward_test_gate_cli.DEFAULT_MIN_TARGET_DATES,
        type=int,
    )
    parser.add_argument(
        "--forward-test-min-predictions",
        default=forward_test_gate_cli.DEFAULT_MIN_PREDICTIONS,
        type=int,
    )
    parser.add_argument(
        "--forward-test-min-threshold-events",
        default=forward_test_gate_cli.DEFAULT_MIN_THRESHOLD_EVENTS,
        type=int,
    )
    parser.add_argument(
        "--forward-test-max-mae",
        default=forward_test_gate_cli.DEFAULT_MAX_MAE,
        type=float,
    )
    parser.add_argument(
        "--forward-test-max-abs-bias",
        default=forward_test_gate_cli.DEFAULT_MAX_ABS_BIAS,
        type=float,
    )
    parser.add_argument(
        "--forward-test-min-interval-coverage",
        default=forward_test_gate_cli.DEFAULT_MIN_INTERVAL_COVERAGE,
        type=float,
    )
    parser.add_argument(
        "--forward-test-max-threshold-brier",
        default=forward_test_gate_cli.DEFAULT_MAX_THRESHOLD_BRIER,
        type=float,
    )
    parser.add_argument(
        "--forward-test-max-threshold-ece",
        default=forward_test_gate_cli.DEFAULT_MAX_THRESHOLD_ECE,
        type=float,
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
    gate_code = _write_gate_report(
        run_dir=args.model_run_dir,
        out_path=paths.gate_out,
        json_out_path=paths.gate_json_out,
    )
    _write_policy_report(run_dir=args.model_run_dir, out_path=paths.policy_out)
    print(f"Wrote prediction JSON: {paths.json_out}")
    print(f"Wrote prediction review: {paths.review_out}")
    print(f"Wrote model gate report: {paths.gate_out}")
    print(f"Wrote model gate JSON: {paths.gate_json_out}")
    print(f"Wrote model policy report: {paths.policy_out}")
    _write_manifest(
        out_path=paths.manifest_out,
        model_run_dir=args.model_run_dir,
        cities=args.cities,
        target_date=args.date,
        threshold_offsets=args.threshold_offsets,
        require_gate=not args.no_require_gate,
        require_selected_source_applied=not args.allow_source_fallback,
        max_packet_age_hours=(
            None if args.no_max_packet_age else args.max_packet_age_hours
        ),
        paths=paths,
        batch_code=batch_code,
        review_code=review_code,
        gate_code=gate_code,
    )
    print(f"Wrote packet manifest: {paths.manifest_out}")
    check_code = _run_packet_check(paths)
    print(f"Wrote packet check: {paths.check_out}")
    settlement_code: int | None = None
    settlement_artifacts: dict[str, str] | None = None
    forward_test_gate_code: int | None = None
    forward_test_gate_artifacts: dict[str, str] | None = None
    settlement_out_dir = args.settlement_out_dir or (
        (args.out_dir or args.model_run_dir) / "forward_test"
    )
    if args.settle:
        try:
            settlement_target_date = args.settle_target_date or _packet_target_date(
                paths.json_out
            )
            settlement_artifacts = _settlement_artifacts(
                out_dir=settlement_out_dir,
                target_date=settlement_target_date,
                settlement_out=args.settlement_out,
                settlement_history=args.settlement_history,
                settlement_report_out=args.settlement_report_out,
                settlement_no_report=args.settlement_no_report,
            )
            settlement_code = _run_settlement(
                packet_path=paths.json_out,
                target_date=settlement_target_date,
                actuals_csv=args.settle_actuals_csv,
                out_dir=settlement_out_dir,
                settlement_out=args.settlement_out,
                settlement_history=args.settlement_history,
                settlement_report_out=args.settlement_report_out,
                settlement_no_report=args.settlement_no_report,
            )
            print("Ran forward-test settlement")
        except ValueError as error:
            settlement_code = 1
            print(f"Forward-test settlement failed: {error}")

    if args.forward_test_gate:
        gate_report_path = (
            args.forward_test_gate_report
            or args.settlement_report_out
            or settlement_out_dir / "report.json"
        )
        gate_out_path = (
            args.forward_test_gate_out or settlement_out_dir / "forward_test_gate.json"
        )
        forward_test_gate_artifacts = {"forward_test_gate_json": str(gate_out_path)}
        forward_test_gate_code = _run_forward_test_gate(
            report_path=gate_report_path,
            out_path=gate_out_path,
            min_target_dates=args.forward_test_min_target_dates,
            min_predictions=args.forward_test_min_predictions,
            min_threshold_events=args.forward_test_min_threshold_events,
            max_mae=args.forward_test_max_mae,
            max_abs_bias=args.forward_test_max_abs_bias,
            min_interval_coverage=args.forward_test_min_interval_coverage,
            max_threshold_brier=args.forward_test_max_threshold_brier,
            max_threshold_ece=args.forward_test_max_threshold_ece,
        )
        print(f"Wrote forward-test gate JSON: {gate_out_path}")

    final_code = _write_manifest(
        out_path=paths.manifest_out,
        model_run_dir=args.model_run_dir,
        cities=args.cities,
        target_date=args.date,
        threshold_offsets=args.threshold_offsets,
        require_gate=not args.no_require_gate,
        require_selected_source_applied=not args.allow_source_fallback,
        max_packet_age_hours=(
            None if args.no_max_packet_age else args.max_packet_age_hours
        ),
        paths=paths,
        batch_code=batch_code,
        review_code=review_code,
        gate_code=gate_code,
        check_code=check_code,
        settlement_code=settlement_code,
        forward_test_gate_code=forward_test_gate_code,
        settlement_artifacts=settlement_artifacts,
        forward_test_gate_artifacts=forward_test_gate_artifacts,
    )
    final_check_code = _run_packet_check(paths)
    print(f"Wrote final packet check: {paths.check_out}")
    if final_check_code != check_code:
        final_code = _write_manifest(
            out_path=paths.manifest_out,
            model_run_dir=args.model_run_dir,
            cities=args.cities,
            target_date=args.date,
            threshold_offsets=args.threshold_offsets,
            require_gate=not args.no_require_gate,
            require_selected_source_applied=not args.allow_source_fallback,
            max_packet_age_hours=(
                None if args.no_max_packet_age else args.max_packet_age_hours
            ),
            paths=paths,
            batch_code=batch_code,
            review_code=review_code,
            gate_code=gate_code,
            check_code=final_check_code,
            settlement_code=settlement_code,
            forward_test_gate_code=forward_test_gate_code,
            settlement_artifacts=settlement_artifacts,
            forward_test_gate_artifacts=forward_test_gate_artifacts,
        )
        _run_packet_check(paths)
    return final_code


if __name__ == "__main__":
    raise SystemExit(main())

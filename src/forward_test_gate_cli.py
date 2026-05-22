"""Gate accumulated forward-test metrics against live-monitor thresholds."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_REPORT = Path("outputs") / "forward_test" / "report.json"
DEFAULT_MIN_TARGET_DATES = 1
DEFAULT_MIN_PREDICTIONS = 10
DEFAULT_MIN_THRESHOLD_EVENTS = 30
DEFAULT_MAX_MAE = 1.50
DEFAULT_MAX_ABS_BIAS = 0.75
DEFAULT_MIN_INTERVAL_COVERAGE = 0.75
DEFAULT_MAX_THRESHOLD_BRIER = 0.12
DEFAULT_MAX_THRESHOLD_ECE = 0.20


@dataclass(frozen=True)
class ForwardTestCheck:
    name: str
    value: float | str
    threshold: float | str
    passed: bool
    detail: str


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"missing forward-test report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("forward-test report must be a JSON object")
    return payload


def _metric(summary: dict[str, Any], name: str) -> float | None:
    value = summary.get(name)
    if value is None:
        return None
    return float(value)


def _required_metric_check(
    *,
    name: str,
    value: float | None,
    threshold: float,
    passed: bool,
    detail: str,
) -> ForwardTestCheck:
    return ForwardTestCheck(
        name=name,
        value="missing" if value is None else value,
        threshold=threshold,
        passed=value is not None and passed,
        detail=detail,
    )


def build_forward_test_gate_checks(
    *,
    report_path: Path,
    min_target_dates: int,
    min_predictions: int,
    min_threshold_events: int,
    max_mae: float,
    max_abs_bias: float,
    min_interval_coverage: float,
    max_threshold_brier: float,
    max_threshold_ece: float,
) -> list[ForwardTestCheck]:
    """Build live forward-test monitoring checks from a report JSON artifact."""
    report = _load_report(report_path)
    summary = report.get("summary") or {}
    checks: list[ForwardTestCheck] = [
        ForwardTestCheck(
            name="schema_version",
            value=str(report.get("schema_version", "missing")),
            threshold="1.0",
            passed=report.get("schema_version") == "1.0",
            detail="forward-test report schema version",
        )
    ]

    n_target_dates = int(summary.get("n_target_dates", 0) or 0)
    n_predictions = int(summary.get("n_predictions", 0) or 0)
    n_threshold_events = int(summary.get("n_threshold_events", 0) or 0)
    checks.extend(
        [
            ForwardTestCheck(
                name="target_date_count",
                value=float(n_target_dates),
                threshold=float(min_target_dates),
                passed=n_target_dates >= min_target_dates,
                detail="unique settled target dates in forward-test report",
            ),
            ForwardTestCheck(
                name="prediction_count",
                value=float(n_predictions),
                threshold=float(min_predictions),
                passed=n_predictions >= min_predictions,
                detail="unique city/date predictions in forward-test report",
            ),
            ForwardTestCheck(
                name="threshold_event_count",
                value=float(n_threshold_events),
                threshold=float(min_threshold_events),
                passed=n_threshold_events >= min_threshold_events,
                detail="settled threshold events in forward-test report",
            ),
        ]
    )

    mae = _metric(summary, "mae_corrected_f")
    bias = _metric(summary, "bias_corrected_f")
    coverage = _metric(summary, "interval_coverage")
    brier = _metric(summary, "threshold_brier_score")
    ece = _metric(summary, "threshold_ece")
    checks.extend(
        [
            _required_metric_check(
                name="mae_corrected_f",
                value=mae,
                threshold=max_mae,
                passed=mae is not None and mae <= max_mae,
                detail="accumulated corrected point-forecast MAE",
            ),
            _required_metric_check(
                name="abs_bias_corrected_f",
                value=None if bias is None else abs(bias),
                threshold=max_abs_bias,
                passed=bias is not None and abs(bias) <= max_abs_bias,
                detail="absolute accumulated corrected forecast bias",
            ),
            _required_metric_check(
                name="interval_coverage",
                value=coverage,
                threshold=min_interval_coverage,
                passed=coverage is not None and coverage >= min_interval_coverage,
                detail="share of actual highs inside forecast interval",
            ),
            _required_metric_check(
                name="threshold_brier_score",
                value=brier,
                threshold=max_threshold_brier,
                passed=brier is not None and brier <= max_threshold_brier,
                detail="accumulated threshold probability Brier score",
            ),
            _required_metric_check(
                name="threshold_ece",
                value=ece,
                threshold=max_threshold_ece,
                passed=ece is not None and ece <= max_threshold_ece,
                detail="accumulated threshold probability calibration error",
            ),
        ]
    )
    return checks


def gate_check_payload(check: ForwardTestCheck) -> dict[str, Any]:
    return {
        "name": check.name,
        "value": check.value,
        "threshold": check.threshold,
        "passed": check.passed,
        "detail": check.detail,
    }


def summarize_gate_checks(checks: list[ForwardTestCheck]) -> dict[str, Any]:
    failed = [check for check in checks if not check.passed]
    return {
        "total_checks": len(checks),
        "passed_checks": len(checks) - len(failed),
        "failed_checks": len(failed),
        "failed_check_names": [check.name for check in failed],
    }


def build_gate_payload(report_path: Path, checks: list[ForwardTestCheck]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "report_path": str(report_path),
        "passed": all(check.passed for check in checks),
        "summary": summarize_gate_checks(checks),
        "checks": [gate_check_payload(check) for check in checks],
    }


def build_artifact_error_payload(report_path: Path, error: ValueError) -> dict[str, Any]:
    check = ForwardTestCheck(
        name="artifact_error",
        value=str(error),
        threshold="forward-test report readable",
        passed=False,
        detail=str(error),
    )
    return build_gate_payload(report_path, [check])


def _format_value(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.4f}"


def render_gate_report(checks: list[ForwardTestCheck]) -> str:
    lines = ["Forward test gate:"]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(
            f"  {status} {check.name}: value={_format_value(check.value)} "
            f"threshold={_format_value(check.threshold)} ({check.detail})"
        )
    outcome = "PASS" if all(check.passed for check in checks) else "FAIL"
    lines.append(f"Outcome: {outcome}")
    return "\n".join(lines)


def write_gate_payload(payload: dict[str, Any], output_path: Path | None) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True)
    if output_path is None:
        print(content)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forward_test_gate",
        description="Fail if accumulated forward-test metrics exceed live thresholds.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-target-dates", default=DEFAULT_MIN_TARGET_DATES, type=int)
    parser.add_argument("--min-predictions", default=DEFAULT_MIN_PREDICTIONS, type=int)
    parser.add_argument(
        "--min-threshold-events",
        default=DEFAULT_MIN_THRESHOLD_EVENTS,
        type=int,
    )
    parser.add_argument("--max-mae", default=DEFAULT_MAX_MAE, type=float)
    parser.add_argument("--max-abs-bias", default=DEFAULT_MAX_ABS_BIAS, type=float)
    parser.add_argument(
        "--min-interval-coverage",
        default=DEFAULT_MIN_INTERVAL_COVERAGE,
        type=float,
    )
    parser.add_argument(
        "--max-threshold-brier",
        default=DEFAULT_MAX_THRESHOLD_BRIER,
        type=float,
    )
    parser.add_argument(
        "--max-threshold-ece",
        default=DEFAULT_MAX_THRESHOLD_ECE,
        type=float,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON gate result instead of text.",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    try:
        checks = build_forward_test_gate_checks(
            report_path=args.report,
            min_target_dates=args.min_target_dates,
            min_predictions=args.min_predictions,
            min_threshold_events=args.min_threshold_events,
            max_mae=args.max_mae,
            max_abs_bias=args.max_abs_bias,
            min_interval_coverage=args.min_interval_coverage,
            max_threshold_brier=args.max_threshold_brier,
            max_threshold_ece=args.max_threshold_ece,
        )
    except ValueError as error:
        payload = build_artifact_error_payload(args.report, error)
        if args.json or args.out is not None:
            write_gate_payload(payload, args.out)
        else:
            print(f"Forward test gate:\n  FAIL artifact_error: {error}\nOutcome: FAIL")
        return 1

    if args.json or args.out is not None:
        write_gate_payload(build_gate_payload(args.report, checks), args.out)
    else:
        print(render_gate_report(checks))
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

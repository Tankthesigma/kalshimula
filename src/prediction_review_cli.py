"""Render gated batch prediction JSON as a compact review report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt_float(value: Any, *, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _threshold_label(threshold: int) -> str:
    return f"P>={threshold}F"


def _thresholds(predictions: list[dict]) -> list[int]:
    values = set()
    for prediction in predictions:
        for row in prediction.get("threshold_probabilities") or []:
            threshold = row.get("threshold_f")
            if threshold is not None:
                values.add(int(threshold))
    return sorted(values)


def _threshold_map(prediction: dict) -> dict[int, dict]:
    rows = prediction.get("threshold_probabilities") or []
    return {int(row["threshold_f"]): row for row in rows if row.get("threshold_f") is not None}


def _prediction_row(prediction: dict, thresholds: list[int]) -> dict[str, str]:
    forecast = prediction.get("forecast") or {}
    calibration = prediction.get("calibration") or {}
    interval_lower = calibration.get("interval_lower_f")
    interval_upper = calibration.get("interval_upper_f")
    interval = (
        f"[{_fmt_float(interval_lower)}, {_fmt_float(interval_upper)}]"
        if interval_lower is not None and interval_upper is not None
        else "n/a"
    )
    threshold_rows = _threshold_map(prediction)
    row = {
        "city": str(prediction.get("city", "n/a")),
        "source": str(prediction.get("selected_source") or calibration.get("source") or "n/a"),
        "raw_f": _fmt_float(forecast.get("point_f")),
        "corrected_f": _fmt_float(calibration.get("corrected_point_f")),
        "interval_f": interval,
        "members": str(forecast.get("n_members", "n/a")),
    }
    for threshold in thresholds:
        row[_threshold_label(threshold)] = _fmt_percent(
            threshold_rows.get(threshold, {}).get("predicted_probability")
        )
    return row


def _table(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    widths = {
        column: max(len(column), *(len(row.get(column, "")) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    lines = [header, divider]
    for row in rows:
        lines.append("  ".join(row.get(column, "").ljust(widths[column]) for column in columns))
    return lines


def _gate_lines(model_gate: dict) -> list[str]:
    required = bool(model_gate.get("required"))
    passed = model_gate.get("passed")
    if passed is True:
        status = "PASS"
    elif passed is False:
        status = "FAIL"
    else:
        status = "not run"
    checks = model_gate.get("checks") or []
    lines = [f"Gate: {status} (required={str(required).lower()}, checks={len(checks)})"]
    failed = [check for check in checks if check.get("passed") is False]
    for check in failed:
        lines.append(
            "  FAIL "
            f"{check.get('name', 'unknown')}: "
            f"value={check.get('value', 'n/a')} threshold={check.get('threshold', 'n/a')}"
        )
    return lines


def build_prediction_review(payload: dict) -> str:
    """Build a human-readable report from a batch prediction JSON payload."""
    predictions = list(payload.get("predictions") or [])
    errors = list(payload.get("errors") or [])
    thresholds = _thresholds(predictions)
    columns = [
        "city",
        "source",
        "raw_f",
        "corrected_f",
        "interval_f",
        "members",
        *[_threshold_label(threshold) for threshold in thresholds],
    ]
    lines = [
        "Prediction review",
        f"Schema: {payload.get('schema_version', 'n/a')}",
        f"Generated: {payload.get('generated_at', 'n/a')}",
        f"Target date: {payload.get('target_date', 'n/a')}",
        *_gate_lines(payload.get("model_gate") or {}),
        f"Predictions: {payload.get('n_predictions', len(predictions))}",
        f"Errors: {payload.get('n_errors', len(errors))}",
    ]
    if errors:
        lines.append("Error details:")
        for error in errors:
            lines.append(f"  {error.get('city', 'n/a')}: {error.get('error', 'n/a')}")
    if predictions:
        rows = [_prediction_row(prediction, thresholds) for prediction in predictions]
        lines.extend(["", *_table(rows, columns)])
    return "\n".join(lines)


def should_fail_review(payload: dict) -> bool:
    """Return true when a review payload should block downstream use."""
    model_gate = payload.get("model_gate") or {}
    if model_gate.get("required") and model_gate.get("passed") is not True:
        return True
    return bool(payload.get("errors") or payload.get("n_errors", 0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prediction_review",
        description="Render batch prediction JSON as a compact review report.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Return zero even when the payload contains gate failures or city errors.",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_prediction_review(payload)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0 if args.allow_errors or not should_fail_review(payload) else 1


if __name__ == "__main__":
    raise SystemExit(main())

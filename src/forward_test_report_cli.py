"""Summarize accumulated forward-test settlement history."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
DEFAULT_HISTORY = Path("outputs") / "forward_test" / "history.csv"

REQUIRED_COLUMNS = {
    "target_date",
    "city",
    "actual_source",
    "observed_high_f",
    "absolute_error_f",
    "error_f",
    "offset_f",
    "predicted_probability",
    "brier",
}


def _read_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"missing forward-test history: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"history missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty forward-test history: {path}")
    return rows


def _float(row: dict[str, str], column: str) -> float | None:
    value = row.get(column)
    if value in {None, ""}:
        return None
    return float(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _unique_latest(
    rows: list[dict[str, str]],
    key_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    latest: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(str(row.get(column) or "").lower() for column in key_columns)
        if key in latest:
            del latest[key]
        latest[key] = row
    return list(latest.values())


def _source_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(str(row.get("actual_source") or "unknown") for row in rows)
    return dict(sorted(counts.items()))


def _prediction_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    errors = [
        value for row in rows if (value := _float(row, "error_f")) is not None
    ]
    absolute_errors = [
        value
        for row in rows
        if (value := _float(row, "absolute_error_f")) is not None
    ]
    return {
        "n_predictions": len(rows),
        "n_cities": len({str(row.get("city") or "").lower() for row in rows}),
        "actual_sources": _source_counts(rows),
        "mae_corrected_f": _mean(absolute_errors),
        "bias_corrected_f": _mean(errors),
    }


def _threshold_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    scores = [value for row in rows if (value := _float(row, "brier")) is not None]
    return {
        "n_threshold_events": len(rows),
        "threshold_brier_score": _mean(scores),
    }


def _interval_coverage(rows: list[dict[str, str]]) -> float | None:
    covered = 0
    total = 0
    for row in rows:
        observed = _float(row, "observed_high_f")
        lower = _float(row, "interval_lower_f")
        upper = _float(row, "interval_upper_f")
        if observed is None or lower is None or upper is None:
            continue
        total += 1
        if lower <= observed <= upper:
            covered += 1
    return covered / total if total else None


def _ece_bucket_index(probability: float, n_buckets: int) -> int:
    return min(int(probability * n_buckets), n_buckets - 1)


def _threshold_calibration_summary(
    rows: list[dict[str, str]], *, n_buckets: int
) -> dict[str, Any]:
    buckets: list[list[dict[str, float]]] = [[] for _ in range(n_buckets)]
    for row in rows:
        probability = _float(row, "predicted_probability")
        if probability is None:
            continue
        outcome_raw = str(row.get("outcome") or "").strip().lower()
        if outcome_raw not in {"true", "false"}:
            continue
        outcome = 1.0 if outcome_raw == "true" else 0.0
        bucket = _ece_bucket_index(min(max(probability, 0.0), 1.0), n_buckets)
        buckets[bucket].append({"probability": probability, "outcome": outcome})

    n_events = sum(len(bucket) for bucket in buckets)
    rows_out: list[dict[str, Any]] = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        bucket_start = index / n_buckets
        bucket_end = (index + 1) / n_buckets
        if not bucket:
            rows_out.append(
                {
                    "bucket_start": bucket_start,
                    "bucket_end": bucket_end,
                    "n": 0,
                    "mean_predicted_probability": None,
                    "observed_frequency": None,
                    "calibration_gap": None,
                    "abs_calibration_gap": None,
                }
            )
            continue
        mean_probability = sum(row["probability"] for row in bucket) / len(bucket)
        observed_frequency = sum(row["outcome"] for row in bucket) / len(bucket)
        gap = mean_probability - observed_frequency
        abs_gap = abs(gap)
        ece += (len(bucket) / n_events) * abs_gap
        rows_out.append(
            {
                "bucket_start": bucket_start,
                "bucket_end": bucket_end,
                "n": len(bucket),
                "mean_predicted_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "calibration_gap": gap,
                "abs_calibration_gap": abs_gap,
            }
        )

    return {
        "n_calibration_buckets": n_buckets,
        "threshold_ece": ece if n_events else None,
        "calibration_buckets": rows_out,
    }


def _daily_summaries(
    prediction_rows: list[dict[str, str]],
    threshold_rows: list[dict[str, str]],
    *,
    n_buckets: int,
) -> list[dict[str, Any]]:
    predictions_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    thresholds_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        predictions_by_date[str(row.get("target_date") or "")].append(row)
    for row in threshold_rows:
        thresholds_by_date[str(row.get("target_date") or "")].append(row)

    summaries: list[dict[str, Any]] = []
    for target_date in sorted(predictions_by_date):
        summaries.append(
            {
                "target_date": target_date,
                **_prediction_summary(predictions_by_date[target_date]),
                "interval_coverage": _interval_coverage(
                    predictions_by_date[target_date]
                ),
                **_threshold_summary(thresholds_by_date[target_date]),
                **_threshold_calibration_summary(
                    thresholds_by_date[target_date],
                    n_buckets=n_buckets,
                ),
            }
        )
    return summaries


def build_forward_test_report(
    history_path: Path,
    *,
    n_buckets: int = 10,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a machine-readable report from forward-test history."""
    if n_buckets < 1:
        raise ValueError("n_buckets must be at least 1")
    rows = _read_history(history_path)
    prediction_rows = _unique_latest(rows, ("target_date", "city"))
    threshold_rows = _unique_latest(rows, ("target_date", "city", "offset_f"))
    dates = sorted({str(row.get("target_date") or "") for row in prediction_rows})

    summary = {
        "n_history_rows": len(rows),
        "n_target_dates": len(dates),
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
        **_prediction_summary(prediction_rows),
        "interval_coverage": _interval_coverage(prediction_rows),
        **_threshold_summary(threshold_rows),
        **_threshold_calibration_summary(threshold_rows, n_buckets=n_buckets),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "history_path": str(history_path),
        "summary": summary,
        "daily": _daily_summaries(
            prediction_rows,
            threshold_rows,
            n_buckets=n_buckets,
        ),
    }


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_forward_test_report(payload: dict[str, Any]) -> str:
    """Render a compact operator-facing forward-test report."""
    summary = payload["summary"]
    source_counts = summary.get("actual_sources") or {}
    sources = ", ".join(f"{key}={value}" for key, value in source_counts.items())
    lines = [
        "Forward test report:",
        f"  History: {payload['history_path']}",
        (
            f"  Dates: {summary['start_date']} -> {summary['end_date']} "
            f"({summary['n_target_dates']})"
        ),
        (
            f"  Predictions: {summary['n_predictions']} "
            f"across {summary['n_cities']} cities"
        ),
        f"  Corrected MAE: {_format_float(summary['mae_corrected_f'])} F",
        f"  Corrected bias: {_format_float(summary['bias_corrected_f'])} F",
        f"  Interval coverage: {_format_float(summary.get('interval_coverage'))}",
        (
            "  Threshold Brier: "
            f"{_format_float(summary['threshold_brier_score'])} "
            f"over {summary['n_threshold_events']} events"
        ),
        f"  Threshold ECE: {_format_float(summary.get('threshold_ece'))}",
        f"  Actual sources: {sources or 'none'}",
        "Daily:",
    ]
    bias = summary.get("bias_corrected_f")
    if bias is not None and abs(float(bias)) > 0.5:
        lines.insert(
            7,
            f"  WARN: forward-test bias {_format_float(float(bias))} F exceeds 0.5 F",
        )
    for row in payload["daily"]:
        lines.append(
            "  "
            f"{row['target_date']}: cities={row['n_cities']} "
            f"mae={_format_float(row['mae_corrected_f'])} "
            f"bias={_format_float(row['bias_corrected_f'])} "
            f"coverage={_format_float(row.get('interval_coverage'))} "
            f"brier={_format_float(row['threshold_brier_score'])} "
            f"ece={_format_float(row.get('threshold_ece'))}"
        )
    return "\n".join(lines)


def write_report_payload(payload: dict[str, Any], output_path: Path | None) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True)
    if output_path is None:
        print(content)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forward_test_report",
        description="Summarize forward-test settlement history.",
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--ece-buckets",
        type=int,
        default=10,
        help="Number of probability buckets used for threshold ECE.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of text.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional JSON output path. Implies --json.",
    )
    args = parser.parse_args(argv)

    try:
        payload = build_forward_test_report(args.history, n_buckets=args.ece_buckets)
    except ValueError as error:
        print(f"Forward test report:\n  FAIL artifact_error: {error}\nOutcome: FAIL")
        return 1

    if args.json or args.out is not None:
        write_report_payload(payload, args.out)
    else:
        print(render_forward_test_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

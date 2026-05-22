"""Settle a prior daily prediction packet against observed daily highs."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from src.config import Station, get_station
from src.fetchers import asos, ncei

SETTLEMENT_SCHEMA_VERSION = "1.0"
DEFAULT_OUT_DIR = Path("outputs") / "forward_test"


@dataclass(frozen=True)
class ObservedHigh:
    high_f: float
    source: str


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _json_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _load_packet(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("prediction packet must be a JSON object")
    return payload


def _read_actuals_csv(path: Path, target: date) -> dict[str, ObservedHigh]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {"city", "target_date", "actual_high_f"} - fieldnames
        if missing:
            raise ValueError(f"actuals CSV missing columns: {sorted(missing)}")
        actuals: dict[str, ObservedHigh] = {}
        for row in reader:
            if str(row.get("target_date", "")).strip() != target.isoformat():
                continue
            city = str(row.get("city", "")).strip().lower()
            high = row.get("actual_high_f")
            if not city or high in {None, ""}:
                continue
            source = str(row.get("actual_source") or "actuals_csv")
            actuals[city] = ObservedHigh(high_f=float(high), source=source)
        return actuals


def _fetch_observed_high(station: Station, target: date) -> ObservedHigh:
    """Fetch observed high with NCEI preferred and ASOS as preliminary fallback."""
    actual = ncei.fetch_daily_high(station, target)
    if actual.high_f is not None:
        return ObservedHigh(high_f=float(actual.high_f), source=actual.source)

    text = asos.fetch_asos_csv(station.nws_station, target)
    observations = asos.parse_asos_csv(text, station.nws_station)
    high_f = asos.daily_high_from_hourly(observations, target)
    if high_f is not None:
        return ObservedHigh(high_f=float(high_f), source="asos")

    raise ValueError("no observed high from NCEI or ASOS")


def _prediction_city(prediction: dict[str, Any]) -> str:
    city = prediction.get("city")
    if not isinstance(city, str) or not city.strip():
        raise ValueError("prediction missing city")
    return city


def _prediction_target_date(packet: dict[str, Any], prediction: dict[str, Any]) -> str:
    target = prediction.get("target_date") or packet.get("target_date")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("prediction missing target_date")
    return target


def _settle_thresholds(
    prediction: dict[str, Any], observed_high_f: float
) -> list[dict[str, Any]]:
    thresholds = prediction.get("threshold_probabilities") or []
    if not isinstance(thresholds, list):
        raise ValueError("threshold_probabilities must be a list")

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        if not isinstance(threshold, dict):
            raise ValueError("threshold probability rows must be objects")
        threshold_f = int(threshold["threshold_f"])
        outcome = observed_high_f >= float(threshold_f)
        row = {
            "offset_f": int(threshold["offset_f"]),
            "threshold_f": threshold_f,
            "predicted_probability": float(threshold["predicted_probability"]),
            "outcome": outcome,
        }
        if "raw_predicted_probability" in threshold:
            row["raw_predicted_probability"] = float(
                threshold["raw_predicted_probability"]
            )
        if "recalibration_used" in threshold:
            row["recalibration_used"] = bool(threshold["recalibration_used"])
        if "recalibration_scope" in threshold:
            row["recalibration_scope"] = str(threshold["recalibration_scope"])
        if "recalibration_n" in threshold:
            row["recalibration_n"] = int(threshold["recalibration_n"])
        rows.append(row)
    return rows


def _settle_prediction(
    packet: dict[str, Any],
    prediction: dict[str, Any],
    *,
    target: date,
    actuals: dict[str, ObservedHigh] | None = None,
) -> dict[str, Any]:
    city = _prediction_city(prediction)
    target_text = _prediction_target_date(packet, prediction)
    if target_text != target.isoformat():
        raise ValueError(
            f"prediction target_date {target_text} does not match {target.isoformat()}"
        )

    if actuals is None:
        station = get_station(city)
        observed = _fetch_observed_high(station, target)
    else:
        observed = actuals.get(city.lower())
        if observed is None:
            raise ValueError("no observed high in actuals CSV")
    forecast = prediction.get("forecast") or {}
    calibration = prediction.get("calibration") or {}
    if not isinstance(forecast, dict) or not isinstance(calibration, dict):
        raise ValueError("prediction forecast/calibration must be objects")

    point_f = _json_float(forecast.get("point_f"))
    corrected_point_f = _json_float(calibration.get("corrected_point_f"))
    if point_f is None and corrected_point_f is None:
        raise ValueError("prediction missing point forecast")
    if corrected_point_f is None:
        corrected_point_f = point_f

    error_f = float(corrected_point_f) - observed.high_f
    return {
        "city": city,
        "target_date": target.isoformat(),
        "actual_source": observed.source,
        "observed_high_f": observed.high_f,
        "predicted_point_f": point_f,
        "predicted_corrected_point_f": corrected_point_f,
        "error_f": error_f,
        "absolute_error_f": abs(error_f),
        "threshold_outcomes": _settle_thresholds(prediction, observed.high_f),
    }


def _summary(rows: list[dict[str, Any]], errors: list[dict[str, str]]) -> dict[str, Any]:
    threshold_scores = [
        (float(threshold["predicted_probability"]) - float(bool(threshold["outcome"]))) ** 2
        for row in rows
        for threshold in row.get("threshold_outcomes") or []
    ]
    sources = sorted({str(row["actual_source"]) for row in rows})
    return {
        "n_predictions": len(rows) + len(errors),
        "n_settled": len(rows),
        "n_errors": len(errors),
        "actual_sources": sources,
        "mae_corrected_f": (
            sum(float(row["absolute_error_f"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "bias_corrected_f": (
            sum(float(row["error_f"]) for row in rows) / len(rows) if rows else None
        ),
        "n_threshold_events": len(threshold_scores),
        "threshold_brier_score": (
            sum(threshold_scores) / len(threshold_scores) if threshold_scores else None
        ),
    }


def build_settlement_payload(
    *,
    packet_path: Path,
    target: date,
    actuals: dict[str, ObservedHigh] | None = None,
    generated_at: datetime | None = None,
) -> tuple[dict[str, Any], int]:
    """Build a settlement payload for one prediction packet."""
    packet = _load_packet(packet_path)
    packet_target = packet.get("target_date")
    if packet_target != target.isoformat():
        raise ValueError(
            f"packet target_date {packet_target!r} does not match {target.isoformat()}"
        )

    predictions = packet.get("predictions") or []
    if not isinstance(predictions, list):
        raise ValueError("packet predictions must be a list")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            errors.append({"city": "<unknown>", "error": "prediction must be an object"})
            continue
        city = str(prediction.get("city") or "<unknown>")
        try:
            rows.append(
                _settle_prediction(packet, prediction, target=target, actuals=actuals)
            )
        except Exception as error:  # noqa: BLE001
            errors.append({"city": city, "error": str(error)})

    created_at = generated_at or datetime.now(UTC)
    payload = {
        "schema_version": SETTLEMENT_SCHEMA_VERSION,
        "generated_at": created_at.isoformat(),
        "packet_path": str(packet_path),
        "packet_generated_at": packet.get("generated_at"),
        "target_date": target.isoformat(),
        "n_rows": len(rows),
        "n_errors": len(errors),
        "summary": _summary(rows, errors),
        "rows": rows,
        "errors": errors,
    }
    return payload, 0 if not errors else 1


HISTORY_COLUMNS = [
    "settled_at",
    "packet_path",
    "target_date",
    "city",
    "actual_source",
    "observed_high_f",
    "predicted_point_f",
    "predicted_corrected_point_f",
    "error_f",
    "absolute_error_f",
    "offset_f",
    "threshold_f",
    "predicted_probability",
    "raw_predicted_probability",
    "recalibration_used",
    "recalibration_scope",
    "recalibration_n",
    "outcome",
    "brier",
]


def settlement_history_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten city settlement rows into one CSV row per threshold event."""
    history_rows: list[dict[str, Any]] = []
    for row in payload["rows"]:
        thresholds = row.get("threshold_outcomes") or []
        for threshold in thresholds:
            probability = float(threshold["predicted_probability"])
            outcome = bool(threshold["outcome"])
            history_rows.append(
                {
                    "settled_at": payload["generated_at"],
                    "packet_path": payload["packet_path"],
                    "target_date": row["target_date"],
                    "city": row["city"],
                    "actual_source": row["actual_source"],
                    "observed_high_f": row["observed_high_f"],
                    "predicted_point_f": row["predicted_point_f"],
                    "predicted_corrected_point_f": row[
                        "predicted_corrected_point_f"
                    ],
                    "error_f": row["error_f"],
                    "absolute_error_f": row["absolute_error_f"],
                    "offset_f": threshold["offset_f"],
                    "threshold_f": threshold["threshold_f"],
                    "predicted_probability": probability,
                    "raw_predicted_probability": threshold.get(
                        "raw_predicted_probability"
                    ),
                    "recalibration_used": threshold.get("recalibration_used"),
                    "recalibration_scope": threshold.get("recalibration_scope"),
                    "recalibration_n": threshold.get("recalibration_n"),
                    "outcome": outcome,
                    "brier": (probability - float(outcome)) ** 2,
                }
            )
    return history_rows


def _read_existing_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_history_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows by rewriting the full CSV through an atomic replace."""
    if not rows:
        return
    existing = _read_existing_history(path)
    output_rows = [*existing, *rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
            writer.writeheader()
            for row in output_rows:
                writer.writerow({key: row.get(key) for key in HISTORY_COLUMNS})
        Path(tmp_name).replace(path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            Path(tmp_name).unlink()
        raise


def write_settlement_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forward_test_settle",
        description="Settle a daily prediction packet against observed highs.",
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--target-date", required=True, type=_parse_date)
    parser.add_argument(
        "--actuals-csv",
        type=Path,
        help=(
            "Optional offline actuals CSV with city,target_date,actual_high_f. "
            "When supplied, no NCEI/ASOS fetch is performed."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--history", type=Path)
    args = parser.parse_args(argv)

    actuals = _read_actuals_csv(args.actuals_csv, args.target_date) if args.actuals_csv else None
    payload, exit_code = build_settlement_payload(
        packet_path=args.packet,
        target=args.target_date,
        actuals=actuals,
    )
    out_path = args.out or args.out_dir / f"{args.target_date.isoformat()}_settlement.json"
    history_path = args.history or args.out_dir / "history.csv"
    write_settlement_payload(payload, out_path)
    append_history_atomic(history_path, settlement_history_rows(payload))
    print(f"Wrote settlement JSON: {out_path}")
    print(f"Appended settlement history: {history_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

"""Report whether a prediction packet is ready for forward-test settlement."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from src.forward_test_actuals_check_cli import build_actuals_check


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _today_utc() -> date:
    return datetime.now(UTC).date()


def _load_packet(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("prediction packet must be a JSON object")
    return payload


def _packet_target_date(packet: dict[str, Any]) -> date:
    raw_target = packet.get("target_date")
    if not isinstance(raw_target, str) or not raw_target.strip():
        raise ValueError("packet missing target_date")
    try:
        return date.fromisoformat(raw_target)
    except ValueError as error:
        raise ValueError(f"invalid packet target_date: {raw_target}") from error


def _packet_city_offsets(packet: dict[str, Any]) -> dict[str, set[str]]:
    predictions = packet.get("predictions") or []
    if not isinstance(predictions, list):
        raise ValueError("packet predictions must be a list")

    expected: dict[str, set[str]] = {}
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        city = str(prediction.get("city") or "").strip().lower()
        if not city:
            continue
        thresholds = prediction.get("threshold_probabilities") or []
        offsets = {
            str(row.get("offset_f"))
            for row in thresholds
            if isinstance(row, dict) and row.get("offset_f") is not None
        }
        expected[city] = offsets
    return expected


def _history_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _settled_city_offsets(
    history_rows: list[dict[str, str]], *, target_date: date
) -> dict[str, set[str]]:
    settled: dict[str, set[str]] = {}
    for row in history_rows:
        if str(row.get("target_date") or "") != target_date.isoformat():
            continue
        city = str(row.get("city") or "").strip().lower()
        offset = str(row.get("offset_f") or "").strip()
        if not city or not offset:
            continue
        settled.setdefault(city, set()).add(offset)
    return settled


def _missing_city_offsets(
    expected: dict[str, set[str]], settled: dict[str, set[str]]
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for city, offsets in sorted(expected.items()):
        city_missing = sorted(offsets - settled.get(city, set()), key=lambda x: int(x))
        if city_missing:
            missing[city] = city_missing
    return missing


def build_pending_status(
    *,
    packet_path: Path,
    history_path: Path | None = None,
    actuals_csv: Path | None = None,
    as_of_date: date | None = None,
) -> dict[str, Any]:
    """Build a machine-readable packet settlement status payload."""
    packet = _load_packet(packet_path)
    target = _packet_target_date(packet)
    as_of = as_of_date or _today_utc()
    expected = _packet_city_offsets(packet)
    history = _history_rows(history_path)
    settled = _settled_city_offsets(history, target_date=target)
    missing = _missing_city_offsets(expected, settled)
    n_expected_offsets = sum(len(offsets) for offsets in expected.values())
    n_settled_offsets = sum(
        len(offsets & settled.get(city, set())) for city, offsets in expected.items()
    )

    if n_settled_offsets == 0:
        settlement_status = "unsettled"
    elif n_settled_offsets < n_expected_offsets:
        settlement_status = "partial"
    else:
        settlement_status = "settled"

    actuals_check = (
        build_actuals_check(packet_path=packet_path, actuals_csv=actuals_csv)
        if actuals_csv is not None
        else None
    )
    actuals_ready = actuals_check is None or actuals_check["passed"]
    ready_to_settle = target < as_of and settlement_status != "settled" and actuals_ready
    if settlement_status == "settled":
        next_action = "already_settled"
    elif target >= as_of:
        next_action = "wait_for_target_date_to_pass"
    elif not actuals_ready:
        next_action = "fill_actuals_csv"
    else:
        next_action = "run_forward_test_settle"

    payload = {
        "schema_version": "1.0",
        "packet_path": str(packet_path),
        "history_path": str(history_path) if history_path is not None else None,
        "actuals_csv": str(actuals_csv) if actuals_csv is not None else None,
        "target_date": target.isoformat(),
        "as_of_date": as_of.isoformat(),
        "packet_generated_at": packet.get("generated_at"),
        "n_predictions": len(expected),
        "n_expected_threshold_rows": n_expected_offsets,
        "n_settled_threshold_rows": n_settled_offsets,
        "settlement_status": settlement_status,
        "ready_to_settle": ready_to_settle,
        "next_action": next_action,
        "missing_city_offsets": missing,
    }
    if actuals_check is not None:
        payload["actuals_check"] = actuals_check
    return payload


def render_pending_status(payload: dict[str, Any]) -> str:
    """Render a compact operator-facing pending settlement status."""
    lines = [
        f"Packet: {payload['packet_path']}",
        f"Target date: {payload['target_date']} (as of {payload['as_of_date']})",
        (
            f"Settlement: {payload['settlement_status']} "
            f"({payload['n_settled_threshold_rows']}/"
            f"{payload['n_expected_threshold_rows']} threshold rows)"
        ),
        f"Ready to settle: {str(payload['ready_to_settle']).lower()}",
        f"Next action: {payload['next_action']}",
    ]
    actuals_check = payload.get("actuals_check")
    if actuals_check is not None:
        lines.append(
            "Actuals CSV: "
            f"{'PASS' if actuals_check['passed'] else 'FAIL'} "
            f"({actuals_check['n_valid_actuals']}/"
            f"{actuals_check['n_expected_cities']} cities)"
        )
    missing = payload.get("missing_city_offsets") or {}
    if missing:
        missing_bits = [
            f"{city}={','.join(offsets)}" for city, offsets in sorted(missing.items())
        ]
        lines.append(f"Missing: {'; '.join(missing_bits)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forward_test_pending",
        description="Show whether a prediction packet needs forward-test settlement.",
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument(
        "--actuals-csv",
        type=Path,
        help="Optional offline actuals CSV to include in readiness status.",
    )
    parser.add_argument("--as-of-date", type=_parse_date)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_pending_status(
        packet_path=args.packet,
        history_path=args.history,
        actuals_csv=args.actuals_csv,
        as_of_date=args.as_of_date,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_pending_status(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

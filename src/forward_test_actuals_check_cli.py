"""Validate an offline actuals CSV before forward-test settlement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.forward_test_actuals_template_cli import _load_packet, build_actuals_template_rows

ACTUALS_CHECK_SCHEMA_VERSION = "1.0"
REQUIRED_ACTUALS_COLUMNS = {"city", "target_date", "actual_high_f"}


def _packet_target_date(packet: dict[str, Any]) -> str:
    target_date = packet.get("target_date")
    if not isinstance(target_date, str) or not target_date.strip():
        raise ValueError("packet missing target_date")
    return target_date


def _expected_cities(packet: dict[str, Any]) -> list[str]:
    rows = build_actuals_template_rows(packet)
    return [row["city"] for row in rows]


def _read_actual_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_ACTUALS_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"actuals CSV missing columns: {missing}")
        return [dict(row) for row in reader], sorted(fieldnames)


def _actual_value_error(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return "missing actual_high_f"
    try:
        float(text)
    except ValueError:
        return "actual_high_f is not numeric"
    return None


def build_actuals_check(
    *,
    packet_path: Path,
    actuals_csv: Path,
    allow_extra: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable validation payload for an actuals CSV."""
    packet = _load_packet(packet_path)
    target_date = _packet_target_date(packet)
    expected_cities = _expected_cities(packet)
    expected_set = set(expected_cities)
    actual_rows, columns = _read_actual_rows(actuals_csv)

    rows_for_target: dict[str, dict[str, str]] = {}
    duplicate_cities: set[str] = set()
    extra_cities: set[str] = set()
    row_errors: list[dict[str, str]] = []

    for row in actual_rows:
        if str(row.get("target_date") or "").strip() != target_date:
            continue
        city = str(row.get("city") or "").strip().lower()
        if not city:
            row_errors.append({"city": "", "error": "missing city"})
            continue
        if city in rows_for_target:
            duplicate_cities.add(city)
            continue
        rows_for_target[city] = row
        if city not in expected_set:
            extra_cities.add(city)
            continue
        value_error = _actual_value_error(str(row.get("actual_high_f") or ""))
        if value_error:
            row_errors.append({"city": city, "error": value_error})

    missing_cities = [city for city in expected_cities if city not in rows_for_target]
    if not allow_extra:
        for city in sorted(extra_cities):
            row_errors.append({"city": city, "error": "extra city for packet target_date"})
    for city in sorted(duplicate_cities):
        row_errors.append({"city": city, "error": "duplicate city for packet target_date"})
    for city in missing_cities:
        row_errors.append({"city": city, "error": "missing city for packet target_date"})

    valid_cities = [
        city
        for city in expected_cities
        if city in rows_for_target
        and not _actual_value_error(str(rows_for_target[city].get("actual_high_f") or ""))
    ]
    return {
        "schema_version": ACTUALS_CHECK_SCHEMA_VERSION,
        "packet_path": str(packet_path),
        "actuals_csv": str(actuals_csv),
        "target_date": target_date,
        "columns": columns,
        "allow_extra": allow_extra,
        "passed": not row_errors,
        "n_expected_cities": len(expected_cities),
        "n_valid_actuals": len(valid_cities),
        "missing_cities": missing_cities,
        "extra_cities": sorted(extra_cities),
        "duplicate_cities": sorted(duplicate_cities),
        "errors": row_errors,
    }


def render_actuals_check(payload: dict[str, Any]) -> str:
    """Render a compact operator-facing actuals preflight result."""
    lines = [
        f"Actuals CSV: {'PASS' if payload['passed'] else 'FAIL'}",
        f"Target date: {payload['target_date']}",
        f"Valid actuals: {payload['n_valid_actuals']}/{payload['n_expected_cities']}",
    ]
    if payload["errors"]:
        lines.append("Errors:")
        for error in payload["errors"]:
            city = error.get("city") or "<blank>"
            lines.append(f"  - {city}: {error['error']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forward_test_actuals_check",
        description="Validate an offline actuals CSV before forward-test settlement.",
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--actuals-csv", required=True, type=Path)
    parser.add_argument(
        "--allow-extra",
        action="store_true",
        help="Allow actual rows for cities not present in the packet target date.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_actuals_check(
        packet_path=args.packet,
        actuals_csv=args.actuals_csv,
        allow_extra=args.allow_extra,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_actuals_check(payload))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

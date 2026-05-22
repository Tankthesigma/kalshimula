"""Create an offline actuals CSV template for forward-test settlement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ACTUALS_TEMPLATE_COLUMNS = [
    "city",
    "target_date",
    "actual_high_f",
    "actual_source",
]


def _load_packet(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("prediction packet must be a JSON object")
    return payload


def build_actuals_template_rows(packet: dict[str, Any]) -> list[dict[str, str]]:
    """Return one blank actual-high row per prediction city in a packet."""
    target_date = packet.get("target_date")
    if not isinstance(target_date, str) or not target_date.strip():
        raise ValueError("packet missing target_date")
    predictions = packet.get("predictions") or []
    if not isinstance(predictions, list):
        raise ValueError("packet predictions must be a list")

    rows = []
    seen = set()
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        city = str(prediction.get("city") or "").strip().lower()
        if not city or city in seen:
            continue
        seen.add(city)
        rows.append(
            {
                "city": city,
                "target_date": target_date,
                "actual_high_f": "",
                "actual_source": "",
            }
        )
    return rows


def _read_existing_actuals(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {"city", "target_date"} - fieldnames
        if missing:
            raise ValueError(f"existing actuals CSV missing columns: {sorted(missing)}")
        rows = {}
        for row in reader:
            city = str(row.get("city") or "").strip().lower()
            target_date = str(row.get("target_date") or "").strip()
            if not city or not target_date:
                continue
            rows[(city, target_date)] = {
                "actual_high_f": str(row.get("actual_high_f") or "").strip(),
                "actual_source": str(row.get("actual_source") or "").strip(),
            }
        return rows


def merge_existing_actuals(
    rows: list[dict[str, str]], existing: dict[tuple[str, str], dict[str, str]]
) -> list[dict[str, str]]:
    """Copy existing actual values into regenerated template rows."""
    merged = []
    for row in rows:
        out = dict(row)
        existing_row = existing.get((row["city"], row["target_date"]))
        if existing_row:
            out["actual_high_f"] = existing_row.get("actual_high_f", "")
            out["actual_source"] = existing_row.get("actual_source", "")
        merged.append(out)
    return merged


def write_actuals_template(rows: list[dict[str, str]], path: Path) -> None:
    """Write actuals template rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTUALS_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forward_test_actuals_template",
        description="Write a blank actuals CSV template for a prediction packet.",
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--no-preserve-existing",
        action="store_true",
        help="Overwrite an existing template without carrying forward actual values.",
    )
    args = parser.parse_args(argv)

    rows = build_actuals_template_rows(_load_packet(args.packet))
    if not args.no_preserve_existing:
        rows = merge_existing_actuals(rows, _read_existing_actuals(args.out))
    write_actuals_template(rows, args.out)
    print(f"Wrote actuals template: {args.out} ({len(rows)} cities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

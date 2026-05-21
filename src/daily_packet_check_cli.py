"""Verify a daily model packet manifest and its referenced artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _artifact_checks(artifacts: dict[str, Any]) -> list[dict]:
    checks = []
    for name, raw_path in sorted(artifacts.items()):
        path = Path(str(raw_path))
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        checks.append(
            {
                "name": f"artifact:{name}",
                "passed": exists and size > 0,
                "detail": f"{path} ({size} bytes)" if exists else f"{path} missing",
            }
        )
    return checks


def _step_checks(steps: dict[str, Any]) -> list[dict]:
    checks = []
    for name, step in sorted(steps.items()):
        exit_code = int(step.get("exit_code", 1))
        checks.append(
            {
                "name": f"step:{name}",
                "passed": exit_code == 0,
                "detail": f"exit_code={exit_code}",
            }
        )
    return checks


def _load_json_artifact(artifacts: dict[str, Any], name: str) -> tuple[dict | None, str]:
    raw_path = artifacts.get(name)
    if raw_path is None:
        return None, f"artifact path missing: {name}"
    path = Path(str(raw_path))
    try:
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{path}: {error}"


def _prediction_json_checks(payload: dict) -> list[dict]:
    artifacts = payload.get("artifacts") or {}
    prediction_json, detail = _load_json_artifact(artifacts, "prediction_json")
    if prediction_json is None:
        return [
            {
                "name": "prediction_json:readable",
                "passed": False,
                "detail": detail,
            }
        ]

    predictions = prediction_json.get("predictions") or []
    errors = prediction_json.get("errors") or []
    model_gate = prediction_json.get("model_gate") or {}
    required_prediction_fields = {
        "artifact_paths",
        "calibration",
        "city",
        "forecast",
        "selected_source",
        "selected_source_applied",
        "station",
        "threshold_probabilities",
    }
    missing_field_rows = []
    for index, prediction in enumerate(predictions):
        missing = sorted(required_prediction_fields - set(prediction))
        if missing:
            missing_field_rows.append(f"row {index}: {missing}")

    return [
        {
            "name": "prediction_json:schema_version",
            "passed": prediction_json.get("schema_version") == "1.0",
            "detail": f"schema_version={prediction_json.get('schema_version', 'missing')}",
        },
        {
            "name": "prediction_json:gate",
            "passed": (not model_gate.get("required")) or model_gate.get("passed") is True,
            "detail": (
                f"required={model_gate.get('required', 'missing')} "
                f"passed={model_gate.get('passed', 'missing')}"
            ),
        },
        {
            "name": "prediction_json:error_count",
            "passed": int(prediction_json.get("n_errors", len(errors))) == 0 and not errors,
            "detail": (
                f"n_errors={prediction_json.get('n_errors', 'missing')} "
                f"errors={len(errors)}"
            ),
        },
        {
            "name": "prediction_json:prediction_count",
            "passed": int(prediction_json.get("n_predictions", -1)) == len(predictions)
            and len(predictions) > 0,
            "detail": (
                f"n_predictions={prediction_json.get('n_predictions', 'missing')} "
                f"rows={len(predictions)}"
            ),
        },
        {
            "name": "prediction_json:prediction_fields",
            "passed": not missing_field_rows,
            "detail": "ok" if not missing_field_rows else "; ".join(missing_field_rows),
        },
    ]


def build_packet_checks(manifest_path: Path) -> tuple[dict, list[dict]]:
    """Return manifest payload and packet verification checks."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = [
        {
            "name": "manifest:schema_version",
            "passed": payload.get("schema_version") == "1.0",
            "detail": f"schema_version={payload.get('schema_version', 'missing')}",
        },
        {
            "name": "manifest:exit_code",
            "passed": int(payload.get("exit_code", 1)) == 0,
            "detail": f"exit_code={payload.get('exit_code', 'missing')}",
        },
    ]
    checks.extend(_step_checks(payload.get("steps") or {}))
    checks.extend(_artifact_checks(payload.get("artifacts") or {}))
    checks.extend(_prediction_json_checks(payload))
    return payload, checks


def render_packet_check_report(manifest_path: Path, payload: dict, checks: list[dict]) -> str:
    lines = [
        "Daily packet check:",
        f"  manifest: {manifest_path}",
        f"  generated_at: {payload.get('generated_at', 'n/a')}",
        f"  target_date: {payload.get('target_date', 'n/a')}",
        f"  cities: {payload.get('cities', 'n/a')}",
        f"  require_gate: {str(bool(payload.get('require_gate'))).lower()}",
    ]
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"  {status} {check['name']}: {check['detail']}")
    outcome = "PASS" if all(check["passed"] for check in checks) else "FAIL"
    lines.append(f"Outcome: {outcome}")
    return "\n".join(lines)


def build_packet_check_payload(manifest_path: Path, payload: dict, checks: list[dict]) -> dict:
    """Build a machine-readable packet-check result."""
    return {
        "schema_version": "1.0",
        "manifest": str(manifest_path),
        "generated_at": payload.get("generated_at"),
        "target_date": payload.get("target_date"),
        "cities": payload.get("cities"),
        "require_gate": bool(payload.get("require_gate")),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="daily_packet_check",
        description="Verify a daily model packet manifest before consuming it.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    payload, checks = build_packet_checks(args.manifest)
    if args.json:
        output = json.dumps(
            build_packet_check_payload(args.manifest, payload, checks),
            indent=2,
            sort_keys=True,
        )
    else:
        output = render_packet_check_report(args.manifest, payload, checks)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0 if all(check["passed"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

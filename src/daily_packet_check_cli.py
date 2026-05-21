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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="daily_packet_check",
        description="Verify a daily model packet manifest before consuming it.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    payload, checks = build_packet_checks(args.manifest)
    report = render_packet_check_report(args.manifest, payload, checks)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0 if all(check["passed"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

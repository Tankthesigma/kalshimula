"""CLI for source-vs-consensus contrarian diagnostics."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.models.source_contrarian_diagnostics import (
    DEFAULT_OFFSETS,
    write_source_contrarian_diagnostics,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="source_contrarian_diagnostics",
        description="Write source-vs-consensus diagnostic artifacts.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--offsets",
        default=",".join(str(offset) for offset in DEFAULT_OFFSETS),
        help="Comma-separated threshold offsets around rounded blend point.",
    )
    args = parser.parse_args(_normalize_offset_arg(argv))

    offsets = _parse_offsets(args.offsets)
    command_args = {
        "input": str(args.input),
        "out_dir": str(args.out_dir),
        "offsets": list(offsets),
    }
    result = write_source_contrarian_diagnostics(
        input_path=args.input,
        output_dir=args.out_dir,
        offsets=offsets,
        command_args=command_args,
        git_commit=_git_commit(),
    )
    promoted = int(result.source_contrarian_summary["promoted"].sum())
    print(
        f"Wrote source contrarian diagnostics to {args.out_dir}: "
        f"{len(result.daily_source_deltas)} daily rows, "
        f"{len(result.source_contrarian_summary)} source rows, "
        f"{promoted} promoted"
    )
    return 0


def _parse_offsets(raw: str) -> tuple[float, ...]:
    offsets = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not offsets:
        raise ValueError("offsets must contain at least one value")
    return offsets


def _normalize_offset_arg(argv: list[str] | None) -> list[str] | None:
    """Allow both ``--offsets=-2,0,2`` and ``--offsets -2,0,2`` forms."""
    if argv is None:
        return None
    normalized = list(argv)
    for index, value in enumerate(normalized[:-1]):
        if value == "--offsets" and normalized[index + 1].startswith("-"):
            normalized[index] = f"--offsets={normalized[index + 1]}"
            del normalized[index + 1]
            break
    return normalized


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())

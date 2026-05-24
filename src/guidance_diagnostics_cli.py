"""CLI for professional guidance no-leak diagnostics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.models.guidance import write_guidance_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--as-of", required=True, help="UTC ISO timestamp")
    parser.add_argument("--target-date")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = write_guidance_diagnostics(
        input_path=args.input,
        output_dir=args.out_dir,
        as_of_ts=args.as_of,
        target_date=args.target_date,
        git_commit=_git_commit(),
    )
    print(
        f"Wrote guidance diagnostics to {args.out_dir}: "
        f"{len(result.latest)} latest rows, {len(result.summary)} summary rows"
    )
    return 0


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""CLI for a weather-only nowcast report from frozen prediction rows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.models.nowcast_report import write_nowcast_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-nowcast", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = write_nowcast_report(
        predictions_path=args.predictions_nowcast,
        output_dir=args.out_dir,
        git_commit=_git_commit(),
    )
    print(
        f"Wrote {len(result.summary)} city rows to "
        f"{args.out_dir / 'nowcast_report_summary.csv'}"
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

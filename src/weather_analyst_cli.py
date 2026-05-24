"""CLI for deterministic weather-only analyst packets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.models.weather_analyst import write_weather_analyst_packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nowcast-summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--guidance-comparison", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = write_weather_analyst_packet(
        nowcast_summary_path=args.nowcast_summary,
        guidance_comparison_path=args.guidance_comparison,
        output_dir=args.out_dir,
        git_commit=_git_commit(),
    )
    print(
        f"Wrote {len(result.rows)} weather analyst rows to "
        f"{args.out_dir / 'weather_analyst_packet.csv'}"
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

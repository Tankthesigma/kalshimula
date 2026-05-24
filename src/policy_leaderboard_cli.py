"""CLI for model policy leaderboard reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.policy_leaderboard import write_policy_leaderboard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="policy_leaderboard",
        description="Write model-only policy leaderboard artifacts.",
    )
    parser.add_argument("--walkforward-summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--source-contrarian-summary", type=Path)
    args = parser.parse_args(argv)

    leaderboard = write_policy_leaderboard(
        walkforward_summary_path=args.walkforward_summary,
        output_dir=args.out_dir,
        source_contrarian_summary_path=args.source_contrarian_summary,
    )
    print(f"Wrote policy leaderboard to {args.out_dir}: {len(leaderboard)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

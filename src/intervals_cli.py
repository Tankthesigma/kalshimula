"""CLI for fitting empirical forecast intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.intervals import write_interval_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="intervals",
        description="Fit empirical city/source forecast intervals.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", default=0.2, type=float)
    args = parser.parse_args(argv)

    table = write_interval_table(args.input, args.out, alpha=args.alpha)
    print(f"Wrote {len(table)} interval rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

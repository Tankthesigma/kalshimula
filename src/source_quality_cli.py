"""CLI wrapper around :mod:`src.source_quality`.

Usage::

    python -m src.source_quality_cli --input smoke.csv --out smoke_quality.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.source_quality import write_source_quality


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="source_quality_cli",
        description="Summarize smoke-test results into per-source reliability stats.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a smoke results CSV.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the summary CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = write_source_quality(Path(args.input), Path(args.out))
    print(f"{len(summary)} source rows summarized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

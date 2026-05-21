"""CLI for writing residual diagnostic summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.diagnostics import write_residual_diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="residual_diagnostics",
        description="Write city/source residual diagnostic CSVs.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    diagnostics = write_residual_diagnostics(input_path=args.input, output_dir=args.out_dir)
    print(
        f"Wrote residual diagnostics to {args.out_dir}: "
        f"{len(diagnostics.source_summary)} source rows, "
        f"{len(diagnostics.monthly_summary)} monthly rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

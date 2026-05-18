"""CLI for writing model report artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.report import write_model_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model_report",
        description="Write raw, bias-corrected, and interval model report CSVs.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--alpha", default=0.2, type=float)
    args = parser.parse_args(argv)

    report = write_model_report(
        input_path=args.input,
        output_dir=args.out_dir,
        alpha=args.alpha,
    )
    print(
        f"Wrote report to {args.out_dir}: "
        f"{len(report.raw_summary)} raw rows, "
        f"{len(report.corrected_evaluation)} corrected rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

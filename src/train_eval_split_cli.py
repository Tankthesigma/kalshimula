"""CLI for leakage-safe train/test evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.train_eval import write_train_eval_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="train_eval_split",
        description="Fit bias/intervals on train dates and evaluate on test dates.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--test-start")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument(
        "--split-strategy",
        choices=["date", "month-stratified"],
        default="date",
        help="Use chronological date split or diagnostic month-stratified split.",
    )
    parser.add_argument("--test-fraction", default=0.2, type=float)
    args = parser.parse_args(argv)
    if args.split_strategy == "date" and not args.test_start:
        parser.error("--test-start is required when --split-strategy=date")

    result = write_train_eval_outputs(
        input_path=args.input,
        output_dir=args.out_dir,
        test_start=args.test_start,
        alpha=args.alpha,
        split_strategy=args.split_strategy,
        test_fraction=args.test_fraction,
    )
    print(
        f"Wrote train/test evaluation to {args.out_dir}: "
        f"{len(result.train_rows)} train rows, {len(result.test_rows)} test rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

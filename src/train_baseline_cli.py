"""CLI for training/evaluating the dependency-free baseline model."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.baseline_training import write_baseline_training_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="train_baseline",
        description="Fit bias-corrected baseline artifacts from collected rows.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--bias-out", required=True, type=Path)
    parser.add_argument("--evaluation-out", required=True, type=Path)
    args = parser.parse_args(argv)

    result = write_baseline_training_outputs(
        input_path=args.input,
        bias_out=args.bias_out,
        evaluation_out=args.evaluation_out,
    )
    print(
        f"Wrote {len(result.bias_table)} bias rows to {args.bias_out} and "
        f"{len(result.evaluation)} evaluation rows to {args.evaluation_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

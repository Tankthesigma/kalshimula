"""CLI for threshold probability calibration diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.threshold_calibration import write_threshold_calibration_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="threshold_calibration",
        description="Evaluate threshold event probabilities from empirical residuals.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--recommended-sources", required=True, type=Path)
    parser.add_argument("--bias-table", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument(
        "--offsets",
        default="-4,-2,0,2,4",
        help="Comma-separated threshold offsets around rounded corrected point.",
    )
    parser.add_argument("--buckets", default=10, type=int)
    args = parser.parse_args(argv)

    result = write_threshold_calibration_outputs(
        input_path=args.input,
        recommended_sources_path=args.recommended_sources,
        bias_table_path=args.bias_table,
        output_dir=args.out_dir,
        validation_start=args.validation_start,
        test_start=args.test_start,
        offsets=_parse_int_list(args.offsets, name="offsets"),
        n_buckets=args.buckets,
    )
    test = result.summary[result.summary["split"] == "test"].iloc[0]
    print(
        f"Wrote threshold calibration to {args.out_dir}: "
        f"{int(test['n_events'])} test events, brier={test['brier_score']:.4f}"
    )
    return 0


def _parse_int_list(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise SystemExit(f"--{name} must be a comma-separated integer list") from exc
    if not parsed:
        raise SystemExit(f"--{name} must contain at least one value")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

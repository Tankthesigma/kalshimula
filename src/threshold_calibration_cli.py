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
    parser.add_argument(
        "--recalibration-prior-strength",
        default=25.0,
        type=float,
        help="Shrinkage strength for validation-bucket probability recalibration.",
    )
    parser.add_argument(
        "--min-recalibration-events",
        default=20,
        type=int,
        help=(
            "Minimum validation events required to apply a city/source bucket "
            "mapping before falling back to a pooled global bucket."
        ),
    )
    parser.add_argument(
        "--gap-min-events",
        default=20,
        type=int,
        help="Minimum test events per city/source bucket in the gap report.",
    )
    parser.add_argument(
        "--gap-probability-min",
        default=0.2,
        type=float,
        help="Lowest raw probability bucket edge included in the gap report.",
    )
    parser.add_argument(
        "--gap-probability-max",
        default=0.8,
        type=float,
        help="Highest raw probability bucket edge included in the gap report.",
    )
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
        recalibration_prior_strength=args.recalibration_prior_strength,
        min_recalibration_events=args.min_recalibration_events,
        probability_gap_min_events=args.gap_min_events,
        probability_gap_min=args.gap_probability_min,
        probability_gap_max=args.gap_probability_max,
    )
    test = result.summary[result.summary["split"] == "test"].iloc[0]
    recalibrated = result.recalibration_comparison[
        result.recalibration_comparison["policy"] == "validation_bucket_recalibrated"
    ].iloc[0]
    print(
        f"Wrote threshold calibration to {args.out_dir}: "
        f"{int(test['n_events'])} test events, "
        f"raw brier={test['brier_score']:.4f}, "
        f"recalibrated brier={recalibrated['brier_score']:.4f}, "
        f"gap buckets={len(result.probability_gap_report)}"
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

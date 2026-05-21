"""CLI for validation-calibrated interval alpha policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.interval_policy import write_interval_policy_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="interval_policy",
        description="Select per-city interval alpha from validation coverage.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--recommended-sources", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument(
        "--alphas",
        default="0.2,0.13,0.1,0.05",
        help="Comma-separated interval alphas to compare.",
    )
    parser.add_argument(
        "--target-coverage",
        default=0.8,
        type=float,
        help="Observed validation coverage threshold used to rank configs.",
    )
    args = parser.parse_args(argv)

    result = write_interval_policy_outputs(
        input_path=args.input,
        recommended_sources_path=args.recommended_sources,
        output_dir=args.out_dir,
        validation_start=args.validation_start,
        test_start=args.test_start,
        alphas=_parse_float_list(args.alphas, name="alphas"),
        target_coverage=args.target_coverage,
    )
    test_rows = result.comparison[result.comparison["split"] == "test"]
    recommended = test_rows[test_rows["recommended"]]
    if recommended.empty:
        summary = "no recommended test row"
    else:
        row = recommended.iloc[0]
        summary = (
            f"coverage={row['interval_coverage_raw']:.3f}, "
            f"width={row['interval_width_raw']:.2f}"
        )
    print(
        f"Wrote interval-policy outputs to {args.out_dir}: "
        f"{len(result.selected_policy)} city/source policies, {summary}"
    )
    return 0


def _parse_float_list(value: str, *, name: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise SystemExit(f"--{name} must be a comma-separated numeric list") from exc
    if not parsed:
        raise SystemExit(f"--{name} must contain at least one value")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

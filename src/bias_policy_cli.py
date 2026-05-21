"""CLI for comparing global and per-city bias policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.bias_policy import write_bias_policy_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bias_policy",
        description="Compare bias policies and write recommended model artifacts.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-eval-dir", required=True, type=Path)
    parser.add_argument("--recommended-sources", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument(
        "--recent-days",
        default="90,180,365",
        help="Comma-separated recent bias windows to compare.",
    )
    parser.add_argument(
        "--alphas",
        default="0.2,0.13",
        help="Comma-separated empirical interval alphas to compare.",
    )
    parser.add_argument(
        "--target-coverage",
        default=0.8,
        type=float,
        help="Observed coverage threshold used to rank configs.",
    )
    args = parser.parse_args(argv)

    result = write_bias_policy_outputs(
        input_path=args.input,
        train_eval_dir=args.train_eval_dir,
        recommended_sources_path=args.recommended_sources,
        output_dir=args.out_dir,
        validation_start=args.validation_start,
        test_start=args.test_start,
        recent_days=_parse_int_list(args.recent_days, name="recent-days"),
        alphas=_parse_float_list(args.alphas, name="alphas"),
        target_coverage=args.target_coverage,
    )
    selected = result.recommended_policy.iloc[0]
    print(
        f"Wrote bias-policy comparison to {args.out_dir}: selected "
        f"{selected['policy']} alpha={selected['alpha']}"
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

"""CLI for leakage-safe walk-forward evaluation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.models.walkforward_eval import DEFAULT_THRESHOLDS, write_walkforward_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="walkforward_eval",
        description="Run leakage-safe walk-forward evaluation over source rows.",
    )
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cities", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--train-window-days", type=int, required=True)
    parser.add_argument("--test-window-days", type=int, required=True)
    parser.add_argument("--step-days", type=int, required=True)
    parser.add_argument(
        "--threshold-offsets",
        default=",".join(str(offset) for offset in DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--holdout-start")
    parser.add_argument("--holdout-end")
    parser.add_argument("--purge-days-before", type=int, default=0)
    parser.add_argument("--purge-days-after", type=int, default=0)
    args = parser.parse_args(_normalize_negative_value_arg(argv, "--threshold-offsets"))

    cities = _parse_csv(args.cities)
    sources = _parse_csv(args.sources)
    offsets = tuple(float(value) for value in _parse_csv(args.threshold_offsets))
    command_args = {
        "rows": str(args.rows),
        "out_dir": str(args.out_dir),
        "cities": cities,
        "sources": sources,
        "train_window_days": args.train_window_days,
        "test_window_days": args.test_window_days,
        "step_days": args.step_days,
        "threshold_offsets": list(offsets),
        "holdout_start": args.holdout_start,
        "holdout_end": args.holdout_end,
        "purge_days_before": args.purge_days_before,
        "purge_days_after": args.purge_days_after,
    }
    result = write_walkforward_outputs(
        rows_path=args.rows,
        output_dir=args.out_dir,
        cities=cities,
        sources=sources,
        train_window_days=args.train_window_days,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
        threshold_offsets=offsets,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        purge_days_before=args.purge_days_before,
        purge_days_after=args.purge_days_after,
        command_args=command_args,
        git_commit=_git_commit(),
    )
    print(
        f"Wrote walk-forward evaluation to {args.out_dir}: "
        f"{len(result.predictions)} predictions, {len(result.events)} threshold events"
    )
    return 0


def _parse_csv(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("comma-separated argument must contain at least one value")
    return values


def _normalize_negative_value_arg(argv: list[str] | None, option: str) -> list[str] | None:
    if argv is None:
        return None
    normalized = list(argv)
    for index, value in enumerate(normalized[:-1]):
        if value == option and normalized[index + 1].startswith("-"):
            normalized[index] = f"{option}={normalized[index + 1]}"
            del normalized[index + 1]
            break
    return normalized


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())

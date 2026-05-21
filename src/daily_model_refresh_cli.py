"""One-command daily model refresh for gated predictions and review output."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from src import predict_batch_cli, prediction_review_cli
from src.config import load_stations


@dataclass(frozen=True)
class RefreshPaths:
    json_out: Path
    review_out: Path


def build_refresh_paths(
    *,
    model_run_dir: Path,
    out_dir: Path | None,
    prefix: str,
) -> RefreshPaths:
    """Return stable output paths for the daily refresh artifacts."""
    directory = out_dir or model_run_dir
    return RefreshPaths(
        json_out=directory / f"{prefix}.json",
        review_out=directory / f"{prefix}.txt",
    )


def _normalize_threshold_offsets(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if (
            arg == "--threshold-offsets"
            and index + 1 < len(argv)
            and argv[index + 1].startswith("-")
            and not argv[index + 1].startswith("--")
        ):
            normalized.append(f"--threshold-offsets={argv[index + 1]}")
            index += 2
            continue
        normalized.append(arg)
        index += 1
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="daily_model_refresh",
        description="Write gated batch prediction JSON and a text review report.",
    )
    parser.add_argument("--model-run-dir", required=True, type=Path)
    parser.add_argument(
        "--cities",
        default=",".join(load_stations().keys()),
        help="Comma-separated city slugs. Defaults to all configured cities.",
    )
    parser.add_argument("--date", default="tomorrow")
    parser.add_argument("--threshold-offsets", default="-2,0,2")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--prefix", default="latest_predictions")
    parser.add_argument(
        "--no-require-gate",
        action="store_true",
        help="Diagnostic mode: do not require model readiness gate before predictions.",
    )
    args = parser.parse_args(_normalize_threshold_offsets(list(argv or sys.argv[1:])))

    paths = build_refresh_paths(
        model_run_dir=args.model_run_dir,
        out_dir=args.out_dir,
        prefix=args.prefix,
    )
    batch_args = [
        "--cities",
        args.cities,
        "--date",
        args.date,
        "--model-run-dir",
        str(args.model_run_dir),
        f"--threshold-offsets={args.threshold_offsets}",
        "--out",
        str(paths.json_out),
    ]
    if not args.no_require_gate:
        batch_args.append("--require-gate")

    batch_code = predict_batch_cli.main(batch_args)
    review_code = prediction_review_cli.main(
        ["--input", str(paths.json_out), "--out", str(paths.review_out)]
    )
    print(f"Wrote prediction JSON: {paths.json_out}")
    print(f"Wrote prediction review: {paths.review_out}")
    return batch_code if batch_code != 0 else review_code


if __name__ == "__main__":
    raise SystemExit(main())

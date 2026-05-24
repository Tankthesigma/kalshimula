"""CLI for leakage-safe climate feature diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.climate_features import write_climate_feature_diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="climate_feature_diagnostics",
        description="Write leakage-safe climate/trend feature diagnostics.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    diagnostics = write_climate_feature_diagnostics(
        input_path=args.input,
        output_dir=args.out_dir,
    )
    print(
        f"Wrote climate feature diagnostics to {args.out_dir}: "
        f"{len(diagnostics.features)} feature rows, {len(diagnostics.summary)} city rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

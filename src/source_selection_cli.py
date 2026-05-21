"""CLI for validation-driven forecast source selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.source_selection import write_source_selection_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="source_selection",
        description="Select one forecast source per city using validation MAE.",
    )
    parser.add_argument("--validation-scores", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    result = write_source_selection_outputs(
        validation_scores_path=args.validation_scores,
        evaluation_path=args.evaluation,
        output_dir=args.out_dir,
    )
    avg_mae = (
        result.summary.iloc[0]["mae_corrected"]
        if not result.summary.empty and "mae_corrected" in result.summary.columns
        else "n/a"
    )
    print(
        f"Wrote {len(result.selected_sources)} selected sources to {args.out_dir} "
        f"(avg corrected MAE: {avg_mae})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

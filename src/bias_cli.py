"""CLI for fitting bias corrections from collected backtest rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.bias import write_bias_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bias",
        description="Fit city/source bias corrections from collected rows.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    table = write_bias_table(args.input, args.out)
    print(f"Wrote {len(table)} bias rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

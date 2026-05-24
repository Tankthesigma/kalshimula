"""CLI for source provenance diagnostics."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.models.source_provenance import write_source_provenance_diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="source_provenance",
        description="Detect duplicate or near-duplicate weather source series.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    result = write_source_provenance_diagnostics(
        input_path=args.input,
        output_dir=args.out_dir,
        git_commit=_git_commit(),
    )
    print(
        f"Wrote source provenance diagnostics to {args.out_dir}: "
        f"{len(result.summary)} source-pair rows"
    )
    return 0


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

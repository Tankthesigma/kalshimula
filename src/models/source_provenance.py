"""Weather-source provenance diagnostics.

Detects suspicious duplicate or near-duplicate source series without using
market data. This protects ensemble/source logic from double-counting the same
underlying forecast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import pandas as pd

PROVENANCE_COLUMNS = [
    "city",
    "source_a",
    "source_b",
    "n_overlap",
    "mean_abs_diff_f",
    "max_abs_diff_f",
    "correlation",
    "identical_rate",
    "duplicate_flag",
]


@dataclass(frozen=True)
class SourceProvenanceDiagnostics:
    summary: pd.DataFrame
    report: str
    manifest: dict[str, object]


def build_source_provenance_diagnostics(
    rows: pd.DataFrame,
    *,
    identical_tolerance_f: float = 0.01,
    duplicate_identical_rate: float = 0.999,
    input_path: str | None = None,
    input_sha256: str | None = None,
    git_commit: str | None = None,
) -> SourceProvenanceDiagnostics:
    """Compare source point series for duplicate/provenance risk."""
    summary = summarize_source_provenance(
        rows,
        identical_tolerance_f=identical_tolerance_f,
        duplicate_identical_rate=duplicate_identical_rate,
    )
    report = render_source_provenance_report(summary)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "input_path": input_path,
        "input_sha256": input_sha256,
        "row_counts": {
            "input_rows": int(len(rows)),
            "summary_rows": int(len(summary)),
            "duplicate_flags": int(summary["duplicate_flag"].sum()) if not summary.empty else 0,
        },
        "thresholds": {
            "identical_tolerance_f": identical_tolerance_f,
            "duplicate_identical_rate": duplicate_identical_rate,
        },
    }
    return SourceProvenanceDiagnostics(summary=summary, report=report, manifest=manifest)


def summarize_source_provenance(
    rows: pd.DataFrame,
    *,
    identical_tolerance_f: float = 0.01,
    duplicate_identical_rate: float = 0.999,
) -> pd.DataFrame:
    """Return one row per city/source pair with overlap and similarity metrics."""
    required = {"city", "target_date", "source", "point_f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"rows missing required columns: {sorted(missing)}")
    clean = rows.loc[:, list(required)].copy()
    clean["city"] = clean["city"].astype(str)
    clean["source"] = clean["source"].astype(str)
    clean["target_date"] = pd.to_datetime(clean["target_date"], errors="coerce")
    clean["point_f"] = pd.to_numeric(clean["point_f"], errors="coerce")
    clean = clean.dropna(subset=["target_date", "point_f"])
    if clean.empty:
        return pd.DataFrame(columns=PROVENANCE_COLUMNS)

    output = []
    for city, group in clean.groupby("city", sort=True):
        pivot = group.pivot_table(
            index="target_date",
            columns="source",
            values="point_f",
            aggfunc="mean",
        )
        for source_a, source_b in combinations(sorted(pivot.columns), 2):
            pair = pivot[[source_a, source_b]].dropna()
            if pair.empty:
                continue
            diffs = (pair[source_a] - pair[source_b]).abs()
            identical_rate = float((diffs <= identical_tolerance_f).mean())
            correlation = (
                float(pair[source_a].corr(pair[source_b]))
                if len(pair) > 1 and pair[source_a].std() > 0 and pair[source_b].std() > 0
                else pd.NA
            )
            output.append(
                {
                    "city": city,
                    "source_a": source_a,
                    "source_b": source_b,
                    "n_overlap": int(len(pair)),
                    "mean_abs_diff_f": float(diffs.mean()),
                    "max_abs_diff_f": float(diffs.max()),
                    "correlation": correlation,
                    "identical_rate": identical_rate,
                    "duplicate_flag": bool(identical_rate >= duplicate_identical_rate),
                }
            )
    return pd.DataFrame(output, columns=PROVENANCE_COLUMNS)


def render_source_provenance_report(summary: pd.DataFrame) -> str:
    """Render a compact markdown provenance report."""
    lines = [
        "# Source Provenance Diagnostics",
        "",
        "Weather-source similarity only. No market data.",
        "",
    ]
    if summary.empty:
        return "\n".join([*lines, "No comparable source pairs.", ""])
    duplicates = summary[summary["duplicate_flag"]].sort_values(
        ["city", "source_a", "source_b"]
    )
    if duplicates.empty:
        lines.append("No duplicate source pairs flagged.")
    else:
        lines.append("## Duplicate Source Flags")
        lines.append("")
        lines.append("| city | source A | source B | n | identical rate | max diff |")
        lines.append("|---|---|---|---:|---:|---:|")
        for row in duplicates.itertuples(index=False):
            lines.append(
                f"| {row.city} | {row.source_a} | {row.source_b} | "
                f"{row.n_overlap} | {row.identical_rate:.3f} | {row.max_abs_diff_f:.3f} |"
            )
    lines.extend(
        [
            "",
            "Duplicate flags should be treated as source-independence risk before using source counts, blends, or contrarian diagnostics.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_source_provenance_diagnostics(
    *,
    input_path: Path,
    output_dir: Path,
    git_commit: str | None = None,
) -> SourceProvenanceDiagnostics:
    """Read rows and write source provenance diagnostics."""
    result = build_source_provenance_diagnostics(
        pd.read_csv(input_path),
        input_path=str(input_path),
        input_sha256=_sha256(input_path),
        git_commit=git_commit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output_dir / "source_provenance.csv", index=False)
    (output_dir / "source_provenance_report.md").write_text(
        result.report,
        encoding="utf-8",
    )
    (output_dir / "source_provenance_manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Professional guidance ingestion and no-leak diagnostics.

This module is mainline-safe. It handles weather guidance rows only: no market
prices, order books, private PnL labels, or trading instructions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

GUIDANCE_COLUMNS = [
    "city",
    "source",
    "station_id",
    "market_type",
    "target_date",
    "issue_ts_utc",
    "valid_ts_utc",
    "available_ts_utc",
    "guidance_point_f",
    "guidance_q10_f",
    "guidance_q50_f",
    "guidance_q90_f",
    "actual_high_f",
    "raw_payload_hash",
]
GUIDANCE_LATEST_COLUMNS = [
    *GUIDANCE_COLUMNS,
    "as_of_ts_utc",
]
GUIDANCE_SUMMARY_COLUMNS = [
    "city",
    "source",
    "market_type",
    "n",
    "mae",
    "bias",
    "rmse",
    "q10_q90_coverage",
    "mean_interval_width_f",
    "latest_available_ts_utc",
]


@dataclass(frozen=True)
class GuidanceDiagnostics:
    latest: pd.DataFrame
    summary: pd.DataFrame
    report: str
    manifest: dict[str, Any]


def load_guidance_csv(path: Path) -> pd.DataFrame:
    """Load normalized guidance rows from CSV."""
    rows = pd.read_csv(path)
    return normalize_guidance_rows(rows)


def normalize_guidance_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize weather guidance rows."""
    required = {
        "city",
        "source",
        "station_id",
        "market_type",
        "target_date",
        "issue_ts_utc",
        "valid_ts_utc",
        "available_ts_utc",
        "guidance_point_f",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"guidance rows missing columns: {sorted(missing)}")
    normalized = rows.copy()
    for column in GUIDANCE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized["city"] = normalized["city"].astype(str).str.strip().str.lower()
    normalized["source"] = normalized["source"].astype(str).str.strip().str.lower()
    normalized["station_id"] = normalized["station_id"].astype(str).str.strip().str.upper()
    normalized["market_type"] = normalized["market_type"].astype(str).str.strip().str.lower()
    normalized["target_date"] = pd.to_datetime(
        normalized["target_date"],
        errors="coerce",
    ).dt.date.astype(str)
    for column in ("issue_ts_utc", "valid_ts_utc", "available_ts_utc"):
        normalized[column] = _utc_iso_series(normalized[column])
    for column in (
        "guidance_point_f",
        "guidance_q10_f",
        "guidance_q50_f",
        "guidance_q90_f",
        "actual_high_f",
    ):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(
        subset=[
            "city",
            "source",
            "station_id",
            "market_type",
            "target_date",
            "issue_ts_utc",
            "valid_ts_utc",
            "available_ts_utc",
            "guidance_point_f",
        ]
    )
    return normalized.loc[:, GUIDANCE_COLUMNS].reset_index(drop=True)


def latest_guidance_as_of(
    rows: pd.DataFrame,
    *,
    as_of_ts: datetime | str,
    target_date: str | None = None,
) -> pd.DataFrame:
    """Return the latest available guidance per city/source/market/date as of a timestamp."""
    as_of_iso = _utc_iso(as_of_ts)
    as_of = pd.Timestamp(as_of_iso)
    clean = normalize_guidance_rows(rows)
    clean["_available"] = pd.to_datetime(clean["available_ts_utc"], utc=True)
    clean["_issue"] = pd.to_datetime(clean["issue_ts_utc"], utc=True)
    filtered = clean[clean["_available"] <= as_of].copy()
    if target_date is not None:
        filtered = filtered[filtered["target_date"] == str(target_date)]
    if filtered.empty:
        return pd.DataFrame(columns=GUIDANCE_LATEST_COLUMNS)
    latest = (
        filtered.sort_values(["_available", "_issue"])
        .drop_duplicates(["city", "source", "market_type", "target_date"], keep="last")
        .drop(columns=["_available", "_issue"])
    )
    latest["as_of_ts_utc"] = as_of_iso
    return latest.loc[:, GUIDANCE_LATEST_COLUMNS].sort_values(
        ["city", "source", "market_type", "target_date"]
    )


def summarize_guidance_accuracy(rows: pd.DataFrame) -> pd.DataFrame:
    """Score guidance rows where actual outcomes are present."""
    clean = normalize_guidance_rows(rows)
    clean = clean.dropna(subset=["guidance_point_f", "actual_high_f"])
    if clean.empty:
        return pd.DataFrame(columns=GUIDANCE_SUMMARY_COLUMNS)
    output = []
    for keys, group in clean.groupby(["city", "source", "market_type"], sort=True):
        errors = group["guidance_point_f"].astype(float) - group["actual_high_f"].astype(float)
        abs_errors = errors.abs()
        coverage, width = _interval_metrics(group)
        output.append(
            {
                "city": keys[0],
                "source": keys[1],
                "market_type": keys[2],
                "n": int(len(group)),
                "mae": float(abs_errors.mean()),
                "bias": float(errors.mean()),
                "rmse": float(math.sqrt((errors**2).mean())),
                "q10_q90_coverage": coverage,
                "mean_interval_width_f": width,
                "latest_available_ts_utc": str(group["available_ts_utc"].max()),
            }
        )
    return pd.DataFrame(output, columns=GUIDANCE_SUMMARY_COLUMNS)


def build_guidance_diagnostics(
    rows: pd.DataFrame,
    *,
    as_of_ts: datetime | str,
    target_date: str | None = None,
    input_path: str | None = None,
    input_sha256: str | None = None,
    git_commit: str | None = None,
) -> GuidanceDiagnostics:
    """Build latest-guidance and accuracy diagnostics."""
    latest = latest_guidance_as_of(rows, as_of_ts=as_of_ts, target_date=target_date)
    summary = summarize_guidance_accuracy(rows)
    report = render_guidance_report(latest, summary)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "input_path": input_path,
        "input_sha256": input_sha256,
        "as_of_ts_utc": _utc_iso(as_of_ts),
        "target_date": target_date,
        "row_counts": {
            "input_rows": int(len(rows)),
            "latest_rows": int(len(latest)),
            "summary_rows": int(len(summary)),
        },
        "notes": [
            "Mainline-safe guidance diagnostics. No market data or trading instructions.",
            "Rows are included in latest_guidance only when available_ts_utc <= as_of_ts_utc.",
        ],
    }
    return GuidanceDiagnostics(
        latest=latest,
        summary=summary,
        report=report,
        manifest=manifest,
    )


def render_guidance_report(latest: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Render a compact markdown report."""
    lines = [
        "# Professional Guidance Diagnostics",
        "",
        "Weather guidance only. No market prices, order books, private PnL labels, or trade instructions.",
        "",
    ]
    lines.append("## Latest Available Guidance")
    lines.append("")
    if latest.empty:
        lines.append("No guidance rows were available as of the requested timestamp.")
    else:
        lines.append("| city | source | station | target | point | available |")
        lines.append("|---|---|---|---|---:|---|")
        for row in latest.itertuples(index=False):
            lines.append(
                f"| {row.city} | {row.source} | {row.station_id} | {row.target_date} | "
                f"{row.guidance_point_f:.1f} | {row.available_ts_utc} |"
            )
    lines.append("")
    lines.append("## Accuracy Summary")
    lines.append("")
    if summary.empty:
        lines.append("No settled guidance rows with actuals were available for scoring.")
    else:
        lines.append("| city | source | n | MAE | bias | RMSE | q10-q90 coverage |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.city} | {row.source} | {row.n} | {_fmt(row.mae)} | "
                f"{_fmt(row.bias)} | {_fmt(row.rmse)} | {_fmt(row.q10_q90_coverage)} |"
            )
    return "\n".join(lines) + "\n"


def write_guidance_diagnostics(
    *,
    input_path: Path,
    output_dir: Path,
    as_of_ts: datetime | str,
    target_date: str | None = None,
    git_commit: str | None = None,
) -> GuidanceDiagnostics:
    """Read normalized guidance rows and write diagnostics artifacts."""
    result = build_guidance_diagnostics(
        load_guidance_csv(input_path),
        as_of_ts=as_of_ts,
        target_date=target_date,
        input_path=str(input_path),
        input_sha256=_sha256(input_path),
        git_commit=git_commit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.latest.to_csv(output_dir / "guidance_latest.csv", index=False)
    result.summary.to_csv(output_dir / "guidance_score_summary.csv", index=False)
    (output_dir / "guidance_report.md").write_text(result.report, encoding="utf-8")
    (output_dir / "guidance_manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _interval_metrics(group: pd.DataFrame) -> tuple[object, object]:
    required = group.dropna(subset=["guidance_q10_f", "guidance_q90_f", "actual_high_f"])
    if required.empty:
        return pd.NA, pd.NA
    actual = required["actual_high_f"].astype(float)
    lower = required["guidance_q10_f"].astype(float)
    upper = required["guidance_q90_f"].astype(float)
    coverage = ((lower <= actual) & (actual <= upper)).mean()
    width = (upper - lower).mean()
    return float(coverage), float(width)


def _utc_iso(value: datetime | str) -> str:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value)
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _utc_iso_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.map(lambda value: value.isoformat() if pd.notna(value) else pd.NA)


def _fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

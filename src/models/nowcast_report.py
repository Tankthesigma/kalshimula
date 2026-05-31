"""Weather-only nowcast report from frozen prediction export rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SUMMARY_COLUMNS = [
    "city",
    "platform",
    "market_type",
    "station_id",
    "target_date",
    "decision_time_label",
    "source_policy",
    "point_f",
    "q10_f",
    "q50_f",
    "q90_f",
    "top_bin_label",
    "top_bin_probability",
    "second_bin_label",
    "second_bin_probability",
    "nowcast_veto_flag",
    "weather_reason_codes",
    "station_rule_confidence",
    "source_independence_score",
    "priority",
]


@dataclass(frozen=True)
class NowcastReportResult:
    summary: pd.DataFrame
    markdown: str
    manifest: dict[str, Any]


def build_nowcast_report(
    predictions: pd.DataFrame,
    *,
    input_path: str | None = None,
    input_sha256: str | None = None,
    git_commit: str | None = None,
) -> NowcastReportResult:
    """Summarize frozen prediction rows into a weather-only report."""
    summary = summarize_nowcast_predictions(predictions)
    markdown = render_nowcast_report(summary)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "input_path": input_path,
        "input_sha256": input_sha256,
        "row_counts": {
            "prediction_rows": int(len(predictions)),
            "summary_rows": int(len(summary)),
        },
        "notes": [
            "Weather-only report. No market prices, order books, private PnL labels, or trade instructions.",
            "Priority is model-readiness triage, not a trading signal.",
        ],
    }
    return NowcastReportResult(summary=summary, markdown=markdown, manifest=manifest)


def summarize_nowcast_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one weather-only summary row per city/platform/market/date/time."""
    required = {
        "city",
        "platform",
        "market_type",
        "station_id",
        "target_date",
        "decision_time_label",
        "source_policy",
        "point_f",
        "q10_f",
        "q50_f",
        "q90_f",
        "bin_label",
        "calibrated_probability",
        "nowcast_veto_flag",
        "weather_reason_codes",
        "station_rule_confidence",
        "source_independence_score",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"nowcast predictions missing columns: {sorted(missing)}")

    clean = predictions.copy()
    clean["calibrated_probability"] = pd.to_numeric(
        clean["calibrated_probability"],
        errors="coerce",
    )
    clean["point_f"] = pd.to_numeric(clean["point_f"], errors="coerce")
    clean["q10_f"] = pd.to_numeric(clean["q10_f"], errors="coerce")
    clean["q50_f"] = pd.to_numeric(clean["q50_f"], errors="coerce")
    clean["q90_f"] = pd.to_numeric(clean["q90_f"], errors="coerce")
    clean["source_independence_score"] = pd.to_numeric(
        clean["source_independence_score"],
        errors="coerce",
    )
    rows = []
    group_cols = [
        "city",
        "platform",
        "market_type",
        "station_id",
        "target_date",
        "decision_time_label",
    ]
    for keys, group in clean.groupby(group_cols, sort=True, dropna=False):
        ordered = group.sort_values(
            ["calibrated_probability", "bin_label"],
            ascending=[False, True],
        )
        top = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else None
        base = group.iloc[0]
        veto = bool(base["nowcast_veto_flag"])
        station_confidence = str(base["station_rule_confidence"])
        independence = float(base["source_independence_score"])
        rows.append(
            {
                "city": keys[0],
                "platform": keys[1],
                "market_type": keys[2],
                "station_id": keys[3],
                "target_date": keys[4],
                "decision_time_label": keys[5],
                "source_policy": base["source_policy"],
                "point_f": float(base["point_f"]),
                "q10_f": float(base["q10_f"]),
                "q50_f": float(base["q50_f"]),
                "q90_f": float(base["q90_f"]),
                "top_bin_label": top["bin_label"],
                "top_bin_probability": float(top["calibrated_probability"]),
                "second_bin_label": second["bin_label"] if second is not None else "",
                "second_bin_probability": (
                    float(second["calibrated_probability"]) if second is not None else pd.NA
                ),
                "nowcast_veto_flag": veto,
                "weather_reason_codes": _text(base["weather_reason_codes"]),
                "station_rule_confidence": station_confidence,
                "source_independence_score": independence,
                "priority": _priority(
                    nowcast_veto_flag=veto,
                    weather_reason_codes=_text(base["weather_reason_codes"]),
                    station_rule_confidence=station_confidence,
                    source_independence_score=independence,
                ),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def render_nowcast_report(summary: pd.DataFrame) -> str:
    """Render a compact markdown report for human weather review."""
    lines = [
        "# Weather-Only Nowcast Report",
        "",
        "This is a model-readiness report, not a trading signal. It contains no market prices, order books, private PnL labels, or trade instructions.",
        "",
    ]
    if summary.empty:
        return "\n".join([*lines, "No rows.", ""])

    lines.extend(
        [
            "## City Summary",
            "",
            "| priority | city | station | point | q10-q90 | top bin | second bin | veto | reasons |",
            "|---|---|---|---:|---:|---|---|---|---|",
        ]
    )
    ordered = summary.sort_values(["priority", "city"], key=_priority_sort_key)
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.priority} | {row.city} | {row.station_id} | "
            f"{row.point_f:.1f} | {row.q10_f:.0f}-{row.q90_f:.0f} | "
            f"{row.top_bin_label} ({row.top_bin_probability:.0%}) | "
            f"{row.second_bin_label} ({_pct(row.second_bin_probability)}) | "
            f"{row.nowcast_veto_flag} | {row.weather_reason_codes} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `high` means the station/rule/source plumbing is clean and no weather-only veto fired.",
            "- `review` means the row is usable but should be checked for station confidence, source independence, or weather conditions.",
            "- `veto` means weather-only conditions make the model stale or risky at this decision time.",
            "- Bobby/private audit still decides whether any model disagreement is market-relevant.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_nowcast_report(
    *,
    predictions_path: Path,
    output_dir: Path,
    git_commit: str | None = None,
) -> NowcastReportResult:
    """Read prediction rows and write summary CSV, markdown, and manifest."""
    result = build_nowcast_report(
        pd.read_csv(predictions_path),
        input_path=str(predictions_path),
        input_sha256=_sha256(predictions_path),
        git_commit=git_commit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output_dir / "nowcast_report_summary.csv", index=False)
    (output_dir / "nowcast_report.md").write_text(result.markdown, encoding="utf-8")
    (output_dir / "nowcast_report_manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _priority(
    *,
    nowcast_veto_flag: bool,
    weather_reason_codes: str,
    station_rule_confidence: str,
    source_independence_score: float,
) -> str:
    if nowcast_veto_flag:
        return "veto"
    if "selected_source_fallback" in weather_reason_codes.split(";"):
        return "review"
    if station_rule_confidence != "high" or source_independence_score < 0.5:
        return "review"
    return "high"


def _priority_sort_key(values: pd.Series) -> pd.Series:
    order = {"high": 0, "review": 1, "veto": 2}
    return values.map(order).fillna(99)


def _pct(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return ""


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

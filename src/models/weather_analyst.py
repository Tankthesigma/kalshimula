"""Weather-only analyst packet from structured desk artifacts.

This module is deliberately deterministic. It is the report/risk-flag layer a
local LLM can later summarize, but it does not call an LLM and it does not use
market prices, order books, private PnL labels, or trading instructions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ANALYST_COLUMNS = [
    "city",
    "platform",
    "market_type",
    "station_id",
    "target_date",
    "decision_time_label",
    "source_policy",
    "calibration_supported",
    "desk_priority",
    "point_f",
    "q10_f",
    "q90_f",
    "top_bin_label",
    "top_bin_probability",
    "nowcast_priority",
    "guidance_agreement",
    "model_minus_nws_f",
    "risk_flags",
    "analyst_note",
]


@dataclass(frozen=True)
class WeatherAnalystPacket:
    rows: pd.DataFrame
    markdown: str
    manifest: dict[str, Any]


def build_weather_analyst_packet(
    nowcast_summary: pd.DataFrame,
    *,
    guidance_comparison: pd.DataFrame | None = None,
    calibration_coverage: set[tuple[str, str]] | None = None,
    git_commit: str | None = None,
) -> WeatherAnalystPacket:
    """Build a deterministic weather-desk analyst packet."""
    rows = summarize_weather_analyst_rows(
        nowcast_summary,
        guidance_comparison=guidance_comparison,
        calibration_coverage=calibration_coverage,
    )
    priority_counts = _priority_counts(rows)
    clean_rows = rows.loc[rows["desk_priority"] == "clean"].copy()
    markdown = render_weather_analyst_packet(rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "row_counts": {
            "nowcast_summary_rows": int(len(nowcast_summary)),
            "guidance_comparison_rows": int(
                0 if guidance_comparison is None else len(guidance_comparison)
            ),
            "analyst_rows": int(len(rows)),
            "clean_rows": int(len(clean_rows)),
            "uncalibrated_rows": int(
                rows["risk_flags"].fillna("").str.contains("uncalibrated_source_policy").sum()
            ),
        },
        "priority_counts": priority_counts,
        "clean_cities": clean_rows["city"].tolist(),
        "notes": [
            "Weather-only analyst packet. No market prices, order books, private PnL labels, or trade instructions.",
            "This is deterministic risk triage, not numeric prediction and not a trading signal.",
            "weather_analyst_clean_rows.csv is the only promotable subset for downstream paper pricing; an empty file means no clean rows passed the weather gate.",
        ],
    }
    return WeatherAnalystPacket(rows=rows, markdown=markdown, manifest=manifest)


def summarize_weather_analyst_rows(
    nowcast_summary: pd.DataFrame,
    *,
    guidance_comparison: pd.DataFrame | None = None,
    calibration_coverage: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Return one operator-readable analyst row per nowcast summary row."""
    if nowcast_summary.empty:
        return pd.DataFrame(columns=ANALYST_COLUMNS)
    guidance = _guidance_map(guidance_comparison)
    output = []
    for row in nowcast_summary.to_dict(orient="records"):
        key = _key(row)
        guidance_row = guidance.get(key, {})
        flags = _risk_flags(
            row,
            guidance_row,
            calibration_coverage=calibration_coverage,
        )
        priority = _desk_priority(flags)
        output.append(
            {
                "city": row["city"],
                "platform": row["platform"],
                "market_type": row["market_type"],
                "station_id": row["station_id"],
                "target_date": row["target_date"],
                "decision_time_label": row["decision_time_label"],
                "source_policy": row.get("source_policy", ""),
                "calibration_supported": _calibration_supported(
                    row,
                    calibration_coverage=calibration_coverage,
                ),
                "desk_priority": priority,
                "point_f": _num(row.get("point_f")),
                "q10_f": _num(row.get("q10_f")),
                "q90_f": _num(row.get("q90_f")),
                "top_bin_label": row.get("top_bin_label", ""),
                "top_bin_probability": _num(row.get("top_bin_probability")),
                "nowcast_priority": row.get("priority", ""),
                "guidance_agreement": guidance_row.get("guidance_agreement", "missing"),
                "model_minus_nws_f": _num(guidance_row.get("model_minus_nws_f")),
                "risk_flags": ";".join(flags),
                "analyst_note": _analyst_note(priority, flags),
            }
        )
    return pd.DataFrame(output, columns=ANALYST_COLUMNS).sort_values(
        ["desk_priority", "city"],
        key=_priority_sort_key,
    )


def render_weather_analyst_packet(rows: pd.DataFrame) -> str:
    """Render a compact markdown packet for human weather review."""
    lines = [
        "# Weather Desk Analyst Packet",
        "",
        "Weather-only risk triage. No market prices, order books, private PnL labels, or trade instructions.",
        "",
    ]
    if rows.empty:
        return "\n".join([*lines, "No rows.", ""])
    lines.extend(
        [
            "| priority | city | market | source | calibrated | point | q10-q90 | top bin | NWS delta | flags | note |",
            "|---|---|---|---|---|---:|---:|---|---:|---|---|",
        ]
    )
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.desk_priority} | {row.city} | {row.market_type} | "
            f"{row.source_policy} | {row.calibration_supported} | "
            f"{row.point_f:.1f} | {row.q10_f:.0f}-{row.q90_f:.0f} | "
            f"{row.top_bin_label} ({_pct(row.top_bin_probability)}) | "
            f"{_signed(row.model_minus_nws_f)} | {row.risk_flags} | "
            f"{row.analyst_note} |"
        )
    lines.extend(
        [
            "",
            "Priority meanings:",
            "- `clean`: station/source/nowcast checks are usable and no large NWS divergence was found.",
            "- `review`: weather evidence is usable but has a station, source, confidence, or NWS-divergence flag.",
            "- `veto`: weather-only checks say the row is stale or unsafe for model review at this decision time.",
            "- Machine-readable promotable subset: `weather_analyst_clean_rows.csv`.",
            "",
            "Bobby/private audit decides whether any clean or divergent weather setup has market edge.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_weather_analyst_packet(
    *,
    nowcast_summary_path: Path,
    output_dir: Path,
    guidance_comparison_path: Path | None = None,
    bias_table_path: Path | None = None,
    interval_table_path: Path | None = None,
    git_commit: str | None = None,
) -> WeatherAnalystPacket:
    """Read weather desk artifacts and write analyst CSV/markdown/manifest."""
    nowcast_summary = pd.read_csv(nowcast_summary_path)
    guidance = (
        pd.read_csv(guidance_comparison_path)
        if guidance_comparison_path is not None and guidance_comparison_path.exists()
        else None
    )
    calibration_coverage = _calibration_coverage(
        bias_table_path=bias_table_path,
        interval_table_path=interval_table_path,
    )
    result = build_weather_analyst_packet(
        nowcast_summary,
        guidance_comparison=guidance,
        calibration_coverage=calibration_coverage,
        git_commit=git_commit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.rows.to_csv(output_dir / "weather_analyst_packet.csv", index=False)
    clean_rows = result.rows.loc[result.rows["desk_priority"] == "clean"].copy()
    clean_rows.to_csv(output_dir / "weather_analyst_clean_rows.csv", index=False)
    (output_dir / "weather_analyst_packet.md").write_text(
        result.markdown,
        encoding="utf-8",
    )
    (output_dir / "weather_analyst_manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _guidance_map(
    guidance_comparison: pd.DataFrame | None,
) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    if guidance_comparison is None or guidance_comparison.empty:
        return {}
    return {_key(row): row for row in guidance_comparison.to_dict(orient="records")}


def _key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("city", "")).lower(),
        str(row.get("market_type", "")).lower(),
        str(row.get("station_id", "")).upper(),
        str(row.get("target_date", "")),
        str(row.get("decision_time_label", "")),
    )


def _risk_flags(
    row: dict[str, Any],
    guidance_row: dict[str, Any],
    *,
    calibration_coverage: set[tuple[str, str]] | None = None,
) -> list[str]:
    flags: list[str] = []
    if _bool(row.get("nowcast_veto_flag")) or str(row.get("priority")) == "veto":
        flags.append("weather_veto")
    if str(row.get("station_rule_confidence", "")).lower() != "high":
        flags.append("station_rule_review")
    if _num(row.get("source_independence_score")) < 0.5:
        flags.append("source_not_independent")
    if calibration_coverage is not None:
        city = str(row.get("city", "")).lower()
        source_policy = str(row.get("source_policy", "")).strip()
        if source_policy and (city, source_policy) not in calibration_coverage:
            flags.append("uncalibrated_source_policy")
    if _num(row.get("top_bin_probability")) < 0.35:
        flags.append("diffuse_distribution")
    agreement = str(guidance_row.get("guidance_agreement", "missing"))
    if agreement == "divergent":
        flags.append("nws_divergent")
    elif agreement == "watch":
        flags.append("nws_watch")
    elif agreement == "missing":
        flags.append("nws_missing")
    return flags


def _calibration_supported(
    row: dict[str, Any],
    *,
    calibration_coverage: set[tuple[str, str]] | None = None,
) -> str:
    if calibration_coverage is None:
        return "unknown"
    city = str(row.get("city", "")).lower()
    source_policy = str(row.get("source_policy", "")).strip()
    if not source_policy:
        return "unknown"
    return "yes" if (city, source_policy) in calibration_coverage else "no"


def _desk_priority(flags: list[str]) -> str:
    if "weather_veto" in flags or "nws_divergent" in flags:
        return "veto"
    if flags and set(flags) == {"diffuse_distribution"}:
        return "clean"
    if flags:
        return "review"
    return "clean"


def _analyst_note(priority: str, flags: list[str]) -> str:
    if priority == "veto":
        if "nws_divergent" in flags and "weather_veto" not in flags:
            return "Do not use this row for model review until the model/NWS divergence clears."
        return "Do not use this row for model review until weather veto clears."
    if priority == "clean" and "diffuse_distribution" in flags:
        return "Weather checks are clean, but the distribution is broad; private audit still decides market relevance."
    if "uncalibrated_source_policy" in flags:
        return "Selected source lacks bias/interval calibration coverage; keep this row out of clean promotion."
    if "nws_watch" in flags:
        return "Model differs from NWS by more than 2F; inspect before relying on it."
    if "station_rule_review" in flags:
        return "Station/rule confidence needs validation before promotion."
    if "diffuse_distribution" in flags:
        return "Distribution is broad; avoid overconfident interpretation."
    if "nws_missing" in flags:
        return "No NWS comparison was available in this packet."
    return "Weather checks are clean; private audit still decides market relevance."


def _priority_sort_key(values: pd.Series) -> pd.Series:
    order = {"clean": 0, "review": 1, "veto": 2}
    return values.map(order).fillna(99)


def _priority_counts(rows: pd.DataFrame) -> dict[str, int]:
    order = ("clean", "review", "veto")
    counts = {priority: 0 for priority in order}
    if rows.empty:
        return counts
    for priority, count in rows["desk_priority"].value_counts().items():
        counts[str(priority)] = int(count)
    return counts


def _calibration_coverage(
    *,
    bias_table_path: Path | None,
    interval_table_path: Path | None,
) -> set[tuple[str, str]] | None:
    if (
        bias_table_path is None
        or interval_table_path is None
        or not bias_table_path.exists()
        or not interval_table_path.exists()
    ):
        return None
    bias = pd.read_csv(bias_table_path)
    interval = pd.read_csv(interval_table_path)
    required = {"city", "source"}
    if required - set(bias.columns) or required - set(interval.columns):
        return None
    bias_pairs = {
        (str(row["city"]).lower(), str(row["source"]).strip())
        for _, row in bias.iterrows()
    }
    interval_pairs = {
        (str(row["city"]).lower(), str(row["source"]).strip())
        for _, row in interval.iterrows()
    }
    return bias_pairs & interval_pairs


def _num(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _pct(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.0%}"


def _signed(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):+.1f}"

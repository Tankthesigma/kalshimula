"""Market-free calibration audit for nowcast probability packets.

This module scores weather-only probability packets against observed daily
highs. It deliberately does not read market prices, order books, private PnL
labels, or execution artifacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Station
from src.fetchers import ncei
from src.models.calibration import calibration_table
from src.models.station_rules import (
    DEFAULT_STATION_RULES_PATH,
    StationRule,
    load_station_rules,
    station_table_hash,
)

AUDIT_SCHEMA_VERSION = "1.0"
PREDICTION_GROUP_COLUMNS = [
    "mode",
    "model_version",
    "city",
    "platform",
    "market_type",
    "station_id",
    "target_date",
    "decision_time_label",
    "as_of_ts_utc",
]
SCORED_ROW_COLUMNS = [
    *PREDICTION_GROUP_COLUMNS,
    "actual_high_f",
    "actual_degree_f",
    "actual_source",
    "expected_high_f",
    "bias_f",
    "absolute_error_f",
    "degree_brier",
    "actual_degree_probability",
    "log_loss",
    "mode_degree_f",
    "mode_probability",
    "top1_hit",
    "q10_f",
    "q90_f",
    "q10_q90_contains_actual",
    "support_min_f",
    "support_max_f",
    "support_n",
]
SUMMARY_COLUMNS = [
    "mode",
    "n_scored_groups",
    "n_reliability_events",
    "mean_degree_brier",
    "mean_expected_high_mae_f",
    "mean_expected_high_bias_f",
    "mean_actual_degree_probability",
    "mean_log_loss",
    "top1_hit_rate",
    "q10_q90_coverage",
    "ece",
    "evidence_label",
]
RELIABILITY_COLUMNS = [
    "mode",
    "bucket_start",
    "bucket_end",
    "n",
    "mean_predicted_probability",
    "observed_frequency",
    "absolute_gap",
    "ece_contribution",
]
EXCLUSION_COLUMNS = [
    *PREDICTION_GROUP_COLUMNS,
    "reason",
]


@dataclass(frozen=True)
class CalibrationAuditResult:
    scored_rows: pd.DataFrame
    summary: pd.DataFrame
    reliability: pd.DataFrame
    exclusions: pd.DataFrame
    manifest: dict[str, Any]


def discover_prediction_files(roots: list[Path]) -> list[Path]:
    """Find frozen nowcast prediction CSVs under packet roots."""
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "predictions_nowcast.csv":
            files.append(root)
            continue
        if not root.exists():
            continue
        files.extend(root.rglob("predictions_nowcast_*/predictions_nowcast.csv"))
    return sorted(set(files))


def read_actuals_csv(path: Path) -> pd.DataFrame:
    """Read observed daily highs from a CSV.

    Required columns: city, target_date, actual_high_f.
    Optional columns: actual_source.
    """
    frame = pd.read_csv(path)
    required = {"city", "target_date", "actual_high_f"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"actuals CSV missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["city"] = frame["city"].astype(str).str.strip().str.lower()
    frame["target_date"] = frame["target_date"].astype(str).str.slice(0, 10)
    frame["actual_high_f"] = pd.to_numeric(frame["actual_high_f"], errors="coerce")
    if "actual_source" not in frame.columns:
        frame["actual_source"] = "actuals_csv"
    frame = frame.dropna(subset=["actual_high_f"])
    return frame[["city", "target_date", "actual_high_f", "actual_source"]]


def fetch_ncei_actuals_for_predictions(
    prediction_files: list[Path],
    *,
    station_rules_path: Path = DEFAULT_STATION_RULES_PATH,
) -> pd.DataFrame:
    """Fetch NCEI TMAX for unique city/date pairs present in prediction files."""
    groups = _prediction_groups(prediction_files)
    if groups.empty:
        return pd.DataFrame(columns=["city", "target_date", "actual_high_f", "actual_source"])
    rules = {
        (rule.city, rule.platform, rule.market_type): rule
        for rule in load_station_rules(station_rules_path)
    }
    rows: list[dict[str, Any]] = []
    unique = groups[["city", "platform", "market_type", "target_date"]].drop_duplicates()
    for record in unique.to_dict("records"):
        key = (
            str(record["city"]).strip().lower(),
            str(record["platform"]).strip().lower(),
            str(record["market_type"]).strip().lower(),
        )
        rule = rules.get(key)
        if rule is None or not rule.ghcnd_bare:
            continue
        target = date.fromisoformat(str(record["target_date"])[:10])
        actual = ncei.fetch_daily_high(_station_from_rule(rule), target)
        if actual.high_f is None:
            continue
        rows.append(
            {
                "city": key[0],
                "target_date": target.isoformat(),
                "actual_high_f": actual.high_f,
                "actual_source": f"ncei:{rule.ghcnd_bare}",
            }
        )
    return pd.DataFrame(rows, columns=["city", "target_date", "actual_high_f", "actual_source"])


def build_calibration_audit(
    prediction_files: list[Path],
    *,
    actuals: pd.DataFrame,
    n_buckets: int = 10,
    min_statistical_n: int = 30,
    station_rules_path: Path = DEFAULT_STATION_RULES_PATH,
    git_commit: str | None = None,
) -> CalibrationAuditResult:
    """Score one or more nowcast packet modes against observed highs."""
    actual_map = _actual_map(actuals)
    scored_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    reliability_events: list[dict[str, Any]] = []

    for prediction_file in prediction_files:
        mode = mode_from_prediction_file(prediction_file)
        predictions = pd.read_csv(prediction_file)
        if predictions.empty:
            continue
        for _, group in predictions.groupby(_group_columns_present(predictions), dropna=False):
            first = group.iloc[0].to_dict()
            group_key = _group_key(first, mode)
            actual = actual_map.get((group_key["city"], group_key["target_date"]))
            if actual is None:
                exclusion_rows.append({**group_key, "reason": "missing_actual_high"})
                continue
            pmf = _parse_pmf(first.get("pmf_degree_json"))
            if not pmf:
                exclusion_rows.append({**group_key, "reason": "missing_or_invalid_pmf"})
                continue
            metrics = _score_pmf(
                pmf,
                actual_high_f=actual["actual_high_f"],
                actual_source=str(actual["actual_source"]),
                row=first,
            )
            scored_rows.append({**group_key, **metrics})
            actual_degree = metrics["actual_degree_f"]
            for degree, probability in pmf.items():
                reliability_events.append(
                    {
                        "mode": mode,
                        "probability": probability,
                        "outcome": int(degree == actual_degree),
                    }
                )

    scored = pd.DataFrame(scored_rows, columns=SCORED_ROW_COLUMNS)
    exclusions = pd.DataFrame(exclusion_rows, columns=EXCLUSION_COLUMNS)
    reliability = _build_reliability(reliability_events, n_buckets=n_buckets)
    summary = _build_summary(
        scored,
        reliability,
        min_statistical_n=min_statistical_n,
    )
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "station_table_hash": station_table_hash(station_rules_path),
        "prediction_files": [str(path) for path in prediction_files],
        "n_prediction_files": len(prediction_files),
        "n_scored_groups": int(len(scored)),
        "n_excluded_groups": int(len(exclusions)),
        "n_buckets": n_buckets,
        "min_statistical_n": min_statistical_n,
        "notes": [
            "Market-free calibration audit: no market prices, order books, PnL labels, or trade instructions.",
            "Realized high source should be official NCEI TMAX for production forward scoring.",
            "Small samples are labeled SMOKE / NOT STATISTICAL EVIDENCE.",
        ],
    }
    return CalibrationAuditResult(
        scored_rows=scored,
        summary=summary,
        reliability=reliability,
        exclusions=exclusions,
        manifest=manifest,
    )


def write_calibration_audit(result: CalibrationAuditResult, out_dir: Path) -> None:
    """Write CSV, JSON, and Markdown audit artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result.scored_rows.to_csv(out_dir / "probability_calibration_rows.csv", index=False)
    result.summary.to_csv(out_dir / "probability_calibration_summary.csv", index=False)
    result.reliability.to_csv(out_dir / "probability_calibration_reliability.csv", index=False)
    result.exclusions.to_csv(out_dir / "probability_calibration_exclusions.csv", index=False)
    (out_dir / "probability_calibration_manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "probability_calibration_report.md").write_text(
        _markdown_report(result),
        encoding="utf-8",
    )


def mode_from_prediction_file(path: Path) -> str:
    """Infer mode name from a predictions_nowcast_* packet directory."""
    parent = path.parent.name
    prefix = "predictions_nowcast_"
    if parent.startswith(prefix):
        return parent.removeprefix(prefix)
    return parent


def _prediction_groups(prediction_files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in prediction_files:
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in PREDICTION_GROUP_COLUMNS)
        except ValueError:
            continue
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=PREDICTION_GROUP_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined["city"] = combined["city"].astype(str).str.strip().str.lower()
    combined["platform"] = combined["platform"].astype(str).str.strip().str.lower()
    combined["market_type"] = combined["market_type"].astype(str).str.strip().str.lower()
    combined["target_date"] = combined["target_date"].astype(str).str.slice(0, 10)
    return combined.drop_duplicates()


def _station_from_rule(rule: StationRule) -> Station:
    return Station(
        slug=rule.city,
        name=rule.station_name,
        nws_station=rule.settlement_station,
        ghcnd_id=rule.ghcnd_id,
        lat=0.0,
        lon=0.0,
        tz=rule.timezone,
        lst_offset_hours=rule.lst_offset,
    )


def _actual_map(actuals: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    frame = actuals.copy()
    frame["city"] = frame["city"].astype(str).str.strip().str.lower()
    frame["target_date"] = frame["target_date"].astype(str).str.slice(0, 10)
    frame["actual_high_f"] = pd.to_numeric(frame["actual_high_f"], errors="coerce")
    frame = frame.dropna(subset=["actual_high_f"])
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        result[(str(row["city"]), str(row["target_date"]))] = {
            "actual_high_f": float(row["actual_high_f"]),
            "actual_source": row.get("actual_source") or "actuals",
        }
    return result


def _group_columns_present(predictions: pd.DataFrame) -> list[str]:
    return [column for column in PREDICTION_GROUP_COLUMNS if column != "mode" and column in predictions]


def _group_key(row: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "model_version": str(row.get("model_version") or ""),
        "city": str(row.get("city") or "").strip().lower(),
        "platform": str(row.get("platform") or "").strip().lower(),
        "market_type": str(row.get("market_type") or "").strip().lower(),
        "station_id": str(row.get("station_id") or "").strip().upper(),
        "target_date": str(row.get("target_date") or "")[:10],
        "decision_time_label": str(row.get("decision_time_label") or ""),
        "as_of_ts_utc": str(row.get("as_of_ts_utc") or ""),
    }


def _parse_pmf(value: Any) -> dict[int, float]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    pmf: dict[int, float] = {}
    for degree_raw, probability_raw in payload.items():
        try:
            degree = int(float(degree_raw))
            probability = float(probability_raw)
        except (TypeError, ValueError):
            continue
        if probability > 0 and math.isfinite(probability):
            pmf[degree] = pmf.get(degree, 0.0) + probability
    total = sum(pmf.values())
    if total <= 0:
        return {}
    return {degree: probability / total for degree, probability in sorted(pmf.items())}


def _score_pmf(
    pmf: dict[int, float],
    *,
    actual_high_f: float,
    actual_source: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    actual_degree = _round_half_up(actual_high_f)
    support = set(pmf) | {actual_degree}
    degree_brier = sum(
        (pmf.get(degree, 0.0) - (1.0 if degree == actual_degree else 0.0)) ** 2
        for degree in support
    )
    expected = sum(degree * probability for degree, probability in pmf.items())
    mode_degree, mode_probability = max(pmf.items(), key=lambda item: (item[1], -abs(item[0] - expected)))
    actual_probability = pmf.get(actual_degree, 0.0)
    q10 = _float_or_none(row.get("q10_f"))
    q90 = _float_or_none(row.get("q90_f"))
    return {
        "actual_high_f": actual_high_f,
        "actual_degree_f": actual_degree,
        "actual_source": actual_source,
        "expected_high_f": expected,
        "bias_f": expected - actual_high_f,
        "absolute_error_f": abs(expected - actual_high_f),
        "degree_brier": degree_brier,
        "actual_degree_probability": actual_probability,
        "log_loss": -math.log(max(actual_probability, 1e-12)),
        "mode_degree_f": mode_degree,
        "mode_probability": mode_probability,
        "top1_hit": int(mode_degree == actual_degree),
        "q10_f": q10,
        "q90_f": q90,
        "q10_q90_contains_actual": _contains(q10, q90, actual_high_f),
        "support_min_f": min(pmf),
        "support_max_f": max(pmf),
        "support_n": len(pmf),
    }


def _build_reliability(events: list[dict[str, Any]], *, n_buckets: int) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=RELIABILITY_COLUMNS)
    frame = pd.DataFrame(events)
    rows = []
    for mode, group in frame.groupby("mode"):
        table = calibration_table(
            group["probability"].astype(float).tolist(),
            group["outcome"].astype(bool).tolist(),
            n_buckets=n_buckets,
        )
        total = int(table["n"].sum()) if not table.empty else 0
        for record in table.to_dict("records"):
            gap = abs(
                float(record["observed_frequency"])
                - float(record["mean_predicted_probability"])
            )
            rows.append(
                {
                    "mode": mode,
                    **record,
                    "absolute_gap": gap,
                    "ece_contribution": gap * (int(record["n"]) / total if total else 0.0),
                }
            )
    return pd.DataFrame(rows, columns=RELIABILITY_COLUMNS)


def _build_summary(
    scored: pd.DataFrame,
    reliability: pd.DataFrame,
    *,
    min_statistical_n: int,
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows = []
    reliability_events = (
        reliability.groupby("mode")["n"].sum().to_dict() if not reliability.empty else {}
    )
    ece = (
        reliability.groupby("mode")["ece_contribution"].sum().to_dict()
        if not reliability.empty
        else {}
    )
    for mode, group in scored.groupby("mode"):
        q10_q90 = group["q10_q90_contains_actual"].dropna()
        n = int(len(group))
        rows.append(
            {
                "mode": mode,
                "n_scored_groups": n,
                "n_reliability_events": int(reliability_events.get(mode, 0)),
                "mean_degree_brier": float(group["degree_brier"].mean()),
                "mean_expected_high_mae_f": float(group["absolute_error_f"].mean()),
                "mean_expected_high_bias_f": float(group["bias_f"].mean()),
                "mean_actual_degree_probability": float(group["actual_degree_probability"].mean()),
                "mean_log_loss": float(group["log_loss"].mean()),
                "top1_hit_rate": float(group["top1_hit"].mean()),
                "q10_q90_coverage": float(q10_q90.mean()) if not q10_q90.empty else math.nan,
                "ece": float(ece.get(mode, math.nan)),
                "evidence_label": _evidence_label(n, min_statistical_n),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _evidence_label(n: int, min_statistical_n: int) -> str:
    if n < min_statistical_n:
        return "SMOKE / NOT STATISTICAL EVIDENCE"
    return "WEATHER QUALITY AUDIT / FORWARD VALIDATION"


def _contains(lower: float | None, upper: float | None, value: float) -> int | None:
    if lower is None or upper is None:
        return None
    return int(lower <= value <= upper)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _markdown_report(result: CalibrationAuditResult) -> str:
    lines = [
        "# Nowcast Probability Calibration Audit",
        "",
        "Market-free report. It uses weather packet probabilities and observed daily highs only.",
        "",
        f"- Generated at: {result.manifest['generated_at']}",
        f"- Scored groups: {result.manifest['n_scored_groups']}",
        f"- Excluded groups: {result.manifest['n_excluded_groups']}",
        "",
        "## Summary",
        "",
    ]
    if result.summary.empty:
        lines.append("No prediction groups had both a valid PMF and an observed high.")
    else:
        lines.append(_markdown_table(result.summary))
    lines.extend(["", "## Reliability", ""])
    if result.reliability.empty:
        lines.append("No reliability events were available.")
    else:
        lines.append(_markdown_table(result.reliability))
    if not result.exclusions.empty:
        lines.extend(["", "## Exclusions", ""])
        reason_counts = (
            result.exclusions["reason"]
            .value_counts()
            .rename_axis("reason")
            .reset_index(name="count")
        )
        lines.append(_markdown_table(reason_counts))
    lines.append("")
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    rows = [columns, ["---"] * len(columns)]
    for record in frame.astype(object).where(pd.notna(frame), "").to_dict("records"):
        rows.append([_format_markdown_cell(record[column]) for column in frame.columns])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _format_markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")

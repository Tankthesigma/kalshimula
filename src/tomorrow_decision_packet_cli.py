"""Build a model-only tomorrow decision packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tomorrow_decision_packet",
        description="Write a no-market-data model decision packet.",
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--source-contrarian-summary", required=True, type=Path)
    parser.add_argument("--walkforward-summary", required=True, type=Path)
    parser.add_argument("--policy-leaderboard", required=True, type=Path)
    parser.add_argument("--selected-sources", type=Path)
    parser.add_argument("--climate-summary", type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args(argv)

    rows = build_packet_rows(
        predictions=json.loads(args.predictions.read_text(encoding="utf-8")),
        source_contrarian_summary=pd.read_csv(args.source_contrarian_summary),
        walkforward_summary=pd.read_csv(args.walkforward_summary),
        policy_leaderboard=pd.read_csv(args.policy_leaderboard),
        selected_sources=(
            pd.read_csv(args.selected_sources)
            if args.selected_sources is not None and args.selected_sources.exists()
            else None
        ),
        climate_summary=(
            pd.read_csv(args.climate_summary)
            if args.climate_summary is not None and args.climate_summary.exists()
            else None
        ),
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_csv, index=False)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_packet(rows), encoding="utf-8")
    print(f"Wrote tomorrow model packet: {args.out_md} and {args.out_csv}")
    return 0


def build_packet_rows(
    *,
    predictions: dict,
    source_contrarian_summary: pd.DataFrame,
    walkforward_summary: pd.DataFrame,
    policy_leaderboard: pd.DataFrame,
    selected_sources: pd.DataFrame | None = None,
    climate_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one model-only decision row per prediction city."""
    contrarian = _index(source_contrarian_summary, "city", "source")
    walkforward = _index(walkforward_summary, "city", "source")
    selected = _selected_map(selected_sources)
    climate = _city_index(climate_summary)
    best_policy = (
        str(policy_leaderboard.iloc[0]["source"])
        if not policy_leaderboard.empty and "source" in policy_leaderboard.columns
        else ""
    )
    rows = []
    for prediction in predictions.get("predictions", []):
        city = str(prediction.get("city", ""))
        selected_source = str(prediction.get("selected_source") or selected.get(city, ""))
        calibration = prediction.get("calibration") or {}
        forecast = prediction.get("forecast") or {}
        multi = prediction.get("multi_source") or {}
        multi_calibration = multi.get("calibration") or {}
        corrected = _float_or_none(calibration.get("corrected_point_f"))
        consensus = _float_or_none(multi_calibration.get("corrected_point_f"))
        source_delta = (
            corrected - consensus
            if corrected is not None and consensus is not None
            else None
        )
        contrarian_row = contrarian.get((city, selected_source), {})
        walkforward_row = walkforward.get((city, selected_source), {})
        climate_row = climate.get(city, {})
        promoted = str(contrarian_row.get("promoted", "")).lower() == "true"
        wf_mae = _float_or_none(walkforward_row.get("mae"))
        risk_flags = _risk_flags(
            promoted=promoted,
            wf_mae=wf_mae,
            source_delta=source_delta,
            climate_row=climate_row,
        )
        rows.append(
            {
                "city": city,
                "target_date": prediction.get("target_date") or predictions.get("target_date"),
                "selected_source": selected_source,
                "best_policy": best_policy,
                "forecast_point_f": forecast.get("point_f"),
                "corrected_point_f": corrected,
                "interval_lower_f": calibration.get("interval_lower_f"),
                "interval_upper_f": calibration.get("interval_upper_f"),
                "consensus_corrected_point_f": consensus,
                "source_vs_consensus_delta_f": source_delta,
                "threshold_probability_count": len(prediction.get("threshold_probabilities") or []),
                "contrarian_promoted": promoted,
                "contrarian_correct_rate": contrarian_row.get("contrarian_correct_rate"),
                "contrarian_ci_lower_95": contrarian_row.get("contrarian_correct_ci_lower_95"),
                "mean_abs_delta_f": contrarian_row.get("mean_abs_delta_f"),
                "walkforward_mae": wf_mae,
                "walkforward_brier": walkforward_row.get("brier_raw"),
                "climate_recent_warming_anomaly_f": climate_row.get("mean_recent_warming_anomaly_f"),
                "risk_flags": ";".join(risk_flags) if risk_flags else "none",
                "manual_priority": _priority(promoted=promoted, wf_mae=wf_mae, risk_flags=risk_flags),
                "human_action": "manual paper-check candidate",
            }
        )
    return pd.DataFrame(rows)


def render_packet(rows: pd.DataFrame) -> str:
    """Render markdown packet."""
    lines = [
        "# Tomorrow Model Packet",
        "",
        "Model-only decision support. No Kalshi prices, no Kalshi API, no trade instructions.",
        "",
        "| city | source | corrected | consensus | delta | promoted | WF MAE | priority | flags |",
        "|---|---|---:|---:|---:|---|---:|---|---|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.city} | {row.selected_source} | {_fmt(row.corrected_point_f)} | "
            f"{_fmt(row.consensus_corrected_point_f)} | {_fmt(row.source_vs_consensus_delta_f)} | "
            f"{row.contrarian_promoted} | {_fmt(row.walkforward_mae)} | "
            f"{row.manual_priority} | {row.risk_flags} |"
        )
    lines.extend(
        [
            "",
            "Use `high` and `medium` rows as manual paper-check candidates only. Bobby's private audit must confirm whether any model disagreement maps to market edge.",
        ]
    )
    return "\n".join(lines) + "\n"


def _index(df: pd.DataFrame, *keys: str) -> dict[tuple[str, ...], dict]:
    if df.empty or any(key not in df.columns for key in keys):
        return {}
    return {
        tuple(str(row[key]) for key in keys): row.to_dict()
        for _, row in df.iterrows()
    }


def _city_index(df: pd.DataFrame | None) -> dict[str, dict]:
    if df is None or df.empty or "city" not in df.columns:
        return {}
    return {str(row["city"]): row.to_dict() for _, row in df.iterrows()}


def _selected_map(df: pd.DataFrame | None) -> dict[str, str]:
    if df is None or df.empty or not {"city", "selected_source"}.issubset(df.columns):
        return {}
    return {str(row["city"]): str(row["selected_source"]) for _, row in df.iterrows()}


def _float_or_none(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _risk_flags(
    *,
    promoted: bool,
    wf_mae: float | None,
    source_delta: float | None,
    climate_row: dict,
) -> list[str]:
    flags = []
    if not promoted:
        flags.append("not_contrarian_promoted")
    if wf_mae is None:
        flags.append("missing_walkforward")
    elif wf_mae > 2.5:
        flags.append("high_walkforward_mae")
    if source_delta is None:
        flags.append("missing_consensus_delta")
    elif abs(source_delta) < 0.5:
        flags.append("low_source_consensus_delta")
    if climate_row and abs(_float_or_none(climate_row.get("mean_recent_warming_anomaly_f")) or 0.0) > 3.0:
        flags.append("large_recent_climate_anomaly")
    return flags


def _priority(*, promoted: bool, wf_mae: float | None, risk_flags: list[str]) -> str:
    if wf_mae is None or "high_walkforward_mae" in risk_flags:
        return "skip"
    if promoted and wf_mae <= 1.5 and "missing_consensus_delta" not in risk_flags:
        return "high"
    if wf_mae <= 2.0:
        return "medium"
    return "low"


def _fmt(value: object) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return ""
    return f"{numeric:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())

"""Candidate correction for GFS lone-outlier weather packets.

This module is mainline-safe: it consumes only weather model output and public
weather guidance. It does not use market prices, private PnL labels, or trading
instructions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.guidance import load_guidance_csv
from src.models.nowcast_predictions import NOWCAST_PREDICTION_COLUMNS, QUANTILES

LONE_OUTLIER_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LoneOutlierCorrectionResult:
    predictions: pd.DataFrame
    corrections: pd.DataFrame
    manifest: dict[str, Any]


def write_lone_outlier_correction(
    *,
    predictions_path: Path,
    prediction_json_path: Path,
    guidance_path: Path,
    output_dir: Path,
    threshold_f: float = 3.0,
    blend_weight: float = 0.5,
    git_commit: str | None = None,
) -> LoneOutlierCorrectionResult:
    """Write candidate corrected nowcast rows and a correction audit table."""
    predictions = pd.read_csv(predictions_path)
    corrected, corrections = apply_lone_outlier_correction(
        predictions,
        prediction_payload=json.loads(prediction_json_path.read_text(encoding="utf-8")),
        guidance_rows=load_guidance_csv(guidance_path),
        threshold_f=threshold_f,
        blend_weight=blend_weight,
    )
    manifest = {
        "schema_version": LONE_OUTLIER_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "threshold_f": threshold_f,
        "blend_weight": blend_weight,
        "row_count": int(len(corrected)),
        "correction_count": int(len(corrections)),
        "notes": [
            "Weather-only candidate correction. No market prices, order books, private PnL labels, or trade instructions.",
            "Correction fires only when gfs_ens is a same-side 3F+ lone outlier versus both NWS and multi-source consensus.",
            "This is not a promoted default; Bobby/private audit must validate paper PnL before operational use.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_dir / "predictions_nowcast.csv", index=False)
    corrections.to_csv(output_dir / "lone_outlier_corrections.csv", index=False)
    (output_dir / "lone_outlier_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return LoneOutlierCorrectionResult(
        predictions=corrected,
        corrections=corrections,
        manifest=manifest,
    )


def apply_lone_outlier_correction(
    predictions: pd.DataFrame,
    *,
    prediction_payload: dict[str, Any],
    guidance_rows: pd.DataFrame,
    threshold_f: float = 3.0,
    blend_weight: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a candidate corrected frozen-schema packet and correction audit rows."""
    if predictions.empty:
        return predictions.copy(), _empty_corrections()
    if not 0 <= blend_weight <= 1:
        raise ValueError("blend_weight must be between 0 and 1")
    consensus = _consensus_points(prediction_payload)
    guidance = _guidance_records(guidance_rows)
    corrected_rows: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    group_cols = [
        "city",
        "platform",
        "market_type",
        "station_id",
        "target_date",
        "decision_time_label",
    ]
    for keys, group in predictions.groupby(group_cols, sort=False):
        group = group.sort_values("bin_lower_f")
        key = _GroupKey(*[str(value) for value in keys])
        source_policy = str(group.iloc[0].get("source_policy") or "")
        original_point = _float_or_none(group.iloc[0].get("point_f"))
        consensus_point = consensus.get((key.city, key.market_type, key.target_date))
        as_of_ts = _parse_utc(group.iloc[0].get("as_of_ts_utc"))
        nws_point = _latest_guidance_point(
            guidance,
            city=key.city,
            market_type=key.market_type,
            station_id=key.station_id,
            target_date=key.target_date,
            as_of_ts=as_of_ts,
        )
        decision = _correction_decision(
            source_policy=source_policy,
            original_point=original_point,
            consensus_point=consensus_point,
            nws_point=nws_point,
            threshold_f=threshold_f,
            blend_weight=blend_weight,
        )
        if decision is None:
            corrected_rows.extend(group.to_dict(orient="records"))
            continue
        rows = _correct_group(group, decision)
        corrected_rows.extend(rows)
        corrections.append(
            {
                "city": key.city,
                "platform": key.platform,
                "market_type": key.market_type,
                "station_id": key.station_id,
                "target_date": key.target_date,
                "decision_time_label": key.decision_time_label,
                **decision,
            }
        )
    corrected = pd.DataFrame(corrected_rows, columns=NOWCAST_PREDICTION_COLUMNS)
    return corrected.reset_index(drop=True), pd.DataFrame(corrections, columns=_correction_columns())


@dataclass(frozen=True)
class _GroupKey:
    city: str
    platform: str
    market_type: str
    station_id: str
    target_date: str
    decision_time_label: str


def _correction_decision(
    *,
    source_policy: str,
    original_point: float | None,
    consensus_point: float | None,
    nws_point: float | None,
    threshold_f: float,
    blend_weight: float,
) -> dict[str, Any] | None:
    if source_policy != "gfs_ens":
        return None
    if original_point is None or consensus_point is None or nws_point is None:
        return None
    gfs_minus_consensus = original_point - consensus_point
    gfs_minus_nws = original_point - nws_point
    if abs(gfs_minus_consensus) < threshold_f or abs(gfs_minus_nws) < threshold_f:
        return None
    if _sign(gfs_minus_consensus) != _sign(gfs_minus_nws):
        return None
    target_point = (consensus_point + nws_point) / 2.0
    corrected_point = original_point + blend_weight * (target_point - original_point)
    return {
        "correction_reason": "gfs_lone_outlier_vs_nws_consensus",
        "original_point_f": original_point,
        "consensus_point_f": consensus_point,
        "nws_guidance_point_f": nws_point,
        "target_point_f": target_point,
        "corrected_point_f": corrected_point,
        "delta_f": corrected_point - original_point,
        "gfs_minus_consensus_f": gfs_minus_consensus,
        "gfs_minus_nws_f": gfs_minus_nws,
    }


def _correct_group(group: pd.DataFrame, decision: dict[str, Any]) -> list[dict[str, Any]]:
    original = _pmf_from_group(group)
    corrected = _shift_pmf_fractional(original, float(decision["delta_f"]))
    quantiles = _quantiles(corrected)
    point = float(decision["corrected_point_f"])
    pmf_json = json.dumps(
        {str(degree): probability for degree, probability in sorted(corrected.items())},
        sort_keys=True,
    )
    base = group.iloc[0].to_dict()
    model_probability = {
        int(float(row.bin_lower_f)): float(row.model_probability)
        for row in group.itertuples(index=False)
        if pd.notna(row.bin_lower_f)
    }
    reasons = _append_reason(
        base.get("weather_reason_codes"),
        (
            "lone_outlier_corrected:"
            f"delta={float(decision['delta_f']):+.2f},"
            f"target={float(decision['target_point_f']):.1f}"
        ),
    )
    rows = []
    for degree, probability in sorted(corrected.items()):
        row = dict(base)
        row.update(
            {
                "model_version": f"{base.get('model_version', 'unknown')}-lone-outlier-candidate",
                "bin_lower_f": degree,
                "bin_upper_f": degree,
                "bin_label": str(degree),
                "model_probability": model_probability.get(degree, 0.0),
                "calibrated_probability": probability,
                "point_f": point,
                **{f"q{q:02d}_f": quantiles[q] for q in QUANTILES},
                "pmf_degree_json": pmf_json,
                "weather_reason_codes": reasons,
            }
        )
        rows.append(row)
    return rows


def _consensus_points(payload: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    output = {}
    for prediction in payload.get("predictions", []):
        city = str(prediction.get("city") or "").strip().lower()
        target_date = str(prediction.get("target_date") or "")
        multi_source = prediction.get("multi_source") or {}
        calibration = multi_source.get("calibration") or {}
        forecast = multi_source.get("forecast") or {}
        point = _float_or_none(calibration.get("corrected_point_f"))
        if point is None:
            point = _float_or_none(forecast.get("point_f"))
        if city and target_date and point is not None:
            output[(city, "high", target_date)] = point
    return output


def _guidance_records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if rows.empty:
        return output
    clean = rows.sort_values("available_ts_utc")
    for row in clean.to_dict(orient="records"):
        point = _float_or_none(row.get("guidance_point_f"))
        if point is None:
            continue
        available = _parse_utc(row.get("available_ts_utc"))
        if available is None:
            continue
        output.append(
            {
                "city": str(row.get("city") or "").strip().lower(),
                "market_type": str(row.get("market_type") or "").strip().lower(),
                "station_id": str(row.get("station_id") or "").strip().upper(),
                "target_date": str(row.get("target_date") or ""),
                "available_ts": available,
                "point": point,
            }
        )
    return output


def _latest_guidance_point(
    rows: list[dict[str, Any]],
    *,
    city: str,
    market_type: str,
    station_id: str,
    target_date: str,
    as_of_ts: datetime | None,
) -> float | None:
    if as_of_ts is None:
        return None
    candidates = [
        row
        for row in rows
        if row["city"] == city
        and row["market_type"] == market_type
        and row["station_id"] == station_id
        and row["target_date"] == target_date
        and row["available_ts"] <= as_of_ts
    ]
    if not candidates:
        return None
    return float(max(candidates, key=lambda row: row["available_ts"])["point"])


def _pmf_from_group(group: pd.DataFrame) -> dict[int, float]:
    pmf: dict[int, float] = {}
    for row in group.itertuples(index=False):
        if pd.isna(row.bin_lower_f):
            continue
        degree = int(float(row.bin_lower_f))
        probability = _float_or_none(row.calibrated_probability)
        if probability is None or probability <= 0:
            continue
        pmf[degree] = pmf.get(degree, 0.0) + probability
    return _normalize(pmf)


def _shift_pmf_fractional(pmf: dict[int, float], delta_f: float) -> dict[int, float]:
    shifted: dict[int, float] = {}
    for degree, probability in pmf.items():
        target = degree + delta_f
        lower = math.floor(target)
        upper = math.ceil(target)
        if lower == upper:
            shifted[int(lower)] = shifted.get(int(lower), 0.0) + probability
            continue
        upper_weight = target - lower
        lower_weight = 1.0 - upper_weight
        shifted[int(lower)] = shifted.get(int(lower), 0.0) + probability * lower_weight
        shifted[int(upper)] = shifted.get(int(upper), 0.0) + probability * upper_weight
    return _normalize(shifted)


def _normalize(pmf: dict[int, float]) -> dict[int, float]:
    total = sum(pmf.values())
    if total <= 0:
        return {}
    return {degree: probability / total for degree, probability in sorted(pmf.items())}


def _quantiles(pmf: dict[int, float]) -> dict[int, float]:
    if not pmf:
        return {q: math.nan for q in QUANTILES}
    rows = sorted(pmf.items())
    output: dict[int, float] = {}
    for q in QUANTILES:
        cutoff = q / 100
        cumulative = 0.0
        selected = rows[-1][0]
        for degree, probability in rows:
            cumulative += probability
            if cumulative >= cutoff:
                selected = degree
                break
        output[q] = float(selected)
    return output


def _append_reason(existing: object, reason: str) -> str:
    text = "" if existing is None or pd.isna(existing) else str(existing)
    return reason if not text else f"{text};{reason}"


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_utc(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _empty_corrections() -> pd.DataFrame:
    return pd.DataFrame(columns=_correction_columns())


def _correction_columns() -> list[str]:
    return [
        "city",
        "platform",
        "market_type",
        "station_id",
        "target_date",
        "decision_time_label",
        "correction_reason",
        "original_point_f",
        "consensus_point_f",
        "nws_guidance_point_f",
        "target_point_f",
        "corrected_point_f",
        "delta_f",
        "gfs_minus_consensus_f",
        "gfs_minus_nws_f",
    ]

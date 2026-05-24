"""Candidate heat-regime correction for weather desk packets.

This module is mainline-safe. It uses only weather-model rows and a fixed
residual-bias table derived from historical weather rows. It does not use
market prices, order books, private PnL labels, or trading instructions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.nowcast_predictions import NOWCAST_PREDICTION_COLUMNS, QUANTILES

HEAT_REGIME_SCHEMA_VERSION = "1.0"
HEAT_CORRECTION_COLUMNS = [
    "city",
    "platform",
    "market_type",
    "station_id",
    "target_date",
    "decision_time_label",
    "correction_reason",
    "warm_threshold_f",
    "original_point_f",
    "correction_f",
    "corrected_point_f",
]

# Fixed candidate table from May 2024-Apr 2026 gfs_ens hot-day residuals.
# Positive means historical actual highs ran hotter than the model; negative
# means the model ran too hot in that city's hot regime.
DEFAULT_HEAT_REGIME_CORRECTIONS: dict[str, tuple[float, float]] = {
    "austin": (89.0, 1.4),
    "boston": (75.0, -0.9),
    "chicago": (80.0, 1.5),
    "denver": (84.0, 2.0),
    "houston": (88.0, 1.9),
    "la": (73.0, 0.5),
    "miami": (89.0, 0.8),
    "nyc": (80.0, -1.1),
    "philadelphia": (82.0, 1.2),
    "phoenix": (100.0, 1.9),
}


@dataclass(frozen=True)
class HeatRegimeCorrectionResult:
    predictions: pd.DataFrame
    corrections: pd.DataFrame
    manifest: dict[str, Any]


def write_heat_regime_correction(
    *,
    predictions_path: Path,
    output_dir: Path,
    git_commit: str | None = None,
    correction_table: dict[str, tuple[float, float]] | None = None,
) -> HeatRegimeCorrectionResult:
    """Write heat-corrected candidate packet plus correction audit rows."""
    predictions = pd.read_csv(predictions_path)
    corrected, corrections = apply_heat_regime_correction(
        predictions,
        correction_table=correction_table,
    )
    manifest = {
        "schema_version": HEAT_REGIME_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "row_count": int(len(corrected)),
        "correction_count": int(len(corrections)),
        "notes": [
            "Weather-only candidate correction. No market prices, order books, private PnL labels, or trade instructions.",
            "Correction fires when the city point forecast is in that city's historical hot regime.",
            "This is not a promoted default; Bobby/private audit must validate paper PnL before operational use.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_dir / "predictions_nowcast.csv", index=False)
    corrections.to_csv(output_dir / "heat_corrections.csv", index=False)
    (output_dir / "heat_regime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return HeatRegimeCorrectionResult(
        predictions=corrected,
        corrections=corrections,
        manifest=manifest,
    )


def apply_heat_regime_correction(
    predictions: pd.DataFrame,
    *,
    correction_table: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a candidate heat-corrected frozen-schema packet and audit rows."""
    if predictions.empty:
        return predictions.copy(), pd.DataFrame(columns=HEAT_CORRECTION_COLUMNS)
    table = correction_table or DEFAULT_HEAT_REGIME_CORRECTIONS
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
        city = str(keys[0])
        market_type = str(keys[2])
        original_point = _float_or_none(group.iloc[0].get("point_f"))
        decision = _correction_decision(
            city=city,
            market_type=market_type,
            original_point=original_point,
            correction_table=table,
        )
        if decision is None:
            corrected_rows.extend(group.to_dict(orient="records"))
            continue
        rows = _correct_group(group, decision)
        corrected_rows.extend(rows)
        corrections.append(
            {
                "city": city,
                "platform": str(keys[1]),
                "market_type": market_type,
                "station_id": str(keys[3]),
                "target_date": str(keys[4]),
                "decision_time_label": str(keys[5]),
                **decision,
            }
        )
    corrected = pd.DataFrame(corrected_rows, columns=NOWCAST_PREDICTION_COLUMNS)
    return (
        corrected.reset_index(drop=True),
        pd.DataFrame(corrections, columns=HEAT_CORRECTION_COLUMNS),
    )


def _correction_decision(
    *,
    city: str,
    market_type: str,
    original_point: float | None,
    correction_table: dict[str, tuple[float, float]],
) -> dict[str, Any] | None:
    if market_type != "high" or original_point is None:
        return None
    threshold_and_correction = correction_table.get(city)
    if threshold_and_correction is None:
        return None
    warm_threshold, correction = threshold_and_correction
    if original_point < warm_threshold or abs(correction) < 1e-9:
        return None
    return {
        "correction_reason": "city_hot_regime_residual_bias",
        "warm_threshold_f": float(warm_threshold),
        "original_point_f": original_point,
        "correction_f": float(correction),
        "corrected_point_f": original_point + float(correction),
    }


def _correct_group(group: pd.DataFrame, decision: dict[str, Any]) -> list[dict[str, Any]]:
    original = _pmf_from_group(group)
    correction = float(decision["correction_f"])
    corrected = _shift_pmf_fractional(original, correction)
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
            "heat_regime_corrected:"
            f"threshold={float(decision['warm_threshold_f']):.1f},"
            f"delta={correction:+.1f}"
        ),
    )
    rows = []
    for degree, probability in sorted(corrected.items()):
        row = dict(base)
        row.update(
            {
                "model_version": f"{base.get('model_version', 'unknown')}-heat-candidate",
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

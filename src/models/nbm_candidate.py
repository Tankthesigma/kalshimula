"""Build an NBM candidate packet in the frozen nowcast prediction schema.

This module is mainline-safe. It consumes public weather guidance rows and a
weather-only raw packet as a schema/template source. It does not use market
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

from src.models.guidance import latest_guidance_as_of, load_guidance_csv
from src.models.nowcast_predictions import NOWCAST_PREDICTION_COLUMNS, QUANTILES

NBM_CANDIDATE_SCHEMA_VERSION = "1.0"
NBM_CANDIDATE_MODEL_VERSION = "nbm-text-candidate-v1"


@dataclass(frozen=True)
class NbmCandidateResult:
    predictions: pd.DataFrame
    manifest: dict[str, Any]


def write_nbm_candidate_predictions(
    *,
    raw_predictions_path: Path,
    guidance_path: Path,
    output_dir: Path,
    as_of_ts: datetime | str,
    git_commit: str | None = None,
    model_version: str = NBM_CANDIDATE_MODEL_VERSION,
) -> NbmCandidateResult:
    """Write ``predictions_nowcast_nbm/predictions_nowcast.csv``."""
    raw = pd.read_csv(raw_predictions_path, dtype={"decision_time_label": "string"})
    guidance = load_guidance_csv(guidance_path)
    nbm = latest_guidance_as_of(guidance, as_of_ts=as_of_ts)
    predictions = build_nbm_candidate_predictions(
        raw_predictions=raw,
        latest_guidance=nbm,
        model_version=model_version,
    )
    manifest = {
        "schema_version": NBM_CANDIDATE_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "model_version": model_version,
        "row_count": int(len(predictions)),
        "input_rows": {
            "raw_predictions": int(len(raw)),
            "latest_nbm_guidance": int(len(nbm)),
        },
        "notes": [
            "Weather-only NBM candidate packet. No market prices, order books, private PnL labels, or trade instructions.",
            "NBM is emitted as a candidate mode only; raw remains the default until private forward scoring proves after-cost market edge.",
            "PMF is a rounded-degree normal approximation from NBM q10/q50/q90 when present, otherwise from deterministic NBM point with conservative spread.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions_nowcast.csv", index=False)
    (output_dir / "predictions_nowcast_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NbmCandidateResult(predictions=predictions, manifest=manifest)


def build_nbm_candidate_predictions(
    *,
    raw_predictions: pd.DataFrame,
    latest_guidance: pd.DataFrame,
    model_version: str = NBM_CANDIDATE_MODEL_VERSION,
) -> pd.DataFrame:
    """Replace raw packet PMFs with NBM-derived candidate PMFs."""
    if raw_predictions.empty or latest_guidance.empty:
        return pd.DataFrame(columns=NOWCAST_PREDICTION_COLUMNS)
    guidance_map = _guidance_map(latest_guidance)
    rows: list[dict[str, Any]] = []
    group_cols = [
        "city",
        "platform",
        "market_type",
        "station_id",
        "target_date",
        "decision_time_label",
    ]
    for keys, group in raw_predictions.groupby(group_cols, sort=False, dropna=False):
        city, _platform, market_type, station_id, target_date, _decision = (
            str(value) for value in keys
        )
        guidance = guidance_map.get(
            (
                city.strip().lower(),
                market_type.strip().lower(),
                station_id.strip().upper(),
                target_date,
            )
        )
        if guidance is None:
            continue
        pmf = _pmf_from_guidance(guidance)
        if not pmf:
            continue
        rows.extend(_rows_from_pmf(group.iloc[0].to_dict(), pmf, model_version=model_version))
    return pd.DataFrame(rows, columns=NOWCAST_PREDICTION_COLUMNS)


def _guidance_map(rows: pd.DataFrame) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows.to_dict(orient="records"):
        key = (
            str(row["city"]).strip().lower(),
            str(row["market_type"]).strip().lower(),
            str(row["station_id"]).strip().upper(),
            str(row["target_date"]),
        )
        output[key] = row
    return output


def _pmf_from_guidance(row: dict[str, Any]) -> dict[int, float]:
    point = _float_or_none(row.get("guidance_q50_f"))
    if point is None:
        point = _float_or_none(row.get("guidance_point_f"))
    if point is None:
        return {}
    q10 = _float_or_none(row.get("guidance_q10_f"))
    q90 = _float_or_none(row.get("guidance_q90_f"))
    sigma = 1.75
    if q10 is not None and q90 is not None and q90 > q10:
        sigma = max(0.75, (q90 - q10) / (2 * 1.2815515655446004))
    radius = max(8, int(math.ceil(4 * sigma)))
    low = int(math.floor(point - radius))
    high = int(math.ceil(point + radius))
    pmf = {
        degree: _normal_cdf((degree + 0.5 - point) / sigma)
        - _normal_cdf((degree - 0.5 - point) / sigma)
        for degree in range(low, high + 1)
    }
    return _normalize({degree: prob for degree, prob in pmf.items() if prob > 0})


def _rows_from_pmf(
    base: dict[str, Any],
    pmf: dict[int, float],
    *,
    model_version: str,
) -> list[dict[str, Any]]:
    quantiles = _quantiles(pmf)
    point = sum(degree * probability for degree, probability in pmf.items())
    pmf_json = json.dumps(
        {str(degree): probability for degree, probability in sorted(pmf.items())},
        sort_keys=True,
    )
    rows = []
    for degree, probability in sorted(pmf.items()):
        row = dict(base)
        row.update(
            {
                "model_version": model_version,
                "bin_lower_f": degree,
                "bin_upper_f": degree,
                "bin_label": str(degree),
                "model_probability": probability,
                "calibrated_probability": probability,
                "point_f": point,
                **{f"q{q:02d}_f": quantiles[q] for q in QUANTILES},
                "pmf_degree_json": pmf_json,
                "source_policy": "nbm_text",
                "weather_reason_codes": _append_reason(
                    row.get("weather_reason_codes"),
                    "nbm_guidance_candidate",
                ),
                "source_independence_score": 1.0,
            }
        )
        rows.append(row)
    return rows


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


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


def _append_reason(existing: Any, reason: str) -> str:
    text = "" if existing is None or pd.isna(existing) else str(existing).strip()
    if not text:
        return reason
    parts = [part for part in text.split(";") if part]
    if reason not in parts:
        parts.append(reason)
    return ";".join(parts)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

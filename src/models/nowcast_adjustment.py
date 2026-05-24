"""Weather-only PMF adjustment from point-in-time nowcast features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.nowcast_features import NOWCAST_FEATURE_COLUMNS
from src.models.nowcast_predictions import NOWCAST_PREDICTION_COLUMNS, QUANTILES


@dataclass(frozen=True)
class NowcastAdjustmentResult:
    predictions: pd.DataFrame
    manifest: dict[str, Any]


def apply_nowcast_adjustments(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Apply physical weather-only constraints to nowcast prediction rows.

    For high-temperature markets, final high cannot be below the observed
    ``high_so_far_f`` available at prediction time. For low-temperature markets,
    final low cannot be above the observed ``low_so_far_f``. The adjustment
    truncates physically impossible degree mass and renormalizes.
    """
    _validate_predictions(predictions)
    _validate_features(features)
    feature_map = _feature_map(features)
    output: list[dict[str, Any]] = []
    group_cols = [
        "city",
        "platform",
        "market_type",
        "station_id",
        "target_date",
        "decision_time_label",
    ]
    for keys, group in predictions.groupby(group_cols, sort=True, dropna=False):
        feature = feature_map.get(tuple(str(value) for value in keys))
        market_type = str(keys[2]).lower()
        if feature is None or market_type not in {"high", "low"}:
            output.extend(group.to_dict(orient="records"))
            continue
        if market_type == "high":
            high_so_far = _float_or_none(feature.get("high_so_far_f"))
            if high_so_far is None:
                output.extend(group.to_dict(orient="records"))
                continue
            adjusted = _truncate_high_pmf(group, int(math.floor(high_so_far + 0.5)))
        else:
            low_so_far = _float_or_none(feature.get("low_so_far_f"))
            if low_so_far is None:
                output.extend(group.to_dict(orient="records"))
                continue
            adjusted = _truncate_low_pmf(group, int(math.floor(low_so_far + 0.5)))
        output.extend(adjusted)
    return pd.DataFrame(output, columns=NOWCAST_PREDICTION_COLUMNS)


def write_nowcast_adjusted_predictions(
    *,
    predictions_path: Path,
    features_path: Path,
    output_dir: Path,
    git_commit: str | None = None,
) -> NowcastAdjustmentResult:
    """Read prediction/features rows and write adjusted frozen-schema output."""
    predictions = pd.read_csv(predictions_path)
    features = pd.read_csv(features_path)
    adjusted = apply_nowcast_adjustments(predictions, features)
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "input_hashes": {
            "predictions_nowcast": _sha256(predictions_path),
            "nowcast_features": _sha256(features_path),
        },
        "row_count": int(len(adjusted)),
        "adjustment": "truncate high PMF below high_so_far_f and low PMF above low_so_far_f",
        "notes": [
            "Weather-only adjustment. No market prices, order books, private PnL labels, or trading instructions.",
            "calibrated_probability and pmf_degree_json are adjusted; model_probability remains the original diagnostic probability by degree.",
            "Adjusted nowcast is a candidate mode, not a promoted default; private audit must validate it before operational use.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(output_dir / "predictions_nowcast.csv", index=False)
    (output_dir / "predictions_nowcast_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NowcastAdjustmentResult(predictions=adjusted, manifest=manifest)


def _truncate_high_pmf(group: pd.DataFrame, min_degree: int) -> list[dict[str, Any]]:
    original = _pmf_from_group(group)
    adjusted = {degree: prob for degree, prob in original.items() if degree >= min_degree}
    adjusted = {min_degree: 1.0} if not adjusted else _normalize(adjusted)
    quantiles = _quantiles(adjusted)
    point = sum(degree * probability for degree, probability in adjusted.items())
    pmf_json = json.dumps(
        {str(degree): probability for degree, probability in sorted(adjusted.items())},
        sort_keys=True,
    )
    base = group.iloc[0].to_dict()
    original_model_probability = {
        int(float(row.bin_lower_f)): float(row.model_probability)
        for row in group.itertuples(index=False)
        if pd.notna(row.bin_lower_f)
    }
    reasons = _append_reason(
        base.get("weather_reason_codes"),
        f"pmf_truncated_below_high_so_far:{min_degree}",
    )
    rows = []
    for degree, probability in sorted(adjusted.items()):
        row = dict(base)
        row.update(
            {
                "bin_lower_f": degree,
                "bin_upper_f": degree,
                "bin_label": str(degree),
                "model_probability": original_model_probability.get(degree, 0.0),
                "calibrated_probability": probability,
                "point_f": point,
                **{f"q{q:02d}_f": quantiles[q] for q in QUANTILES},
                "pmf_degree_json": pmf_json,
                "weather_reason_codes": reasons,
            }
        )
        rows.append(row)
    return rows


def _truncate_low_pmf(group: pd.DataFrame, max_degree: int) -> list[dict[str, Any]]:
    original = _pmf_from_group(group)
    adjusted = {degree: prob for degree, prob in original.items() if degree <= max_degree}
    adjusted = {max_degree: 1.0} if not adjusted else _normalize(adjusted)
    quantiles = _quantiles(adjusted)
    point = sum(degree * probability for degree, probability in adjusted.items())
    pmf_json = json.dumps(
        {str(degree): probability for degree, probability in sorted(adjusted.items())},
        sort_keys=True,
    )
    base = group.iloc[0].to_dict()
    original_model_probability = {
        int(float(row.bin_lower_f)): float(row.model_probability)
        for row in group.itertuples(index=False)
        if pd.notna(row.bin_lower_f)
    }
    reasons = _append_reason(
        base.get("weather_reason_codes"),
        f"pmf_truncated_above_low_so_far:{max_degree}",
    )
    rows = []
    for degree, probability in sorted(adjusted.items()):
        row = dict(base)
        row.update(
            {
                "bin_lower_f": degree,
                "bin_upper_f": degree,
                "bin_label": str(degree),
                "model_probability": original_model_probability.get(degree, 0.0),
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


def _feature_map(features: pd.DataFrame) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    rows = {}
    for row in features.to_dict(orient="records"):
        key = (
            str(row["city"]),
            str(row["platform"]),
            str(row["market_type"]),
            str(row["station_id"]),
            str(row["target_date"]),
            str(row["decision_time_label"]),
        )
        rows[key] = row
    return rows


def _validate_predictions(predictions: pd.DataFrame) -> None:
    missing = set(NOWCAST_PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"nowcast predictions missing columns: {sorted(missing)}")


def _validate_features(features: pd.DataFrame) -> None:
    missing = set(NOWCAST_FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"nowcast features missing columns: {sorted(missing)}")


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_reason(existing: object, reason: str) -> str:
    if existing is None or pd.isna(existing) or not str(existing).strip():
        return reason
    parts = [part for part in str(existing).split(";") if part]
    if reason not in parts:
        parts.append(reason)
    return ";".join(parts)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

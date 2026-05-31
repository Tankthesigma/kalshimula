"""Build the frozen mainline nowcast prediction export for private audit joins."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.nowcast_features import NOWCAST_FEATURE_COLUMNS
from src.models.station_rules import (
    DEFAULT_STATION_RULES_PATH,
    StationRule,
    load_station_rules,
    station_table_hash,
)

NOWCAST_PREDICTION_SCHEMA_VERSION = "1.0"
NOWCAST_PREDICTION_COLUMNS = [
    "model_version",
    "city",
    "platform",
    "market_type",
    "station_id",
    "target_date",
    "prediction_ts_utc",
    "prediction_time_local",
    "decision_time_label",
    "as_of_ts_utc",
    "bin_lower_f",
    "bin_upper_f",
    "bin_label",
    "model_probability",
    "calibrated_probability",
    "point_f",
    "q05_f",
    "q10_f",
    "q20_f",
    "q25_f",
    "q30_f",
    "q40_f",
    "q50_f",
    "q60_f",
    "q70_f",
    "q75_f",
    "q80_f",
    "q90_f",
    "q95_f",
    "pmf_degree_json",
    "source_policy",
    "nowcast_veto_flag",
    "weather_reason_codes",
    "station_rule_confidence",
    "source_independence_score",
    "feature_hash",
]
QUANTILES = (5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95)


@dataclass(frozen=True)
class NowcastPredictionResult:
    predictions: pd.DataFrame
    manifest: dict[str, Any]


def load_prediction_payload(path: Path) -> dict[str, Any]:
    """Load a batch prediction JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_nowcast_prediction_rows(
    payload: dict[str, Any],
    *,
    features: pd.DataFrame | None = None,
    station_rules: list[StationRule] | None = None,
    station_rules_path: Path = DEFAULT_STATION_RULES_PATH,
    decision_time_label: str,
    as_of_ts_utc: datetime | str | None = None,
    market_type: str = "high",
    model_version: str = "mainline-nowcast-v1",
) -> pd.DataFrame:
    """Convert prediction JSON into the frozen ``predictions_nowcast.csv`` schema."""
    if market_type.strip().lower() != "high":
        raise ValueError(
            "nowcast prediction export currently supports only high-temperature "
            "model packets; low-market rows require a separately trained low "
            "temperature model"
        )
    rules = station_rules or load_station_rules(station_rules_path)
    rule_map = {(rule.city, rule.platform, rule.market_type): rule for rule in rules}
    feature_map = _feature_map(features)
    rows: list[dict[str, Any]] = []

    for prediction in payload.get("predictions", []):
        city = str(prediction.get("city") or "").strip().lower()
        if not city:
            continue
        rule = rule_map.get((city, "kalshi", market_type.strip().lower()))
        if rule is None:
            continue
        feature = feature_map.get((city, "kalshi", rule.market_type, rule.settlement_station))
        rows.extend(
            _rows_for_prediction(
                prediction,
                rule=rule,
                feature=feature,
                decision_time_label=decision_time_label,
                as_of_ts_utc=as_of_ts_utc,
                model_version=model_version,
            )
        )

    return pd.DataFrame(rows, columns=NOWCAST_PREDICTION_COLUMNS)


def write_nowcast_predictions(
    *,
    predictions_json_path: Path,
    output_dir: Path,
    decision_time_label: str,
    nowcast_features_path: Path | None = None,
    station_rules_path: Path = DEFAULT_STATION_RULES_PATH,
    as_of_ts_utc: datetime | str | None = None,
    market_type: str = "high",
    model_version: str = "mainline-nowcast-v1",
    git_commit: str | None = None,
) -> NowcastPredictionResult:
    """Write ``predictions_nowcast.csv`` and its manifest."""
    payload = load_prediction_payload(predictions_json_path)
    features = _read_features(nowcast_features_path)
    predictions = build_nowcast_prediction_rows(
        payload,
        features=features,
        station_rules_path=station_rules_path,
        decision_time_label=decision_time_label,
        as_of_ts_utc=as_of_ts_utc,
        market_type=market_type,
        model_version=model_version,
    )
    manifest = {
        "schema_version": NOWCAST_PREDICTION_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "model_version": model_version,
        "input_hashes": {
            "predictions_json": _sha256(predictions_json_path),
            "nowcast_features": (
                _sha256(nowcast_features_path) if nowcast_features_path is not None else None
            ),
        },
        "station_table_hash": station_table_hash(station_rules_path),
        "prediction_date_range": _prediction_date_range(predictions),
        "decision_time_labels": sorted(predictions["decision_time_label"].dropna().unique()),
        "market_type": market_type,
        "no_leak_max_observation_ts": _max_observation_ts(features),
        "source_independence_summary": {
            "source_independence_score_default": 1.0,
            "note": "Detailed duplicate-source audit is emitted by source_provenance_cli.",
        },
        "row_count": int(len(predictions)),
        "notes": [
            "Mainline-safe export: no market prices, order books, private PnL labels, or trade instructions.",
            "pmf_degree_json is bias-corrected when calibration contains bias_correction_f.",
            "calibrated_probability is the degree-bin probability from the corrected PMF.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions_nowcast.csv", index=False)
    (output_dir / "predictions_nowcast_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NowcastPredictionResult(predictions=predictions, manifest=manifest)


def _rows_for_prediction(
    prediction: dict[str, Any],
    *,
    rule: StationRule,
    feature: dict[str, Any] | None,
    decision_time_label: str,
    as_of_ts_utc: datetime | str | None,
    model_version: str,
) -> list[dict[str, Any]]:
    raw_pmf = _pmf_from_forecast(prediction.get("forecast") or {})
    if not raw_pmf:
        return []
    calibration = prediction.get("calibration") or {}
    bias_correction = _float_or_zero(calibration.get("bias_correction_f"))
    calibrated_pmf = _shift_pmf(raw_pmf, bias_correction)
    quantiles = _quantiles_from_pmf(calibrated_pmf)
    point = _float_or_none(calibration.get("corrected_point_f"))
    if point is None:
        point = _float_or_none((prediction.get("forecast") or {}).get("point_f"))
    generated_at = _normalize_ts(prediction.get("generated_at"))
    target_date = str(prediction.get("target_date") or "")
    feature_values = _feature_values(
        feature,
        fallback_as_of=as_of_ts_utc or generated_at,
        fallback_decision_time_label=decision_time_label,
    )
    pmf_json = json.dumps(
        {str(degree): probability for degree, probability in sorted(calibrated_pmf.items())},
        sort_keys=True,
    )
    source_policy = _source_policy(prediction)
    raw_by_degree = {int(k): float(v) for k, v in raw_pmf.items()}

    rows = []
    for degree, probability in sorted(calibrated_pmf.items()):
        rows.append(
            {
                "model_version": model_version,
                "city": rule.city,
                "platform": rule.platform,
                "market_type": rule.market_type,
                "station_id": rule.settlement_station,
                "target_date": target_date,
                "prediction_ts_utc": generated_at,
                "prediction_time_local": feature_values["prediction_time_local"],
                "decision_time_label": feature_values["decision_time_label"],
                "as_of_ts_utc": feature_values["as_of_ts_utc"],
                "bin_lower_f": degree,
                "bin_upper_f": degree,
                "bin_label": str(degree),
                "model_probability": raw_by_degree.get(degree, 0.0),
                "calibrated_probability": probability,
                "point_f": point,
                **{f"q{q:02d}_f": quantiles[q] for q in QUANTILES},
                "pmf_degree_json": pmf_json,
                "source_policy": source_policy,
                "nowcast_veto_flag": feature_values["nowcast_veto_flag"],
                "weather_reason_codes": feature_values["weather_reason_codes"],
                "station_rule_confidence": rule.rule_confidence,
                "source_independence_score": _source_independence_score(source_policy),
                "feature_hash": feature_values["feature_hash"],
            }
        )
    return rows


def _read_features(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    features = pd.read_csv(path, dtype={"decision_time_label": "string"})
    missing = set(NOWCAST_FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"nowcast features missing columns: {sorted(missing)}")
    return features


def _feature_map(features: pd.DataFrame | None) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if features is None or features.empty:
        return {}
    rows = {}
    for row in features.to_dict(orient="records"):
        key = (
            str(row["city"]).strip().lower(),
            str(row["platform"]).strip().lower(),
            str(row["market_type"]).strip().lower(),
            str(row["station_id"]).strip().upper(),
        )
        rows[key] = row
    return rows


def _feature_values(
    feature: dict[str, Any] | None,
    *,
    fallback_as_of: datetime | str | None,
    fallback_decision_time_label: str,
) -> dict[str, Any]:
    if feature is None:
        as_of = _normalize_ts(fallback_as_of)
        return {
            "prediction_time_local": "",
            "decision_time_label": fallback_decision_time_label,
            "as_of_ts_utc": as_of,
            "nowcast_veto_flag": False,
            "weather_reason_codes": "",
            "feature_hash": "",
        }
    return {
        "prediction_time_local": str(feature.get("prediction_time_local") or ""),
        "decision_time_label": str(feature.get("decision_time_label") or fallback_decision_time_label),
        "as_of_ts_utc": _normalize_ts(feature.get("as_of_ts_utc") or fallback_as_of),
        "nowcast_veto_flag": bool(feature.get("nowcast_veto_flag")),
        "weather_reason_codes": str(feature.get("weather_reason_codes") or ""),
        "feature_hash": str(feature.get("feature_hash") or ""),
    }


def _pmf_from_forecast(forecast: dict[str, Any]) -> dict[int, float]:
    bins = forecast.get("bin_probabilities") or {}
    pmf: dict[int, float] = {}
    for key, value in bins.items():
        probability = _float_or_none(value)
        if probability is None or probability <= 0:
            continue
        pmf[int(float(key))] = pmf.get(int(float(key)), 0.0) + probability
    return _normalize_pmf(pmf)


def _shift_pmf(pmf: dict[int, float], shift_f: float) -> dict[int, float]:
    if abs(shift_f) < 1e-9:
        return _normalize_pmf(pmf)
    shifted: dict[int, float] = {}
    for degree, probability in pmf.items():
        shifted_degree = int(math.floor(float(degree) + shift_f + 0.5))
        shifted[shifted_degree] = shifted.get(shifted_degree, 0.0) + probability
    return _normalize_pmf(shifted)


def _normalize_pmf(pmf: dict[int, float]) -> dict[int, float]:
    total = float(sum(pmf.values()))
    if total <= 0:
        return {}
    return {degree: probability / total for degree, probability in sorted(pmf.items())}


def _quantiles_from_pmf(pmf: dict[int, float]) -> dict[int, float]:
    if not pmf:
        return {q: math.nan for q in QUANTILES}
    rows = sorted(pmf.items())
    quantiles: dict[int, float] = {}
    for q in QUANTILES:
        threshold = q / 100
        cumulative = 0.0
        selected = rows[-1][0]
        for degree, probability in rows:
            cumulative += probability
            if cumulative >= threshold:
                selected = degree
                break
        quantiles[q] = float(selected)
    return quantiles


def _source_policy(prediction: dict[str, Any]) -> str:
    selected = str(prediction.get("selected_source") or "").strip()
    if prediction.get("selected_source_applied") and selected:
        return selected
    return "openmeteo_naive"


def _source_independence_score(source_policy: str) -> float:
    return 0.0 if source_policy == "hrrr" else 1.0


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_zero(value: Any) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _normalize_ts(value: datetime | str | None) -> str:
    if value is None or value == "":
        return ""
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


def _prediction_date_range(predictions: pd.DataFrame) -> dict[str, str | None]:
    if predictions.empty:
        return {"start": None, "end": None}
    dates = predictions["target_date"].dropna().astype(str)
    return {"start": dates.min(), "end": dates.max()}


def _max_observation_ts(features: pd.DataFrame | None) -> str | None:
    if features is None or features.empty or "latest_obs_ts_utc" not in features.columns:
        return None
    values = pd.to_datetime(features["latest_obs_ts_utc"], errors="coerce").dropna()
    if values.empty:
        return None
    return values.max().isoformat()


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()

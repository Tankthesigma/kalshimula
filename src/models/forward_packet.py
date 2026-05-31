"""Build market-free forward packet v2 artifacts from nowcast prediction rows."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.models.station_rules import (
    DEFAULT_STATION_RULES_PATH,
    StationRule,
    load_station_rules,
    station_table_hash,
)

FORWARD_PACKET_SCHEMA_VERSION = "2.0"
PACKET_REQUIRED_COLUMNS = {
    "model_version",
    "city",
    "market_type",
    "station_id",
    "target_date",
    "prediction_ts_utc",
    "decision_time_label",
    "as_of_ts_utc",
    "bin_lower_f",
    "bin_upper_f",
    "model_probability",
    "calibrated_probability",
    "point_f",
    "q10_f",
    "q50_f",
    "q90_f",
    "pmf_degree_json",
    "source_policy",
    "station_rule_confidence",
    "feature_hash",
}
BANNED_PACKET_KEYS = {
    "ticker",
    "price",
    "odds",
    "spread",
    "depth",
    "volume",
    "fee",
    "rebate",
    "pnl",
    "roi",
    "entry_price",
    "executable_price",
    "quote_source",
    "book_snapshot_id",
    "fill_model",
    "side",
}


@dataclass(frozen=True)
class ForwardPacketResult:
    payload: dict[str, Any]
    packets: list[dict[str, Any]]


def build_forward_packet_payload(
    predictions: pd.DataFrame,
    *,
    station_rules: list[StationRule] | None = None,
    station_rules_path: Path = DEFAULT_STATION_RULES_PATH,
    generated_at: datetime | str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Return a schema-v2 market-free packet payload.

    The output is intentionally weather-only. It may be joined by the private
    lane on ``(city, settlement_station, target_date, market_type, as_of_utc)``,
    but it carries no tickers, prices, sides, fills, fees, or PnL labels.
    """
    _validate_prediction_columns(predictions)
    rules = station_rules or load_station_rules(station_rules_path)
    rule_map = {
        (rule.city, rule.market_type, rule.settlement_station): rule
        for rule in rules
        if rule.platform == "kalshi"
    }
    packets = [
        _packet_from_group(group, rule=rule_map.get(_group_rule_key(group)))
        for _keys, group in predictions.groupby(
            ["city", "station_id", "target_date", "market_type", "as_of_ts_utc"],
            sort=True,
            dropna=False,
        )
    ]
    packets = [packet for packet in packets if packet is not None]
    payload = {
        "schema_version": FORWARD_PACKET_SCHEMA_VERSION,
        "generated_at_utc": _normalize_ts(generated_at or datetime.now(UTC)),
        "git_commit": git_commit,
        "station_table_hash": (
            station_table_hash(station_rules_path)
            if station_rules_path.exists()
            else _station_rules_hash(rules)
        ),
        "packet_count": len(packets),
        "packets": packets,
        "notes": [
            "Market-free weather packet. No tickers, prices, order books, fills, fees, PnL, or trade instructions.",
            "Private-lane packet join key: city, settlement_station, target_date, market_type, as_of_utc.",
            "decision_label is an audit label only; as_of_utc is the authoritative time key.",
        ],
    }
    _assert_market_free(payload)
    return payload


def write_forward_packet_payload(
    *,
    predictions_path: Path,
    output_path: Path,
    station_rules_path: Path = DEFAULT_STATION_RULES_PATH,
    generated_at: datetime | str | None = None,
    git_commit: str | None = None,
) -> ForwardPacketResult:
    """Read ``predictions_nowcast.csv`` and write a schema-v2 JSON payload."""
    predictions = pd.read_csv(predictions_path, dtype={"decision_time_label": "string"})
    payload = build_forward_packet_payload(
        predictions,
        station_rules_path=station_rules_path,
        generated_at=generated_at,
        git_commit=git_commit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ForwardPacketResult(payload=payload, packets=payload["packets"])


def _packet_from_group(group: pd.DataFrame, *, rule: StationRule | None) -> dict[str, Any] | None:
    if group.empty or rule is None:
        return None
    first = group.iloc[0].to_dict()
    pmf = _pmf_from_rows(group)
    if not pmf:
        return None
    as_of = _normalize_ts(first.get("as_of_ts_utc"))
    target = str(first.get("target_date") or "")
    lst_start, lst_end = _lst_window(rule, target)
    source_name = str(first.get("source_policy") or "unknown").strip() or "unknown"
    packet = {
        "schema_version": FORWARD_PACKET_SCHEMA_VERSION,
        "packet_id": "",
        "generated_at_utc": _normalize_ts(first.get("prediction_ts_utc")),
        "as_of_utc": as_of,
        "target_date": target,
        "decision_label": str(first.get("decision_time_label") or ""),
        "market_type": rule.market_type,
        "model_version": str(first.get("model_version") or ""),
        "config_version": "",
        "city": rule.city,
        "settlement_station": rule.settlement_station,
        "ghcnd_id": rule.ghcnd_id,
        "station_timezone": rule.timezone,
        "lst_offset_hours": rule.lst_offset,
        "settlement_unit": rule.unit,
        "settlement_rounding": rule.rounding_rule,
        "station_rule_hash": _station_rule_hash(rule),
        "station_rule_confidence": rule.rule_confidence,
        "join_key": {
            "city": rule.city,
            "settlement_station": rule.settlement_station,
            "target_date": target,
            "market_type": rule.market_type,
            "as_of_utc": as_of,
        },
        "sources": [
            {
                "name": source_name,
                "cycle_id": None,
                "available_at_utc": as_of,
                "source_updated_at_utc": as_of,
                "used": True,
                "stale": False,
            }
        ],
        "expected_temp_f": _float_or_none(first.get("point_f")),
        "median_temp_f": _float_or_none(first.get("q50_f")),
        "p10_temp_f": _float_or_none(first.get("q10_f")),
        "p90_temp_f": _float_or_none(first.get("q90_f")),
        "uncertainty_sigma_f": _sigma_from_quantiles(first),
        "tail_low_prob": _tail_low_prob(pmf, first),
        "tail_high_prob": _tail_high_prob(pmf, first),
        "source_disagreement_f": None,
        "nowcast_delta_f": None,
        "bin_probabilities": _bin_probabilities(pmf),
        "lst_window_start_utc": lst_start,
        "lst_window_end_utc": lst_end,
        "dst_active_local": _dst_active(rule, target),
        "lst_civil_mismatch": _lst_civil_mismatch(rule, target),
        "near_lst_boundary": False,
        "prelim_final_vulnerability": "unknown",
        "completeness": {
            "has_station_rule": True,
            "has_cycle_pinned_guidance": source_name in {"nbm_text", "nws"},
            "has_asos_obs_asof": bool(str(first.get("feature_hash") or "")),
            "has_full_bin_pmf": True,
            "has_lst_flags": True,
            "no_future_sources": True,
        },
        "weather_reason_codes": str(first.get("weather_reason_codes") or ""),
        "feature_hash": str(first.get("feature_hash") or ""),
    }
    packet["packet_id"] = _payload_hash({k: v for k, v in packet.items() if k != "packet_id"})
    _assert_market_free(packet)
    return packet


def _validate_prediction_columns(predictions: pd.DataFrame) -> None:
    missing = PACKET_REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns for forward packet v2: {sorted(missing)}")
    banned = BANNED_PACKET_KEYS & {str(column).lower() for column in predictions.columns}
    if banned:
        raise ValueError(
            f"market fields are not allowed in forward packet input: {sorted(banned)}"
        )


def _group_rule_key(group: pd.DataFrame) -> tuple[str, str, str]:
    first = group.iloc[0]
    return (
        str(first["city"]).strip().lower(),
        str(first["market_type"]).strip().lower(),
        str(first["station_id"]).strip().upper(),
    )


def _pmf_from_rows(group: pd.DataFrame) -> dict[int, float]:
    pmf: dict[int, float] = {}
    for row in group.to_dict(orient="records"):
        degree = _float_or_none(row.get("bin_lower_f"))
        probability = _float_or_none(row.get("calibrated_probability"))
        if degree is None or probability is None or probability < 0:
            continue
        pmf[int(round(degree))] = pmf.get(int(round(degree)), 0.0) + probability
    return _normalize_pmf(pmf)


def _normalize_pmf(pmf: dict[int, float]) -> dict[int, float]:
    total = float(sum(pmf.values()))
    if total <= 0:
        return {}
    return {degree: probability / total for degree, probability in sorted(pmf.items())}


def _bin_probabilities(pmf: dict[int, float]) -> list[dict[str, Any]]:
    return [
        {
            "bin_id": f"between_{degree}_{degree}",
            "bin_type": "between",
            "lower_f": degree,
            "upper_f": degree,
            "lower_inclusive": True,
            "upper_inclusive": True,
            "probability": probability,
        }
        for degree, probability in sorted(pmf.items())
    ]


def _lst_window(rule: StationRule, target: str) -> tuple[str, str]:
    target_date = date.fromisoformat(target)
    offset = timezone(timedelta(hours=rule.lst_offset))
    start = datetime.combine(target_date, datetime.min.time(), tzinfo=offset)
    end = start + timedelta(days=1)
    return start.astimezone(UTC).isoformat(), end.astimezone(UTC).isoformat()


def _dst_active(rule: StationRule, target: str) -> bool:
    zone = ZoneInfo(rule.timezone)
    local_noon = datetime.combine(date.fromisoformat(target), datetime.min.time()).replace(
        hour=12,
        tzinfo=zone,
    )
    offset = local_noon.utcoffset()
    if offset is None:
        return False
    return not math.isclose(offset.total_seconds() / 3600, float(rule.lst_offset))


def _lst_civil_mismatch(rule: StationRule, target: str) -> bool:
    return rule.dst_policy == "lst_year_round" and _dst_active(rule, target)


def _sigma_from_quantiles(row: dict[str, Any]) -> float | None:
    q10 = _float_or_none(row.get("q10_f"))
    q90 = _float_or_none(row.get("q90_f"))
    if q10 is None or q90 is None or q90 <= q10:
        return None
    return (q90 - q10) / (2 * 1.2815515655446004)


def _tail_low_prob(pmf: dict[int, float], row: dict[str, Any]) -> float:
    p10 = _float_or_none(row.get("q10_f"))
    if p10 is None:
        return 0.0
    return float(sum(probability for degree, probability in pmf.items() if degree < p10))


def _tail_high_prob(pmf: dict[int, float], row: dict[str, Any]) -> float:
    p90 = _float_or_none(row.get("q90_f"))
    if p90 is None:
        return 0.0
    return float(sum(probability for degree, probability in pmf.items() if degree > p90))


def _station_rule_hash(rule: StationRule) -> str:
    return _payload_hash(
        {
            "city": rule.city,
            "platform": rule.platform,
            "market_type": rule.market_type,
            "settlement_station": rule.settlement_station,
            "ghcnd_id": rule.ghcnd_id,
            "timezone": rule.timezone,
            "lst_offset": rule.lst_offset,
            "unit": rule.unit,
            "rounding_rule": rule.rounding_rule,
            "settlement_source": rule.settlement_source,
            "rule_confidence": rule.rule_confidence,
        }
    )


def _station_rules_hash(rules: list[StationRule]) -> str:
    return _payload_hash([_station_rule_hash(rule) for rule in rules])


def _payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _assert_market_free(payload: Any) -> None:
    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_lower = str(key).lower()
                if key_lower in BANNED_PACKET_KEYS:
                    raise ValueError(f"market field {'.'.join((*path, str(key)))} is not allowed")
                walk(child, (*path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    walk(payload, ())


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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

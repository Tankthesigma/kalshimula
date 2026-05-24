"""Normalize NWS forecast API payloads into guidance rows."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Station, load_stations
from src.fetchers.common import c_to_f, iso_date_prefix_matches, safe_float
from src.fetchers.nws import fetch_forecast_payload
from src.models.guidance import GUIDANCE_COLUMNS, normalize_guidance_rows

NWS_GUIDANCE_SOURCE = "nws_forecast"


def guidance_rows_from_nws_forecast_payload(
    payload: dict[str, Any],
    *,
    city: str,
    station_id: str,
    target: date,
    fetched_at: datetime | str,
    market_type: str = "high",
    source: str = NWS_GUIDANCE_SOURCE,
) -> pd.DataFrame:
    """Return one normalized guidance row from an NWS forecast payload."""
    point = _daily_high_from_payload(payload, target)
    if point is None:
        return pd.DataFrame(columns=GUIDANCE_COLUMNS)
    issue_ts = _issue_timestamp(payload, fallback=fetched_at)
    valid_ts = _valid_timestamp(payload, target, fallback=fetched_at)
    row = {
        "city": city,
        "source": source,
        "station_id": station_id,
        "market_type": market_type,
        "target_date": target.isoformat(),
        "issue_ts_utc": issue_ts,
        "valid_ts_utc": valid_ts,
        "available_ts_utc": issue_ts,
        "guidance_point_f": point,
        "guidance_q10_f": pd.NA,
        "guidance_q50_f": point,
        "guidance_q90_f": pd.NA,
        "actual_high_f": pd.NA,
        "raw_payload_hash": _payload_hash(payload),
    }
    return normalize_guidance_rows(pd.DataFrame([row]))


def fetch_nws_guidance_rows(
    stations: dict[str, Station] | None = None,
    *,
    target: date,
    cities: list[str] | None = None,
    fetched_at: datetime | str | None = None,
) -> pd.DataFrame:
    """Fetch NWS forecast guidance rows for configured cities."""
    station_map = stations or load_stations()
    selected = cities or sorted(station_map)
    fetched = fetched_at or datetime.now(UTC)
    rows = []
    for city in selected:
        station = station_map[city]
        payload, _ = fetch_forecast_payload(station)
        rows.append(
            guidance_rows_from_nws_forecast_payload(
                payload,
                city=city,
                station_id=station.nws_station,
                target=target,
                fetched_at=fetched,
            )
        )
    if not rows:
        return pd.DataFrame(columns=GUIDANCE_COLUMNS)
    return normalize_guidance_rows(pd.concat(rows, ignore_index=True))


def write_nws_guidance_rows(
    *,
    output_path: Path,
    target: date,
    cities: list[str] | None = None,
    fetched_at: datetime | str | None = None,
) -> pd.DataFrame:
    """Fetch and write normalized NWS forecast guidance rows."""
    rows = fetch_nws_guidance_rows(target=target, cities=cities, fetched_at=fetched_at)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)
    return rows


def _daily_high_from_payload(payload: dict[str, Any], target: date) -> float | None:
    properties = payload.get("properties") if isinstance(payload, dict) else None
    periods = properties.get("periods", []) if isinstance(properties, dict) else []
    if not isinstance(periods, list):
        return None
    candidates = []
    for period in periods:
        if not isinstance(period, dict):
            continue
        if not period.get("isDaytime"):
            continue
        if not iso_date_prefix_matches(period.get("startTime"), target):
            continue
        value = safe_float(period.get("temperature"))
        if value is None:
            continue
        unit = period.get("temperatureUnit")
        if unit == "C":
            candidates.append(c_to_f(value))
        elif unit == "F" or unit is None:
            candidates.append(value)
    return max(candidates) if candidates else None


def _issue_timestamp(payload: dict[str, Any], *, fallback: datetime | str) -> str:
    properties = payload.get("properties") if isinstance(payload, dict) else {}
    for key in ("generatedAt", "updateTime", "updated"):
        value = properties.get(key) if isinstance(properties, dict) else None
        if value:
            return _utc_iso(value)
    return _utc_iso(fallback)


def _valid_timestamp(
    payload: dict[str, Any],
    target: date,
    *,
    fallback: datetime | str,
) -> str:
    properties = payload.get("properties") if isinstance(payload, dict) else None
    periods = properties.get("periods", []) if isinstance(properties, dict) else []
    end_times = []
    if isinstance(periods, list):
        for period in periods:
            if (
                isinstance(period, dict)
                and period.get("isDaytime")
                and iso_date_prefix_matches(period.get("startTime"), target)
                and period.get("endTime")
            ):
                end_times.append(_utc_iso(period["endTime"]))
    return max(end_times) if end_times else _utc_iso(fallback)


def _payload_hash(payload: dict[str, Any]) -> str:
    import hashlib

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_iso(value: datetime | str) -> str:
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

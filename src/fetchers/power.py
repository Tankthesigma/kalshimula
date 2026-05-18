"""NASA POWER — historical fallback daily-high parser/fetcher.

POWER returns one row per day keyed by ``YYYYMMDD`` string, e.g.::

    {
      "properties": {
        "parameter": {
          "T2M_MAX": {
            "20250102": 4.7,
            "20250103": 5.2,
            ...
          }
        }
      }
    }

Values are in degrees Celsius. POWER uses ``-999`` (and occasionally
``-9999``) as a fill value for missing data; we treat anything <= -999 as
missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx

from src.fetchers.common import c_to_f, is_missing_value, safe_float

POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_FILL_VALUE = -999.0


@dataclass(frozen=True)
class PowerDailyHigh:
    station: str
    target_date: date
    high_f: float | None
    source: str = "power"


def _power_key(target: date) -> str:
    return target.strftime("%Y%m%d")


def parse_daily_high(
    payload: dict, target: date, station: str
) -> PowerDailyHigh:
    """Pull T2M_MAX for ``target`` from a POWER payload and convert to F."""
    properties = payload.get("properties") if isinstance(payload, dict) else None
    if not isinstance(properties, dict):
        return PowerDailyHigh(station=station, target_date=target, high_f=None)

    parameter = properties.get("parameter")
    if not isinstance(parameter, dict):
        return PowerDailyHigh(station=station, target_date=target, high_f=None)

    t2m_max = parameter.get("T2M_MAX")
    if not isinstance(t2m_max, dict):
        return PowerDailyHigh(station=station, target_date=target, high_f=None)

    raw = t2m_max.get(_power_key(target))
    if is_missing_value(raw):
        return PowerDailyHigh(station=station, target_date=target, high_f=None)
    value = safe_float(raw)
    if value is None or value <= POWER_FILL_VALUE:
        return PowerDailyHigh(station=station, target_date=target, high_f=None)

    return PowerDailyHigh(
        station=station, target_date=target, high_f=c_to_f(value)
    )


def fetch_daily_high(
    lat: float, lon: float, target: date, station: str
) -> PowerDailyHigh:
    """Thin HTTP wrapper around :func:`parse_daily_high`. No API key required."""
    params = {
        "parameters": "T2M_MAX",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": _power_key(target),
        "end": _power_key(target),
        "format": "JSON",
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.get(POWER_DAILY_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    return parse_daily_high(payload, target, station)

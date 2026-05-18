"""National Weather Service official forecast — daily-high parser/fetcher.

The NWS forecast API returns periods of the form::

    {
      "properties": {
        "periods": [
          {
            "name": "Today",
            "startTime": "2026-05-18T06:00:00-04:00",
            "endTime":   "2026-05-18T18:00:00-04:00",
            "isDaytime": true,
            "temperature": 78,
            "temperatureUnit": "F"
          },
          ...
        ]
      }
    }

We pick daytime periods whose ``startTime`` falls on the target calendar date
and return the max temperature in Fahrenheit. The parser is pure; the fetch
wrapper is intentionally thin — gridpoint resolution is left to the integration
layer Tanmay owns. If you do call ``fetch_daily_high_forecast`` with a station
whose ``nws_station`` does not look like a usable forecast URL, the wrapper
raises ``NotImplementedError`` so we never silently hit the wrong endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx

from src.config import Station, nws_user_agent
from src.fetchers.common import (
    c_to_f,
    iso_date_prefix_matches,
    safe_float,
)


@dataclass(frozen=True)
class NwsDailyHighForecast:
    station: str
    target_date: date
    high_f: float | None
    source: str = "nws"


def parse_daily_high_forecast(
    payload: dict, target: date, station: str
) -> NwsDailyHighForecast:
    """Pull the daily high (Fahrenheit) for ``target`` from an NWS payload.

    Returns ``high_f=None`` when no daytime period on ``target`` has a usable
    numeric temperature.
    """
    properties = payload.get("properties") if isinstance(payload, dict) else None
    periods = (
        properties.get("periods", []) if isinstance(properties, dict) else []
    )
    if not isinstance(periods, list):
        return NwsDailyHighForecast(
            station=station, target_date=target, high_f=None
        )

    candidates: list[float] = []
    for period in periods:
        if not isinstance(period, dict):
            continue
        if not period.get("isDaytime"):
            continue
        if not iso_date_prefix_matches(period.get("startTime"), target):
            continue
        temp = safe_float(period.get("temperature"))
        if temp is None:
            continue
        unit = period.get("temperatureUnit")
        if unit == "C":
            candidates.append(c_to_f(temp))
        elif unit == "F" or unit is None:
            candidates.append(temp)
        else:
            # Unknown unit — skip rather than guess.
            continue

    high_f = max(candidates) if candidates else None
    return NwsDailyHighForecast(
        station=station, target_date=target, high_f=high_f
    )


def fetch_daily_high_forecast(
    station: Station, target: date
) -> NwsDailyHighForecast:
    """Thin HTTP wrapper around :func:`parse_daily_high_forecast`.

    ``station.nws_station`` is treated as a full forecast URL when it starts
    with ``http``; otherwise we raise ``NotImplementedError``. Full gridpoint
    resolution (``/points/{lat},{lon}``) belongs in the integration layer.
    """
    url = station.nws_station
    if not isinstance(url, str) or not url.startswith("http"):
        raise NotImplementedError(
            "NWS gridpoint resolution is owned by the integration layer; "
            "station.nws_station must be a full forecast URL for this wrapper."
        )

    headers = {
        "User-Agent": nws_user_agent(),
        "Accept": "application/geo+json",
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()

    return parse_daily_high_forecast(payload, target, station.nws_station)

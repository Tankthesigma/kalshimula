"""National Weather Service official forecast parser/fetcher."""

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

NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"


@dataclass(frozen=True)
class NwsDailyHighForecast:
    station: str
    target_date: date
    high_f: float | None
    source: str = "nws"


def parse_daily_high_forecast(
    payload: dict, target: date, station: str
) -> NwsDailyHighForecast:
    """Pull the daily high (Fahrenheit) for ``target`` from an NWS payload."""
    properties = payload.get("properties") if isinstance(payload, dict) else None
    periods = properties.get("periods", []) if isinstance(properties, dict) else []
    if not isinstance(periods, list):
        return NwsDailyHighForecast(station=station, target_date=target, high_f=None)

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

    high_f = max(candidates) if candidates else None
    return NwsDailyHighForecast(station=station, target_date=target, high_f=high_f)


def forecast_url_from_points_payload(payload: dict) -> str:
    """Extract the forecast URL from an NWS points response."""
    properties = payload.get("properties") if isinstance(payload, dict) else None
    forecast_url = properties.get("forecast") if isinstance(properties, dict) else None
    if not isinstance(forecast_url, str) or not forecast_url.startswith("http"):
        raise ValueError("NWS points response did not include a forecast URL")
    return forecast_url


def resolve_forecast_url(station: Station) -> str:
    """Resolve a station's lat/lon to the NWS forecast URL."""
    headers = _headers()
    url = NWS_POINTS_URL.format(lat=station.lat, lon=station.lon)
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return forecast_url_from_points_payload(response.json())


def fetch_daily_high_forecast(
    station: Station, target: date
) -> NwsDailyHighForecast:
    """Fetch NWS daily high for a station/date, resolving gridpoint URL if needed."""
    url = station.nws_station if station.nws_station.startswith("http") else resolve_forecast_url(station)
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=_headers())
        response.raise_for_status()
        payload = response.json()
    return parse_daily_high_forecast(payload, target, station.nws_station)


def _headers() -> dict[str, str]:
    return {
        "User-Agent": nws_user_agent(),
        "Accept": "application/geo+json",
    }

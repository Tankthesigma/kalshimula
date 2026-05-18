"""NCEI Access Data Service — observed daily TMAX parser/fetcher.

NCEI returns a JSON document of the form::

    {
      "results": [
        {
          "date":     "2025-01-02T00:00:00",
          "datatype": "TMAX",
          "station":  "GHCND:USW00094728",
          "value":    156
        },
        ...
      ]
    }

TMAX is in tenths of degrees Celsius (so ``156`` → 15.6 C → 60.08 F).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx

from src.config import Station
from src.fetchers.common import (
    c_tenths_to_f,
    iso_date_prefix_matches,
    safe_float,
)

NCEI_DATA_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


@dataclass(frozen=True)
class NceiDailyHigh:
    station: str
    target_date: date
    high_f: float | None
    source: str = "ncei"


def parse_daily_high(
    payload: dict, target: date, station: str
) -> NceiDailyHigh:
    """Extract the daily TMAX (Fahrenheit) for ``target`` from a payload."""
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return NceiDailyHigh(station=station, target_date=target, high_f=None)

    candidates: list[float] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        if row.get("datatype") != "TMAX":
            continue
        if not iso_date_prefix_matches(row.get("date"), target):
            continue
        raw = safe_float(row.get("value"))
        if raw is None:
            continue
        candidates.append(c_tenths_to_f(raw))

    high_f = max(candidates) if candidates else None
    return NceiDailyHigh(station=station, target_date=target, high_f=high_f)


def fetch_daily_high(station: Station, target: date) -> NceiDailyHigh:
    """Thin HTTP wrapper around :func:`parse_daily_high`.

    Uses NCEI's token-less Access Data Service endpoint. No API key required.
    """
    params = {
        "dataset": "daily-summaries",
        "stations": station.ghcnd_bare,
        "dataTypes": "TMAX",
        "startDate": target.isoformat(),
        "endDate": target.isoformat(),
        "format": "json",
        "units": "metric",
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.get(NCEI_DATA_URL, params=params)
        response.raise_for_status()
        # The Access Data Service returns a top-level list, not a dict; wrap
        # it so the parser sees a consistent shape.
        body = response.json()

    if isinstance(body, list):
        payload: dict = {"results": body}
    elif isinstance(body, dict):
        payload = body if "results" in body else {"results": [body]}
    else:
        payload = {"results": []}

    return parse_daily_high(payload, target, station.ghcnd_bare)

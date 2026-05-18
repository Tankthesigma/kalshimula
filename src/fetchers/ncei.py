"""NCEI Access Data Service — observed daily TMAX parser/fetcher.

The Access Data Service usually returns either a top-level list of rows or
a dict with a ``results`` key. Each row looks like::

    {
      "date":     "2025-01-02T00:00:00",
      "datatype": "TMAX",
      "station":  "GHCND:USW00094728",
      "value":    156
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


def _coerce_results(payload: object) -> list[dict]:
    """Normalize all the shapes NCEI may hand us into a list of row-dicts.

    Recognized shapes:
    - ``{"results": [...]}`` — the documented JSON-format response.
    - ``[...]`` — a bare top-level list (Access Data Service sometimes does this).
    - ``{"date": ..., "datatype": ..., "value": ...}`` — a single row.
    - anything else — empty list.
    """
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [row for row in results if isinstance(row, dict)]
        if "datatype" in payload or "value" in payload:
            return [payload]
    return []


def parse_daily_high(
    payload: dict | list, target: date, station: str
) -> NceiDailyHigh:
    """Extract the daily TMAX (Fahrenheit) for ``target`` from a payload.

    ``datatype`` matching is case-insensitive. The ``station`` field on each
    row is not required — when present, it is ignored (we trust the caller's
    station tag since the API was queried for one station already).
    """
    rows = _coerce_results(payload)

    candidates: list[float] = []
    for row in rows:
        datatype = row.get("datatype")
        if not isinstance(datatype, str) or datatype.upper() != "TMAX":
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
        body = response.json()

    return parse_daily_high(body, target, station.ghcnd_bare)

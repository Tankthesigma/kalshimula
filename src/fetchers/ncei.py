"""NCEI Access Data Service — observed daily TMAX parser/fetcher.

The Access Data Service returns one of two row shapes:

**Modern (Access Data Service, what we hit with ``units=metric``):**
::

    [{"DATE": "2025-01-01", "STATION": "USW00094728", "TMAX": "10.6"}, ...]

Keys are uppercase. Each requested ``dataType`` becomes its own column (no
``datatype`` field). With ``units=metric`` the value is already Celsius (so
"10.6" is 10.6 °C).

**Legacy (CDO-style v2 API):**
::

    {"results": [
       {"date": "2025-01-02T00:00:00", "datatype": "TMAX",
        "station": "GHCND:USW00094728", "value": 156}
    ]}

Keys are lowercase. ``datatype`` and ``value`` are separate columns. ``value``
is in tenths of degrees Celsius (so ``156`` is 15.6 °C).

The parser handles both shapes — the row-level check decides which path to
use. This keeps backward compatibility with cached payloads and existing
unit tests while making the parser work correctly against the live API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx

from src.config import Station
from src.fetchers.common import (
    c_tenths_to_f,
    c_to_f,
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
    """Normalize the various shapes NCEI may hand us into a list of row-dicts.

    Recognized shapes:
    - ``{"results": [...]}`` — legacy CDO-style JSON-format response.
    - ``[...]`` — bare top-level list (Access Data Service usually returns this).
    - ``{"date": ...}`` or ``{"datatype": ...}`` or ``{"TMAX": ...}`` — a
      single row not wrapped in a container.
    - anything else — empty list.
    """
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [row for row in results if isinstance(row, dict)]
        if "datatype" in payload or "value" in payload or "TMAX" in payload:
            return [payload]
    return []


def _row_high_f(row: dict, target: date) -> float | None:
    """Extract the TMAX in Fahrenheit from one row, or None if it doesn't apply.

    Tries the Access Data Service shape first (uppercase ``TMAX`` field with a
    Celsius value), then falls back to the legacy CDO shape (``datatype`` /
    ``value`` in tenths of Celsius).
    """
    # Modern Access Data Service shape: TMAX is a column, value is Celsius.
    if "TMAX" in row:
        date_str = row.get("DATE") or row.get("date")
        if not iso_date_prefix_matches(date_str, target):
            return None
        raw = safe_float(row.get("TMAX"))
        if raw is None:
            return None
        return c_to_f(raw)

    # Legacy CDO shape: datatype + value column, value in tenths of Celsius.
    datatype = row.get("datatype")
    if not isinstance(datatype, str) or datatype.upper() != "TMAX":
        return None
    if not iso_date_prefix_matches(row.get("date"), target):
        return None
    raw = safe_float(row.get("value"))
    if raw is None:
        return None
    return c_tenths_to_f(raw)


def parse_daily_high(
    payload: dict | list, target: date, station: str
) -> NceiDailyHigh:
    """Extract the daily TMAX (Fahrenheit) for ``target`` from a payload.

    Accepts both the modern Access Data Service shape and the legacy
    CDO-style shape (see module docstring). The ``station`` field on each
    row is not required.
    """
    candidates: list[float] = []
    for row in _coerce_results(payload):
        high_f = _row_high_f(row, target)
        if high_f is not None:
            candidates.append(high_f)

    return NceiDailyHigh(
        station=station,
        target_date=target,
        high_f=max(candidates) if candidates else None,
    )


def fetch_daily_high(station: Station, target: date) -> NceiDailyHigh:
    """Thin HTTP wrapper around :func:`parse_daily_high`.

    Uses NCEI's token-less Access Data Service endpoint with ``units=metric``,
    so TMAX values come back in degrees Celsius (which :func:`parse_daily_high`
    converts directly via :func:`c_to_f`).
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

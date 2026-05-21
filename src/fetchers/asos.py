"""Iowa State ASOS — hourly observation parser and daily-high helper.

The Iowa Environmental Mesonet's ASOS endpoint returns CSV with rows like::

    station,valid,tmpf
    KORD,2025-01-02 13:53,32.0
    KORD,2025-01-02 14:53,33.1

Values are already in Fahrenheit. ``valid`` is local-standard-time-ish; we
keep it as a naive ``datetime`` and let downstream features.py handle LST
re-aggregation. ``M`` is the canonical missing-value marker.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime

import httpx

from src.fetchers.common import normalize_station, safe_float

ASOS_CSV_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
)


@dataclass(frozen=True)
class AsosHourlyObservation:
    station: str
    valid_time: datetime
    temp_f: float | None
    source: str = "asos"


def _parse_valid(value: str) -> datetime | None:
    """Accept the common ISO/ASOS timestamp shapes; return None on anything else.

    Trailing ``Z`` is dropped (IEM occasionally emits it for UTC).
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1]
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _asos_station_key(value: object) -> str:
    """Comparison key for ASOS station-code matching.

    IEM normalises ICAO codes by dropping the leading ``K`` for US stations
    in the CSV output (``KORD`` -> ``ORD``) even though the request parameter
    still uses the full ICAO. We strip both sides for the comparison so the
    canonical config code (``KORD``) matches what comes back in the rows
    (``ORD``).
    """
    base = normalize_station(value)  # type: ignore[arg-type]
    if base.startswith("K") and len(base) == 4:
        return base[1:]
    return base


def parse_asos_csv(text: str, station: str) -> list[AsosHourlyObservation]:
    """Parse an ASOS CSV blob into a list of observations for ``station``.

    Rows whose ``valid`` timestamp can't be parsed are dropped entirely.
    Rows whose ``tmpf`` is missing or non-numeric are kept with ``temp_f=None``
    so downstream code can still see the timestamp coverage.

    Station filtering is case-insensitive and tolerant of the leading-``K``
    stripping that IEM applies to US ICAO codes in CSV output.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    cleaned = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    reader = csv.DictReader(io.StringIO(cleaned))
    if reader.fieldnames is None:
        return []

    has_station_col = "station" in reader.fieldnames
    wanted = _asos_station_key(station)
    out: list[AsosHourlyObservation] = []
    for row in reader:
        if has_station_col and _asos_station_key(row.get("station") or "") != wanted:
            continue
        valid_time = _parse_valid(row.get("valid", ""))
        if valid_time is None:
            continue
        out.append(
            AsosHourlyObservation(
                station=station,
                valid_time=valid_time,
                temp_f=safe_float(row.get("tmpf")),
            )
        )
    return out


def daily_high_from_hourly(
    observations: list[AsosHourlyObservation], target: date
) -> float | None:
    """Max ``temp_f`` across observations whose ``valid_time`` is on ``target``.

    Returns None when no observation on ``target`` has a numeric temperature.
    """
    if not observations:
        return None
    values: list[float] = []
    for obs in observations:
        if obs.valid_time.date() != target:
            continue
        if obs.temp_f is None:
            continue
        values.append(obs.temp_f)
    return max(values) if values else None


def fetch_asos_csv(station: str, target: date) -> str:
    """Fetch one calendar day of ASOS hourly obs as CSV text."""
    params = {
        "station": station,
        "data": "tmpf",
        "year1": target.year,
        "month1": target.month,
        "day1": target.day,
        "year2": target.year,
        "month2": target.month,
        "day2": target.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": 3,
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.get(ASOS_CSV_URL, params=params)
        response.raise_for_status()
        return response.text

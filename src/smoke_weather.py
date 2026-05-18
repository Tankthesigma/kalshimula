"""Manual smoke harness for the weather data sources.

Hits each fetcher per city/date and collects a uniform :class:`SmokeResult`
record so we can see which sources are healthy and which are erroring without
having to invoke each one by hand. Designed for diagnostics, not for the
modeling pipeline — Tanmay's collection layer remains the canonical fetch
path during training/backtest.

The smoke harness is the only place in this lane that calls real fetchers,
and it is meant to be driven by :mod:`src.smoke_weather_cli` outside of
pytest. Tests for this module monkeypatch the per-source ``fetch_*`` calls
so they never touch the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.config import Station, get_station
from src.fetchers import ncei, nws, power
from src.fetchers.common import compact_error

SMOKE_COLUMNS: tuple[str, ...] = (
    "city",
    "target_date",
    "source",
    "ok",
    "high_f",
    "error",
)


@dataclass(frozen=True)
class SmokeResult:
    """One per (city, source) probe. ``ok=False`` iff an exception was caught."""

    city: str
    target_date: date
    source: str
    ok: bool
    high_f: float | None
    error: str | None


def _safe_call(
    *, city: str, target: date, source: str, fn
) -> SmokeResult:
    """Call ``fn``, returning a SmokeResult that captures success or failure."""
    try:
        record = fn()
    except Exception as exc:  # noqa: BLE001 — that's the point of smoke
        return SmokeResult(
            city=city,
            target_date=target,
            source=source,
            ok=False,
            high_f=None,
            error=compact_error(exc),
        )
    high_f = getattr(record, "high_f", None)
    return SmokeResult(
        city=city,
        target_date=target,
        source=source,
        ok=True,
        high_f=high_f,
        error=None,
    )


def smoke_city(city: str, target: date) -> list[SmokeResult]:
    """Probe NWS, NCEI, and POWER for one city/date and return one result each.

    An invalid city slug surfaces as a single ``source="config"`` row with
    ``ok=False`` so the caller still sees what went wrong. ASOS is not probed
    live — its parser is tested separately and its fetcher requires careful
    timezone handling that is not in scope for this diagnostic.
    """
    try:
        station = get_station(city)
    except Exception as exc:  # noqa: BLE001
        return [
            SmokeResult(
                city=city,
                target_date=target,
                source="config",
                ok=False,
                high_f=None,
                error=compact_error(exc),
            )
        ]
    return _smoke_station(city=city, station=station, target=target)


def _smoke_station(
    *, city: str, station: Station, target: date
) -> list[SmokeResult]:
    return [
        _safe_call(
            city=city,
            target=target,
            source="nws",
            fn=lambda: nws.fetch_daily_high_forecast(station, target),
        ),
        _safe_call(
            city=city,
            target=target,
            source="ncei",
            fn=lambda: ncei.fetch_daily_high(station, target),
        ),
        _safe_call(
            city=city,
            target=target,
            source="power",
            fn=lambda: power.fetch_daily_high(
                station.lat, station.lon, target, station.nws_station
            ),
        ),
    ]


def smoke_cities(cities: list[str], target: date) -> list[SmokeResult]:
    """Probe every city in ``cities`` and flatten the per-source results."""
    results: list[SmokeResult] = []
    for city in cities:
        results.extend(smoke_city(city, target))
    return results


def smoke_results_to_dataframe(results: list[SmokeResult]) -> pd.DataFrame:
    """Build a stable-column DataFrame, even when ``results`` is empty."""
    rows = [
        {
            "city": r.city,
            "target_date": r.target_date.isoformat(),
            "source": r.source,
            "ok": r.ok,
            "high_f": r.high_f,
            "error": r.error,
        }
        for r in results
    ]
    return pd.DataFrame(rows, columns=list(SMOKE_COLUMNS))

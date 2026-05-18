"""Cache-backed weather data integration helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import date
from typing import TypeVar

from src.cache import JsonCache, JsonPayload
from src.config import Station
from src.datasets.backtest import BacktestRow, make_backtest_row
from src.datasets.joins import ActualRecord, ForecastRecord
from src.fetchers.ncei import NceiDailyHigh
from src.fetchers.nws import NwsDailyHighForecast
from src.fetchers.power import PowerDailyHigh

T = TypeVar("T")


def forecast_record_from_nws(city: str, forecast: NwsDailyHighForecast) -> ForecastRecord:
    """Convert an NWS daily-high forecast into the shared forecast shape."""
    return ForecastRecord(
        city=city,
        target_date=forecast.target_date,
        source=forecast.source,
        forecast_high_f=forecast.high_f,
    )


def actual_record_from_ncei(city: str, actual: NceiDailyHigh) -> ActualRecord:
    """Convert an NCEI observed high into the shared actual shape."""
    return ActualRecord(
        city=city,
        target_date=actual.target_date,
        actual_high_f=actual.high_f,
    )


def actual_record_from_power(city: str, actual: PowerDailyHigh) -> ActualRecord:
    """Convert a POWER fallback high into the shared actual shape."""
    return ActualRecord(
        city=city,
        target_date=actual.target_date,
        actual_high_f=actual.high_f,
    )


def backtest_row_from_records(
    forecast: ForecastRecord, actual: ActualRecord
) -> BacktestRow | None:
    """Create a backtest row when both forecast and actual values are present."""
    if forecast.city != actual.city or forecast.target_date != actual.target_date:
        raise ValueError("forecast and actual records must refer to the same city/date")
    if forecast.forecast_high_f is None or actual.actual_high_f is None:
        return None
    return make_backtest_row(
        city=forecast.city,
        target_date=forecast.target_date,
        source=forecast.source,
        point_f=forecast.forecast_high_f,
        actual_high_f=actual.actual_high_f,
    )


def fetch_with_cache(
    *,
    cache: JsonCache,
    namespace: str,
    params: dict[str, object],
    fetch: Callable[[], JsonPayload],
) -> JsonPayload:
    """Read a JSON payload from cache, fetching and storing on miss."""
    cached = cache.get(namespace, params)
    if cached is not None:
        return cached
    payload = fetch()
    cache.set(namespace, params, payload)
    return payload


def dataclass_payload(value: object) -> JsonPayload:
    """Convert a dataclass record into a JSON-friendly payload."""
    if not is_dataclass(value):
        raise TypeError("value must be a dataclass instance")
    payload = asdict(value)
    for key, item in list(payload.items()):
        if isinstance(item, date):
            payload[key] = item.isoformat()
    return payload


def station_cache_params(station: Station, target: date, source: str) -> dict[str, object]:
    """Stable cache params for station/date/source fetches."""
    return {
        "source": source,
        "slug": station.slug,
        "station": station.nws_station,
        "target_date": target.isoformat(),
    }

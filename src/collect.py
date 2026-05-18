"""Cache-backed collection service for weather training data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.cache import JsonCache
from src.config import get_station
from src.datasets.backtest import BacktestRow, backtest_rows_to_dataframe
from src.datasets.collection import date_range
from src.fetchers import ncei, nws, power
from src.fetchers.ncei import NceiDailyHigh
from src.fetchers.nws import NwsDailyHighForecast
from src.fetchers.power import PowerDailyHigh
from src.pipeline.weather import (
    actual_record_from_ncei,
    actual_record_from_power,
    backtest_row_from_records,
    forecast_record_from_nws,
)


@dataclass(frozen=True)
class CollectionResult:
    """Result from collecting one city/date range."""

    city: str
    start: date
    end: date
    rows: list[BacktestRow]


def collect_backtest_rows(
    *, city: str, start: date, end: date, cache_root: Path
) -> CollectionResult:
    """Collect cache-backed NWS-vs-actual rows for a city/date range."""
    station = get_station(city)
    cache = JsonCache(cache_root)
    rows: list[BacktestRow] = []

    for target in date_range(start, end):
        forecast = _cached_nws_forecast(cache, city, station, target)
        actual = _cached_ncei_actual(cache, city, station, target)
        if actual.high_f is None:
            actual = _cached_power_actual(cache, city, station, target)

        forecast_record = forecast_record_from_nws(city, forecast)
        actual_record = actual_record_from_ncei(city, actual) if isinstance(
            actual, NceiDailyHigh
        ) else actual_record_from_power(city, actual)
        row = backtest_row_from_records(forecast_record, actual_record)
        if row is not None:
            rows.append(row)

    return CollectionResult(city=city, start=start, end=end, rows=rows)


def write_collection_csv(result: CollectionResult, path: Path) -> None:
    """Write collection rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    backtest_rows_to_dataframe(result.rows).to_csv(path, index=False)


def _cached_nws_forecast(
    cache: JsonCache, city: str, station, target: date
) -> NwsDailyHighForecast:
    params = _params(city, target, "nws")
    cached = cache.get("nws", params)
    if isinstance(cached, dict):
        return _nws_from_payload(cached)
    forecast = nws.fetch_daily_high_forecast(station, target)
    cache.set("nws", params, _record_payload(forecast))
    return forecast


def _cached_ncei_actual(cache: JsonCache, city: str, station, target: date) -> NceiDailyHigh:
    params = _params(city, target, "ncei")
    cached = cache.get("ncei", params)
    if isinstance(cached, dict):
        return _ncei_from_payload(cached)
    actual = ncei.fetch_daily_high(station, target)
    cache.set("ncei", params, _record_payload(actual))
    return actual


def _cached_power_actual(
    cache: JsonCache, city: str, station, target: date
) -> PowerDailyHigh:
    params = _params(city, target, "power")
    cached = cache.get("power", params)
    if isinstance(cached, dict):
        return _power_from_payload(cached)
    actual = power.fetch_daily_high(station.lat, station.lon, target, station.nws_station)
    cache.set("power", params, _record_payload(actual))
    return actual


def _params(city: str, target: date, source: str) -> dict[str, object]:
    return {"city": city, "target_date": target.isoformat(), "source": source}


def _record_payload(record) -> dict[str, object]:
    return {
        "station": record.station,
        "target_date": record.target_date.isoformat(),
        "high_f": record.high_f,
        "source": record.source,
    }


def _nws_from_payload(payload: dict) -> NwsDailyHighForecast:
    return NwsDailyHighForecast(
        station=str(payload["station"]),
        target_date=date.fromisoformat(str(payload["target_date"])),
        high_f=_optional_float(payload.get("high_f")),
        source=str(payload.get("source", "nws")),
    )


def _ncei_from_payload(payload: dict) -> NceiDailyHigh:
    return NceiDailyHigh(
        station=str(payload["station"]),
        target_date=date.fromisoformat(str(payload["target_date"])),
        high_f=_optional_float(payload.get("high_f")),
        source=str(payload.get("source", "ncei")),
    )


def _power_from_payload(payload: dict) -> PowerDailyHigh:
    return PowerDailyHigh(
        station=str(payload["station"]),
        target_date=date.fromisoformat(str(payload["target_date"])),
        high_f=_optional_float(payload.get("high_f")),
        source=str(payload.get("source", "power")),
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)

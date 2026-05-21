"""Cache-backed collection service for weather training data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from src.cache import JsonCache
from src.config import get_station
from src.datasets.backtest import BacktestRow, backtest_rows_to_dataframe
from src.datasets.collection import date_range
from src.datasets.joins import ActualRecord, ForecastRecord
from src.fetchers import ncei, nws, openmeteo, power
from src.fetchers.ncei import NceiDailyHigh
from src.fetchers.nws import NwsDailyHighForecast
from src.fetchers.power import PowerDailyHigh
from src.models.ensemble import naive_forecast_from_members
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
    """Collect cache-backed forecast-vs-actual rows for a city/date range."""
    station = get_station(city)
    cache = JsonCache(cache_root)
    rows: list[BacktestRow] = []
    _prefetch_historical_openmeteo_points(cache, city, station, start, end)

    for target in date_range(start, end):
        actual = _cached_ncei_actual(cache, city, station, target)
        if actual.high_f is None:
            actual = _cached_power_actual(cache, city, station, target)
        actual_record = (
            actual_record_from_ncei(city, actual)
            if isinstance(actual, NceiDailyHigh)
            else actual_record_from_power(city, actual)
        )

        if target < date.today():
            row = _historical_openmeteo_row(cache, city, station, target, actual_record)
        else:
            forecast = _cached_nws_forecast(cache, city, station, target)
            row = backtest_row_from_records(
                forecast_record_from_nws(city, forecast), actual_record
            )

        if row is not None:
            rows.append(row)

    return CollectionResult(city=city, start=start, end=end, rows=rows)


def write_collection_csv(result: CollectionResult, path: Path) -> None:
    """Write collection rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    backtest_rows_to_dataframe(result.rows).to_csv(path, index=False)


def _historical_openmeteo_row(
    cache: JsonCache, city: str, station, target: date, actual_record: ActualRecord
) -> BacktestRow | None:
    params = _params(city, target, "openmeteo_naive")
    cached = cache.get("openmeteo_naive", params)
    if isinstance(cached, dict):
        point_f = _optional_float(cached.get("point_f"))
    else:
        use_historical = target < date.today() - timedelta(days=2)
        sources = [
            openmeteo.fetch_source(
                slug,
                lat=station.lat,
                lon=station.lon,
                target=target,
                use_historical=use_historical,
            )
            for slug, *_ in openmeteo.SOURCES
        ]
        members = openmeteo.members_dataframe(sources)
        point_f = None if members.empty else naive_forecast_from_members(members).point_f
        cache.set(
            "openmeteo_naive",
            params,
            {"city": city, "target_date": target.isoformat(), "point_f": point_f},
        )

    if point_f is None:
        return None
    return backtest_row_from_records(
        forecast=ForecastRecord(
            city=city,
            target_date=target,
            source="openmeteo_naive",
            forecast_high_f=point_f,
        ),
        actual=actual_record,
    )


def _prefetch_historical_openmeteo_points(cache: JsonCache, city: str, station, start: date, end: date) -> None:
    targets = [
        target
        for target in date_range(start, end)
        if target < date.today()
        and not isinstance(cache.get("openmeteo_naive", _params(city, target, "openmeteo_naive")), dict)
    ]
    if not targets:
        return

    for use_historical, group in _group_contiguous_by_historical_mode(targets).items():
        for range_start, range_end in group:
            by_target: dict[date, list[openmeteo.ModelDailyHigh]] = {
                target: [] for target in date_range(range_start, range_end)
            }
            for slug, *_ in openmeteo.SOURCES:
                for forecast in openmeteo.fetch_source_range(
                    slug,
                    lat=station.lat,
                    lon=station.lon,
                    start=range_start,
                    end=range_end,
                    use_historical=use_historical,
                ):
                    if forecast.target_date in by_target:
                        by_target[forecast.target_date].append(forecast)

            for target, forecasts in by_target.items():
                members = openmeteo.members_dataframe(forecasts)
                point_f = None if members.empty else naive_forecast_from_members(members).point_f
                cache.set(
                    "openmeteo_naive",
                    _params(city, target, "openmeteo_naive"),
                    {
                        "city": city,
                        "target_date": target.isoformat(),
                        "point_f": point_f,
                    },
                )


def _group_contiguous_by_historical_mode(
    targets: list[date],
) -> dict[bool, list[tuple[date, date]]]:
    grouped: dict[bool, list[tuple[date, date]]] = {True: [], False: []}
    if not targets:
        return grouped

    sorted_targets = sorted(targets)
    current_mode = _uses_historical_forecast(sorted_targets[0])
    range_start = sorted_targets[0]
    previous = sorted_targets[0]

    for target in sorted_targets[1:]:
        mode = _uses_historical_forecast(target)
        if mode != current_mode or (target - previous).days != 1:
            grouped[current_mode].append((range_start, previous))
            range_start = target
            current_mode = mode
        previous = target

    grouped[current_mode].append((range_start, previous))
    return grouped


def _uses_historical_forecast(target: date) -> bool:
    return target < date.today() - timedelta(days=2)


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

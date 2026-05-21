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

OPENMETEO_NAIVE_SOURCE = "openmeteo_naive"
OPENMETEO_MODES = {"naive", "sources", "both"}


@dataclass(frozen=True)
class CollectionResult:
    """Result from collecting one city/date range."""

    city: str
    start: date
    end: date
    rows: list[BacktestRow]


def collect_backtest_rows(
    *,
    city: str,
    start: date,
    end: date,
    cache_root: Path,
    openmeteo_mode: str = "naive",
) -> CollectionResult:
    """Collect cache-backed forecast-vs-actual rows for a city/date range."""
    if openmeteo_mode not in OPENMETEO_MODES:
        raise ValueError(f"openmeteo_mode must be one of {sorted(OPENMETEO_MODES)}")
    station = get_station(city)
    cache = JsonCache(cache_root)
    rows: list[BacktestRow] = []
    _prefetch_historical_openmeteo_points(
        cache, city, station, start, end, openmeteo_mode
    )

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
            rows.extend(
                _historical_openmeteo_rows(
                    cache, city, station, target, actual_record, openmeteo_mode
                )
            )
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


def _historical_openmeteo_rows(
    cache: JsonCache,
    city: str,
    station,
    target: date,
    actual_record: ActualRecord,
    openmeteo_mode: str,
) -> list[BacktestRow]:
    rows: list[BacktestRow] = []
    for source in _openmeteo_output_sources(openmeteo_mode):
        point_f = _cached_openmeteo_point(cache, city, station, target, source)
        if point_f is None:
            continue
        row = backtest_row_from_records(
            forecast=ForecastRecord(
                city=city,
                target_date=target,
                source=source,
                forecast_high_f=point_f,
            ),
            actual=actual_record,
        )
        if row is not None:
            rows.append(row)
    return rows


def _prefetch_historical_openmeteo_points(
    cache: JsonCache,
    city: str,
    station,
    start: date,
    end: date,
    openmeteo_mode: str,
) -> None:
    targets = [
        target
        for target in date_range(start, end)
        if target < date.today()
        and _missing_openmeteo_outputs(cache, city, target, openmeteo_mode)
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
                _cache_openmeteo_points(cache, city, target, forecasts)


def _missing_openmeteo_outputs(
    cache: JsonCache, city: str, target: date, openmeteo_mode: str
) -> bool:
    return any(
        not isinstance(cache.get(source, _params(city, target, source)), dict)
        for source in _openmeteo_output_sources(openmeteo_mode)
    )


def _openmeteo_output_sources(openmeteo_mode: str) -> list[str]:
    source_slugs = [slug for slug, *_ in openmeteo.SOURCES]
    if openmeteo_mode == "naive":
        return [OPENMETEO_NAIVE_SOURCE]
    if openmeteo_mode == "sources":
        return source_slugs
    if openmeteo_mode == "both":
        return [OPENMETEO_NAIVE_SOURCE, *source_slugs]
    raise ValueError(f"openmeteo_mode must be one of {sorted(OPENMETEO_MODES)}")


def _cached_openmeteo_point(
    cache: JsonCache, city: str, station, target: date, source: str
) -> float | None:
    params = _params(city, target, source)
    cached = cache.get(source, params)
    if isinstance(cached, dict):
        return _optional_float(cached.get("point_f"))

    use_historical = target < date.today() - timedelta(days=2)
    forecasts = [
        openmeteo.fetch_source(
            slug,
            lat=station.lat,
            lon=station.lon,
            target=target,
            use_historical=use_historical,
        )
        for slug, *_ in openmeteo.SOURCES
    ]
    _cache_openmeteo_points(cache, city, target, forecasts)
    cached = cache.get(source, params)
    if isinstance(cached, dict):
        return _optional_float(cached.get("point_f"))
    return None


def _cache_openmeteo_points(
    cache: JsonCache,
    city: str,
    target: date,
    forecasts: list[openmeteo.ModelDailyHigh],
) -> None:
    members = openmeteo.members_dataframe(forecasts)
    naive_point = None if members.empty else naive_forecast_from_members(members).point_f
    _cache_openmeteo_point(cache, city, target, OPENMETEO_NAIVE_SOURCE, naive_point)

    by_source = {forecast.source: forecast for forecast in forecasts}
    for slug, *_ in openmeteo.SOURCES:
        forecast = by_source.get(slug)
        point_f = None
        if forecast is not None and forecast.members_f:
            point_f = sum(forecast.members_f) / len(forecast.members_f)
        _cache_openmeteo_point(cache, city, target, slug, point_f)


def _cache_openmeteo_point(
    cache: JsonCache, city: str, target: date, source: str, point_f: float | None
) -> None:
    cache.set(
        source,
        _params(city, target, source),
        {
            "city": city,
            "target_date": target.isoformat(),
            "source": source,
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

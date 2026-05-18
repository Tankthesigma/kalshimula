from datetime import date

import pytest

from src.cache import JsonCache
from src.config import Station
from src.datasets.joins import ActualRecord, ForecastRecord
from src.fetchers.ncei import NceiDailyHigh
from src.fetchers.nws import NwsDailyHighForecast
from src.fetchers.power import PowerDailyHigh
from src.pipeline.weather import (
    actual_record_from_ncei,
    actual_record_from_power,
    backtest_row_from_records,
    dataclass_payload,
    fetch_with_cache,
    forecast_record_from_nws,
    station_cache_params,
)


def test_forecast_record_from_nws() -> None:
    forecast = NwsDailyHighForecast(
        station="KDEN", target_date=date(2025, 1, 1), high_f=70.0
    )

    record = forecast_record_from_nws("denver", forecast)

    assert record == ForecastRecord("denver", date(2025, 1, 1), "nws", 70.0)


def test_actual_records_from_observed_sources() -> None:
    ncei = NceiDailyHigh(station="USW00003017", target_date=date(2025, 1, 1), high_f=68)
    power = PowerDailyHigh(station="KDEN", target_date=date(2025, 1, 2), high_f=69)

    assert actual_record_from_ncei("denver", ncei) == ActualRecord(
        "denver", date(2025, 1, 1), 68
    )
    assert actual_record_from_power("denver", power) == ActualRecord(
        "denver", date(2025, 1, 2), 69
    )


def test_backtest_row_from_records_requires_matching_city_and_date() -> None:
    forecast = ForecastRecord("denver", date(2025, 1, 1), "nws", 70)
    actual = ActualRecord("chicago", date(2025, 1, 1), 68)

    with pytest.raises(ValueError):
        backtest_row_from_records(forecast, actual)


def test_backtest_row_from_records_skips_missing_values() -> None:
    row = backtest_row_from_records(
        ForecastRecord("denver", date(2025, 1, 1), "nws", None),
        ActualRecord("denver", date(2025, 1, 1), 68),
    )

    assert row is None


def test_backtest_row_from_records_creates_row() -> None:
    row = backtest_row_from_records(
        ForecastRecord("denver", date(2025, 1, 1), "nws", 70),
        ActualRecord("denver", date(2025, 1, 1), 68),
    )

    assert row is not None
    assert row.absolute_error_f == 2.0


def test_fetch_with_cache_fetches_only_on_miss(tmp_path) -> None:
    cache = JsonCache(tmp_path)
    calls = 0

    def fetch():
        nonlocal calls
        calls += 1
        return {"high_f": 70}

    params = {"city": "denver", "date": "2025-01-01"}

    assert fetch_with_cache(cache=cache, namespace="nws", params=params, fetch=fetch) == {
        "high_f": 70
    }
    assert fetch_with_cache(cache=cache, namespace="nws", params=params, fetch=fetch) == {
        "high_f": 70
    }
    assert calls == 1


def test_dataclass_payload_serializes_dates() -> None:
    payload = dataclass_payload(
        NwsDailyHighForecast(station="KDEN", target_date=date(2025, 1, 1), high_f=70)
    )

    assert payload == {
        "station": "KDEN",
        "target_date": "2025-01-01",
        "high_f": 70,
        "source": "nws",
    }


def test_dataclass_payload_rejects_non_dataclasses() -> None:
    with pytest.raises(TypeError):
        dataclass_payload({"high_f": 70})


def test_station_cache_params_are_stable() -> None:
    station = Station(
        slug="denver",
        name="Denver",
        nws_station="KDEN",
        ghcnd_id="GHCND:USW00003017",
        lat=39.8,
        lon=-104.6,
        tz="America/Denver",
        lst_offset_hours=-7,
    )

    assert station_cache_params(station, date(2025, 1, 1), "ncei") == {
        "source": "ncei",
        "slug": "denver",
        "station": "KDEN",
        "target_date": "2025-01-01",
    }

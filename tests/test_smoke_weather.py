"""Tests for src.smoke_weather — fully offline via monkeypatched fetchers."""

from __future__ import annotations

from datetime import date

import pytest

from src.config import Station
from src.fetchers.ncei import NceiDailyHigh
from src.fetchers.nws import NwsDailyHighForecast
from src.fetchers.power import PowerDailyHigh
from src.smoke_weather import (
    SMOKE_COLUMNS,
    SmokeResult,
    smoke_cities,
    smoke_city,
    smoke_results_to_dataframe,
)

TARGET = date(2025, 1, 2)


def _station(slug: str = "denver") -> Station:
    return Station(
        slug=slug,
        name=slug.title(),
        nws_station="KDEN",
        ghcnd_id="GHCND:USW00003017",
        lat=39.8328,
        lon=-104.6575,
        tz="America/Denver",
        lst_offset_hours=-7,
    )


_ASOS_CSV = (
    "station,valid,tmpf\n"
    "KDEN,2025-01-02 12:53,71.0\n"
    "KDEN,2025-01-02 13:53,73.5\n"
    "KDEN,2025-01-02 14:53,72.0\n"
)


@pytest.fixture
def patch_all_ok(monkeypatch):
    """Make every source return a healthy record."""
    monkeypatch.setattr(
        "src.smoke_weather.get_station",
        lambda city: _station(city),
    )
    monkeypatch.setattr(
        "src.smoke_weather.nws.fetch_daily_high_forecast",
        lambda station, target: NwsDailyHighForecast(
            station=station.nws_station, target_date=target, high_f=75.0
        ),
    )
    monkeypatch.setattr(
        "src.smoke_weather.ncei.fetch_daily_high",
        lambda station, target: NceiDailyHigh(
            station=station.ghcnd_bare, target_date=target, high_f=70.0
        ),
    )
    monkeypatch.setattr(
        "src.smoke_weather.power.fetch_daily_high",
        lambda lat, lon, target, station: PowerDailyHigh(
            station=station, target_date=target, high_f=72.0
        ),
    )
    monkeypatch.setattr(
        "src.smoke_weather.asos.fetch_asos_csv",
        lambda station, target: _ASOS_CSV,
    )


class TestSmokeCity:
    def test_all_sources_ok(self, patch_all_ok):
        results = smoke_city("denver", TARGET)
        assert len(results) == 4
        assert {r.source for r in results} == {"nws", "ncei", "power", "asos"}
        assert all(r.ok for r in results)
        assert all(r.error is None for r in results)
        highs = {r.source: r.high_f for r in results}
        assert highs["nws"] == 75.0
        assert highs["ncei"] == 70.0
        assert highs["power"] == 72.0
        # ASOS daily high = max(71.0, 73.5, 72.0) = 73.5 from the fixture CSV.
        assert highs["asos"] == pytest.approx(73.5)

    def test_source_exception_becomes_not_ok(self, monkeypatch):
        monkeypatch.setattr(
            "src.smoke_weather.get_station",
            lambda city: _station(city),
        )

        def boom(*a, **k):
            raise RuntimeError("nws blew up")

        monkeypatch.setattr("src.smoke_weather.nws.fetch_daily_high_forecast", boom)
        monkeypatch.setattr(
            "src.smoke_weather.ncei.fetch_daily_high",
            lambda *a, **k: NceiDailyHigh(
                station="X", target_date=TARGET, high_f=50.0
            ),
        )
        monkeypatch.setattr(
            "src.smoke_weather.power.fetch_daily_high",
            lambda *a, **k: PowerDailyHigh(
                station="X", target_date=TARGET, high_f=50.0
            ),
        )
        monkeypatch.setattr(
            "src.smoke_weather.asos.fetch_asos_csv",
            lambda station, target: _ASOS_CSV,
        )

        results = smoke_city("denver", TARGET)
        by_source = {r.source: r for r in results}
        assert by_source["nws"].ok is False
        assert by_source["nws"].error is not None
        assert "nws blew up" in by_source["nws"].error
        assert by_source["ncei"].ok is True
        assert by_source["power"].ok is True
        assert by_source["asos"].ok is True

    def test_asos_exception_becomes_not_ok(self, monkeypatch):
        # ASOS failures must surface as ok=False without taking out the other
        # three sources — the smoke harness is per-source isolated.
        monkeypatch.setattr(
            "src.smoke_weather.get_station", lambda city: _station(city)
        )
        monkeypatch.setattr(
            "src.smoke_weather.nws.fetch_daily_high_forecast",
            lambda *a, **k: NwsDailyHighForecast(
                station="X", target_date=TARGET, high_f=70.0
            ),
        )
        monkeypatch.setattr(
            "src.smoke_weather.ncei.fetch_daily_high",
            lambda *a, **k: NceiDailyHigh(
                station="X", target_date=TARGET, high_f=70.0
            ),
        )
        monkeypatch.setattr(
            "src.smoke_weather.power.fetch_daily_high",
            lambda *a, **k: PowerDailyHigh(
                station="X", target_date=TARGET, high_f=70.0
            ),
        )

        def boom(*a, **k):
            raise RuntimeError("asos 500")

        monkeypatch.setattr("src.smoke_weather.asos.fetch_asos_csv", boom)

        results = smoke_city("denver", TARGET)
        by_source = {r.source: r for r in results}
        assert by_source["asos"].ok is False
        assert by_source["asos"].error is not None
        assert "asos 500" in by_source["asos"].error
        assert by_source["nws"].ok is True
        assert by_source["ncei"].ok is True
        assert by_source["power"].ok is True

    def test_asos_no_observations_today_is_ok_with_none(self, monkeypatch):
        # ASOS happy-path with zero rows for the target — fetch + parse both
        # succeed, but daily_high returns None. Should be ok=True/high=None,
        # matching how NWS/NCEI/POWER behave when source has no data.
        monkeypatch.setattr(
            "src.smoke_weather.get_station", lambda city: _station(city)
        )
        monkeypatch.setattr(
            "src.smoke_weather.nws.fetch_daily_high_forecast",
            lambda *a, **k: NwsDailyHighForecast(
                station="X", target_date=TARGET, high_f=None
            ),
        )
        monkeypatch.setattr(
            "src.smoke_weather.ncei.fetch_daily_high",
            lambda *a, **k: NceiDailyHigh(
                station="X", target_date=TARGET, high_f=None
            ),
        )
        monkeypatch.setattr(
            "src.smoke_weather.power.fetch_daily_high",
            lambda *a, **k: PowerDailyHigh(
                station="X", target_date=TARGET, high_f=None
            ),
        )
        # CSV with header + a row on a different date → daily_high=None
        monkeypatch.setattr(
            "src.smoke_weather.asos.fetch_asos_csv",
            lambda station, target: (
                "station,valid,tmpf\n"
                "KDEN,2025-01-01 12:53,40.0\n"
            ),
        )
        results = smoke_city("denver", TARGET)
        asos_result = next(r for r in results if r.source == "asos")
        assert asos_result.ok is True
        assert asos_result.high_f is None
        assert asos_result.error is None

    def test_invalid_city_returns_config_error(self, monkeypatch):
        def missing(city):
            raise KeyError(f"unknown city {city}")

        monkeypatch.setattr("src.smoke_weather.get_station", missing)
        results = smoke_city("atlantis", TARGET)
        assert len(results) == 1
        assert results[0].source == "config"
        assert results[0].ok is False
        assert results[0].error is not None
        assert "atlantis" in results[0].error

    def test_high_f_can_be_none_on_ok(self, monkeypatch):
        # ok=True even when the fetcher returns no high — that's a source
        # availability signal, not a fetcher failure.
        monkeypatch.setattr(
            "src.smoke_weather.get_station",
            lambda city: _station(city),
        )
        monkeypatch.setattr(
            "src.smoke_weather.nws.fetch_daily_high_forecast",
            lambda *a, **k: NwsDailyHighForecast(
                station="X", target_date=TARGET, high_f=None
            ),
        )
        monkeypatch.setattr(
            "src.smoke_weather.ncei.fetch_daily_high",
            lambda *a, **k: NceiDailyHigh(
                station="X", target_date=TARGET, high_f=None
            ),
        )
        monkeypatch.setattr(
            "src.smoke_weather.power.fetch_daily_high",
            lambda *a, **k: PowerDailyHigh(
                station="X", target_date=TARGET, high_f=None
            ),
        )
        monkeypatch.setattr(
            "src.smoke_weather.asos.fetch_asos_csv",
            lambda station, target: "station,valid,tmpf\n",
        )
        results = smoke_city("denver", TARGET)
        assert all(r.ok for r in results)
        assert all(r.high_f is None for r in results)


class TestSmokeCities:
    def test_combines_multiple_cities(self, patch_all_ok):
        results = smoke_cities(["denver", "nyc"], TARGET)
        assert len(results) == 8  # 4 sources × 2 cities
        assert sorted({r.city for r in results}) == ["denver", "nyc"]


class TestSmokeResultsToDataframe:
    def test_stable_columns_when_empty(self):
        df = smoke_results_to_dataframe([])
        assert list(df.columns) == list(SMOKE_COLUMNS)
        assert len(df) == 0

    def test_dataframe_rows_from_results(self):
        results = [
            SmokeResult(
                city="denver",
                target_date=TARGET,
                source="nws",
                ok=True,
                high_f=75.0,
                error=None,
            ),
            SmokeResult(
                city="denver",
                target_date=TARGET,
                source="ncei",
                ok=False,
                high_f=None,
                error="RuntimeError: boom",
            ),
        ]
        df = smoke_results_to_dataframe(results)
        assert list(df.columns) == list(SMOKE_COLUMNS)
        assert len(df) == 2
        # target_date is serialized as ISO string for stable CSV output.
        assert df.loc[0, "target_date"] == "2025-01-02"
        assert bool(df.loc[0, "ok"]) is True
        assert bool(df.loc[1, "ok"]) is False
        assert df.loc[1, "error"] == "RuntimeError: boom"

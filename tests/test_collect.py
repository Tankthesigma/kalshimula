from datetime import date

import pandas as pd

from src import collect
from src.collect import CollectionResult, collect_backtest_rows, write_collection_csv
from src.datasets.backtest import make_backtest_row
from src.fetchers.ncei import NceiDailyHigh
from src.fetchers.nws import NwsDailyHighForecast
from src.fetchers.power import PowerDailyHigh


def test_collect_backtest_rows_uses_ncei_actual(monkeypatch, tmp_path) -> None:
    calls = {"nws": 0, "ncei": 0, "power": 0}

    def fake_nws(station, target):
        calls["nws"] += 1
        return NwsDailyHighForecast(station=station.nws_station, target_date=target, high_f=70)

    def fake_ncei(station, target):
        calls["ncei"] += 1
        return NceiDailyHigh(station=station.ghcnd_bare, target_date=target, high_f=68)

    def fake_power(lat, lon, target, station):
        calls["power"] += 1
        return PowerDailyHigh(station=station, target_date=target, high_f=67)

    monkeypatch.setattr(collect.nws, "fetch_daily_high_forecast", fake_nws)
    monkeypatch.setattr(collect.ncei, "fetch_daily_high", fake_ncei)
    monkeypatch.setattr(collect.power, "fetch_daily_high", fake_power)

    result = collect_backtest_rows(
        city="denver",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
        cache_root=tmp_path,
    )

    assert len(result.rows) == 1
    assert result.rows[0].absolute_error_f == 2.0
    assert calls == {"nws": 1, "ncei": 1, "power": 0}


def test_collect_backtest_rows_falls_back_to_power(monkeypatch, tmp_path) -> None:
    def fake_nws(station, target):
        return NwsDailyHighForecast(station=station.nws_station, target_date=target, high_f=70)

    def fake_ncei(station, target):
        return NceiDailyHigh(station=station.ghcnd_bare, target_date=target, high_f=None)

    def fake_power(lat, lon, target, station):
        return PowerDailyHigh(station=station, target_date=target, high_f=66)

    monkeypatch.setattr(collect.nws, "fetch_daily_high_forecast", fake_nws)
    monkeypatch.setattr(collect.ncei, "fetch_daily_high", fake_ncei)
    monkeypatch.setattr(collect.power, "fetch_daily_high", fake_power)

    result = collect_backtest_rows(
        city="denver",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
        cache_root=tmp_path,
    )

    assert len(result.rows) == 1
    assert result.rows[0].actual_high_f == 66.0


def test_collect_backtest_rows_uses_cache_on_second_call(monkeypatch, tmp_path) -> None:
    calls = 0

    def fake_nws(station, target):
        nonlocal calls
        calls += 1
        return NwsDailyHighForecast(station=station.nws_station, target_date=target, high_f=70)

    def fake_ncei(station, target):
        return NceiDailyHigh(station=station.ghcnd_bare, target_date=target, high_f=68)

    monkeypatch.setattr(collect.nws, "fetch_daily_high_forecast", fake_nws)
    monkeypatch.setattr(collect.ncei, "fetch_daily_high", fake_ncei)

    for _ in range(2):
        collect_backtest_rows(
            city="denver",
            start=date(2025, 1, 1),
            end=date(2025, 1, 1),
            cache_root=tmp_path,
        )

    assert calls == 1


def test_write_collection_csv(tmp_path) -> None:
    path = tmp_path / "out" / "rows.csv"
    rows = [
        make_backtest_row(
            city="denver",
            target_date=date(2025, 1, 1),
            source="nws",
            point_f=70,
            actual_high_f=68,
        )
    ]
    write_collection_csv(
        CollectionResult(
            city="denver", start=date(2025, 1, 1), end=date(2025, 1, 1), rows=rows
        ),
        path,
    )

    df = pd.read_csv(path)

    assert df.iloc[0]["city"] == "denver"
    assert df.iloc[0]["absolute_error_f"] == 2.0

from datetime import date

import pandas as pd

from src import collect
from src.collect import CollectionResult, collect_backtest_rows, write_collection_csv
from src.datasets.backtest import make_backtest_row
from src.fetchers.ncei import NceiDailyHigh
from src.fetchers.nws import NwsDailyHighForecast
from src.fetchers.openmeteo import ModelDailyHigh
from src.fetchers.power import PowerDailyHigh


def test_collect_backtest_rows_uses_ncei_actual_for_future_nws(monkeypatch, tmp_path) -> None:
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
        start=date(2099, 1, 1),
        end=date(2099, 1, 1),
        cache_root=tmp_path,
    )

    assert len(result.rows) == 1
    assert result.rows[0].source == "nws"
    assert result.rows[0].absolute_error_f == 2.0
    assert calls == {"nws": 1, "ncei": 1, "power": 0}


def test_collect_backtest_rows_uses_openmeteo_for_historical_date(monkeypatch, tmp_path) -> None:
    calls = {"openmeteo": 0, "nws": 0}

    def fake_nws(station, target):
        calls["nws"] += 1
        return NwsDailyHighForecast(station=station.nws_station, target_date=target, high_f=70)

    def fake_ncei(station, target):
        return NceiDailyHigh(station=station.ghcnd_bare, target_date=target, high_f=68)

    def fake_fetch_source_range(slug, *, lat, lon, start, end, use_historical):
        calls["openmeteo"] += 1
        assert use_historical is True
        assert start == end == date(2025, 1, 1)
        return [ModelDailyHigh(source=slug, target_date=start, members_f=[66.0, 70.0])]

    monkeypatch.setattr(collect.nws, "fetch_daily_high_forecast", fake_nws)
    monkeypatch.setattr(collect.ncei, "fetch_daily_high", fake_ncei)
    monkeypatch.setattr(collect.openmeteo, "fetch_source_range", fake_fetch_source_range)

    result = collect_backtest_rows(
        city="denver",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
        cache_root=tmp_path,
    )

    assert len(result.rows) == 1
    assert result.rows[0].source == "openmeteo_naive"
    assert result.rows[0].point_f == 68.0
    assert result.rows[0].actual_high_f == 68.0
    assert calls["openmeteo"] == len(collect.openmeteo.SOURCES)
    assert calls["nws"] == 0


def test_collect_backtest_rows_can_emit_openmeteo_source_rows(
    monkeypatch, tmp_path
) -> None:
    def fake_ncei(station, target):
        return NceiDailyHigh(station=station.ghcnd_bare, target_date=target, high_f=68)

    def fake_fetch_source_range(slug, *, lat, lon, start, end, use_historical):
        members = [66.0, 70.0] if slug == "gfs_ens" else [70.0]
        return [ModelDailyHigh(source=slug, target_date=start, members_f=members)]

    monkeypatch.setattr(collect.ncei, "fetch_daily_high", fake_ncei)
    monkeypatch.setattr(collect.openmeteo, "fetch_source_range", fake_fetch_source_range)

    result = collect_backtest_rows(
        city="denver",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
        cache_root=tmp_path,
        openmeteo_mode="both",
    )

    rows_by_source = {row.source: row for row in result.rows}
    source_slugs = {slug for slug, *_ in collect.openmeteo.SOURCES}
    assert set(rows_by_source) == {collect.OPENMETEO_NAIVE_SOURCE, *source_slugs}
    assert rows_by_source["gfs_ens"].point_f == 68.0
    assert rows_by_source[collect.OPENMETEO_NAIVE_SOURCE].point_f == 69.5
    assert all(row.actual_high_f == 68.0 for row in result.rows)


def test_collect_backtest_rows_sources_mode_excludes_naive(monkeypatch, tmp_path) -> None:
    def fake_ncei(station, target):
        return NceiDailyHigh(station=station.ghcnd_bare, target_date=target, high_f=68)

    def fake_fetch_source_range(slug, *, lat, lon, start, end, use_historical):
        return [ModelDailyHigh(source=slug, target_date=start, members_f=[70.0])]

    monkeypatch.setattr(collect.ncei, "fetch_daily_high", fake_ncei)
    monkeypatch.setattr(collect.openmeteo, "fetch_source_range", fake_fetch_source_range)

    result = collect_backtest_rows(
        city="denver",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
        cache_root=tmp_path,
        openmeteo_mode="sources",
    )

    assert {row.source for row in result.rows} == {
        slug for slug, *_ in collect.openmeteo.SOURCES
    }


def test_collect_backtest_rows_falls_back_to_power(monkeypatch, tmp_path) -> None:
    def fake_ncei(station, target):
        return NceiDailyHigh(station=station.ghcnd_bare, target_date=target, high_f=None)

    def fake_power(lat, lon, target, station):
        return PowerDailyHigh(station=station, target_date=target, high_f=66)

    def fake_fetch_source_range(slug, *, lat, lon, start, end, use_historical):
        return [ModelDailyHigh(source=slug, target_date=start, members_f=[70.0])]

    monkeypatch.setattr(collect.ncei, "fetch_daily_high", fake_ncei)
    monkeypatch.setattr(collect.power, "fetch_daily_high", fake_power)
    monkeypatch.setattr(collect.openmeteo, "fetch_source_range", fake_fetch_source_range)

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

    def fake_ncei(station, target):
        return NceiDailyHigh(station=station.ghcnd_bare, target_date=target, high_f=68)

    def fake_fetch_source_range(slug, *, lat, lon, start, end, use_historical):
        nonlocal calls
        calls += 1
        return [ModelDailyHigh(source=slug, target_date=start, members_f=[70.0])]

    monkeypatch.setattr(collect.ncei, "fetch_daily_high", fake_ncei)
    monkeypatch.setattr(collect.openmeteo, "fetch_source_range", fake_fetch_source_range)

    for _ in range(2):
        collect_backtest_rows(
            city="denver",
            start=date(2025, 1, 1),
            end=date(2025, 1, 1),
            cache_root=tmp_path,
        )

    assert calls == len(collect.openmeteo.SOURCES)


def test_collect_backtest_rows_prefetches_openmeteo_range(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_ncei(station, target):
        return NceiDailyHigh(station=station.ghcnd_bare, target_date=target, high_f=68)

    def fake_fetch_source_range(slug, *, lat, lon, start, end, use_historical):
        calls.append((slug, start, end))
        return [
            ModelDailyHigh(source=slug, target_date=target, members_f=[70.0])
            for target in collect.date_range(start, end)
        ]

    monkeypatch.setattr(collect.ncei, "fetch_daily_high", fake_ncei)
    monkeypatch.setattr(collect.openmeteo, "fetch_source_range", fake_fetch_source_range)

    result = collect_backtest_rows(
        city="denver",
        start=date(2025, 1, 1),
        end=date(2025, 1, 3),
        cache_root=tmp_path,
    )

    assert len(result.rows) == 3
    assert len(calls) == len(collect.openmeteo.SOURCES)
    assert all(start == date(2025, 1, 1) for _, start, _ in calls)
    assert all(end == date(2025, 1, 3) for _, _, end in calls)


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

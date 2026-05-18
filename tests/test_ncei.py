"""Tests for src.fetchers.ncei — parser-focused, fully offline."""

from __future__ import annotations

from datetime import date

import pytest

from src.fetchers.ncei import NceiDailyHigh, parse_daily_high

STATION = "USW00094728"
TARGET = date(2025, 1, 2)


def _row(
    *,
    date_str: str = "2025-01-02",
    datatype: str = "TMAX",
    value: object = 100,
    station: str = STATION,
) -> dict:
    return {
        "date": date_str,
        "datatype": datatype,
        "value": value,
        "station": station,
    }


def _payload(*rows: dict) -> dict:
    return {"results": list(rows)}


class TestParseDailyHigh:
    def test_matching_tmax_converts_tenths_celsius_to_fahrenheit(self):
        # 100 tenths C = 10 C = 50 F
        result = parse_daily_high(
            _payload(_row(value=100)), TARGET, STATION
        )
        assert isinstance(result, NceiDailyHigh)
        assert result.station == STATION
        assert result.target_date == TARGET
        assert result.source == "ncei"
        assert result.high_f == pytest.approx(50.0)

    def test_non_tmax_rows_ignored(self):
        result = parse_daily_high(
            _payload(
                _row(datatype="TMIN", value=10),
                _row(datatype="PRCP", value=0),
            ),
            TARGET,
            STATION,
        )
        assert result.high_f is None

    def test_multiple_matching_tmax_returns_max(self):
        # Duplicates can happen with multi-station GHCND extracts.
        result = parse_daily_high(
            _payload(
                _row(value=100),  # 50 F
                _row(value=156),  # 60.08 F
                _row(value=120),  # 53.6 F
            ),
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(60.08)

    def test_missing_results_returns_none(self):
        assert parse_daily_high({}, TARGET, STATION).high_f is None
        assert (
            parse_daily_high({"results": []}, TARGET, STATION).high_f is None
        )

    def test_results_not_a_list_returns_none(self):
        assert (
            parse_daily_high({"results": "oops"}, TARGET, STATION).high_f
            is None
        )

    def test_bad_or_missing_values_ignored(self):
        result = parse_daily_high(
            _payload(
                _row(value=None),
                _row(value=""),
                _row(value="M"),
                _row(value="abc"),
            ),
            TARGET,
            STATION,
        )
        assert result.high_f is None

    def test_datetime_style_date_strings(self):
        # NCEI sometimes returns "2025-01-02T00:00:00".
        result = parse_daily_high(
            _payload(_row(date_str="2025-01-02T00:00:00", value=200)),
            TARGET,
            STATION,
        )
        # 200 tenths C = 20 C = 68 F
        assert result.high_f == pytest.approx(68.0)

    def test_different_date_ignored(self):
        result = parse_daily_high(
            _payload(_row(date_str="2025-01-03", value=200)),
            TARGET,
            STATION,
        )
        assert result.high_f is None

    def test_non_dict_row_skipped(self):
        result = parse_daily_high(
            {"results": ["bad", _row(value=100)]},
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(50.0)

    def test_mixed_good_and_bad_rows(self):
        # Good row should still produce a value despite garbage neighbours.
        result = parse_daily_high(
            _payload(
                _row(value=None),
                _row(value=156),  # 60.08 F
                _row(datatype="TMIN", value=200),
            ),
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(60.08)

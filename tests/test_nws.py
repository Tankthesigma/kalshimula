"""Tests for src.fetchers.nws — parser-focused, fully offline."""

from __future__ import annotations

from datetime import date

import pytest

from src.fetchers.nws import (
    NwsDailyHighForecast,
    parse_daily_high_forecast,
)

STATION = "https://api.weather.gov/gridpoints/OKX/33,37/forecast"
TARGET = date(2025, 1, 2)


def _period(
    *,
    name: str = "Today",
    start: str = "2025-01-02T06:00:00-05:00",
    end: str = "2025-01-02T18:00:00-05:00",
    is_daytime: bool = True,
    temperature: object = 50,
    unit: str = "F",
) -> dict:
    return {
        "name": name,
        "startTime": start,
        "endTime": end,
        "isDaytime": is_daytime,
        "temperature": temperature,
        "temperatureUnit": unit,
    }


def _payload(*periods: dict) -> dict:
    return {"properties": {"periods": list(periods)}}


class TestParseDailyHighForecast:
    def test_matching_fahrenheit_daytime(self):
        result = parse_daily_high_forecast(
            _payload(_period(temperature=72, unit="F")),
            TARGET,
            STATION,
        )
        assert isinstance(result, NwsDailyHighForecast)
        assert result.station == STATION
        assert result.target_date == TARGET
        assert result.source == "nws"
        assert result.high_f == pytest.approx(72.0)

    def test_matching_celsius_converts(self):
        result = parse_daily_high_forecast(
            _payload(_period(temperature=20, unit="C")),
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(68.0)

    def test_nighttime_period_ignored(self):
        result = parse_daily_high_forecast(
            _payload(
                _period(
                    name="Tonight",
                    start="2025-01-02T18:00:00-05:00",
                    end="2025-01-03T06:00:00-05:00",
                    is_daytime=False,
                    temperature=40,
                )
            ),
            TARGET,
            STATION,
        )
        assert result.high_f is None

    def test_multiple_daytime_returns_max(self):
        # Two daytime periods on the same date — pick the higher temp.
        result = parse_daily_high_forecast(
            _payload(
                _period(name="This Afternoon", temperature=68),
                _period(
                    name="Late Afternoon",
                    start="2025-01-02T15:00:00-05:00",
                    temperature=75,
                ),
                _period(
                    name="Early Morning",
                    start="2025-01-02T06:00:00-05:00",
                    temperature=60,
                ),
            ),
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(75.0)

    def test_missing_periods_returns_none(self):
        assert (
            parse_daily_high_forecast({}, TARGET, STATION).high_f is None
        )
        assert (
            parse_daily_high_forecast(
                {"properties": {}}, TARGET, STATION
            ).high_f
            is None
        )
        assert (
            parse_daily_high_forecast(
                {"properties": {"periods": []}}, TARGET, STATION
            ).high_f
            is None
        )

    def test_bad_temperature_returns_none(self):
        result = parse_daily_high_forecast(
            _payload(_period(temperature=None)),
            TARGET,
            STATION,
        )
        assert result.high_f is None

    def test_non_numeric_temperature_returns_none(self):
        result = parse_daily_high_forecast(
            _payload(_period(temperature="N/A")),
            TARGET,
            STATION,
        )
        assert result.high_f is None

    def test_different_date_ignored(self):
        result = parse_daily_high_forecast(
            _payload(
                _period(start="2025-01-03T06:00:00-05:00", temperature=80)
            ),
            TARGET,
            STATION,
        )
        assert result.high_f is None

    def test_unknown_temperature_unit_skipped(self):
        # An unknown unit (e.g. "K") must not silently leak through.
        result = parse_daily_high_forecast(
            _payload(_period(temperature=300, unit="K")),
            TARGET,
            STATION,
        )
        assert result.high_f is None

    def test_periods_not_a_list_returns_none(self):
        result = parse_daily_high_forecast(
            {"properties": {"periods": "oops"}}, TARGET, STATION
        )
        assert result.high_f is None

    def test_non_dict_period_skipped(self):
        result = parse_daily_high_forecast(
            {"properties": {"periods": ["bad", _period(temperature=71)]}},
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(71.0)

"""Tests for src.fetchers.power — parser-focused, fully offline."""

from __future__ import annotations

from datetime import date

import pytest

from src.fetchers.power import PowerDailyHigh, parse_daily_high

STATION = "nyc_lga"
TARGET = date(2025, 1, 2)
TARGET_KEY = "20250102"


def _payload(t2m_max: dict | None) -> dict:
    if t2m_max is None:
        return {"properties": {"parameter": {}}}
    return {"properties": {"parameter": {"T2M_MAX": t2m_max}}}


class TestParseDailyHigh:
    def test_celsius_converts_to_fahrenheit(self):
        # 20 C → 68 F
        result = parse_daily_high(
            _payload({TARGET_KEY: 20.0}), TARGET, STATION
        )
        assert isinstance(result, PowerDailyHigh)
        assert result.station == STATION
        assert result.target_date == TARGET
        assert result.source == "power"
        assert result.high_f == pytest.approx(68.0)

    def test_matching_yyyymmdd_key_returns_high(self):
        result = parse_daily_high(
            _payload(
                {
                    "20250101": 10.0,
                    TARGET_KEY: 15.0,  # 15 C → 59 F
                    "20250103": 12.0,
                }
            ),
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(59.0)

    def test_missing_target_returns_none(self):
        result = parse_daily_high(
            _payload({"20250103": 15.0}), TARGET, STATION
        )
        assert result.high_f is None

    def test_missing_parameter_shape_returns_none(self):
        assert parse_daily_high({}, TARGET, STATION).high_f is None
        assert (
            parse_daily_high({"properties": {}}, TARGET, STATION).high_f
            is None
        )
        assert (
            parse_daily_high(
                {"properties": {"parameter": {}}}, TARGET, STATION
            ).high_f
            is None
        )
        assert (
            parse_daily_high(
                {"properties": {"parameter": {"T2M_MAX": "oops"}}},
                TARGET,
                STATION,
            ).high_f
            is None
        )

    def test_bad_or_missing_value_returns_none(self):
        # POWER fill value: -999 means missing.
        assert (
            parse_daily_high(
                _payload({TARGET_KEY: -999.0}), TARGET, STATION
            ).high_f
            is None
        )
        assert (
            parse_daily_high(
                _payload({TARGET_KEY: None}), TARGET, STATION
            ).high_f
            is None
        )
        assert (
            parse_daily_high(
                _payload({TARGET_KEY: "M"}), TARGET, STATION
            ).high_f
            is None
        )
        assert (
            parse_daily_high(
                _payload({TARGET_KEY: "abc"}), TARGET, STATION
            ).high_f
            is None
        )

    def test_numeric_string_value_accepted(self):
        # POWER sometimes hands back stringified floats.
        result = parse_daily_high(
            _payload({TARGET_KEY: "20.0"}), TARGET, STATION
        )
        assert result.high_f == pytest.approx(68.0)

    def test_properties_not_dict_returns_none(self):
        assert (
            parse_daily_high({"properties": "oops"}, TARGET, STATION).high_f
            is None
        )

    def test_dataclass_metadata_preserved(self):
        result = parse_daily_high(
            _payload({TARGET_KEY: 0.0}), TARGET, STATION
        )
        assert result.target_date == TARGET
        assert result.station == STATION
        assert result.source == "power"

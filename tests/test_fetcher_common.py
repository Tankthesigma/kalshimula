"""Tests for src.fetchers.common helpers — pure, deterministic, offline."""

from __future__ import annotations

from datetime import date

import pytest

from src.fetchers.common import (
    c_tenths_to_f,
    c_to_f,
    iso_date_prefix_matches,
    safe_float,
)


class TestCToF:
    def test_freezing(self):
        assert c_to_f(0) == pytest.approx(32.0)

    def test_boiling(self):
        assert c_to_f(100) == pytest.approx(212.0)

    def test_negative(self):
        assert c_to_f(-40) == pytest.approx(-40.0)

    def test_float_input(self):
        assert c_to_f(20.0) == pytest.approx(68.0)


class TestCTenthsToF:
    def test_tenths_freezing(self):
        # 0 tenths C = 0 C = 32 F
        assert c_tenths_to_f(0) == pytest.approx(32.0)

    def test_tenths_typical_summer_high(self):
        # 300 tenths C = 30 C = 86 F
        assert c_tenths_to_f(300) == pytest.approx(86.0)

    def test_tenths_negative(self):
        # -100 tenths C = -10 C = 14 F
        assert c_tenths_to_f(-100) == pytest.approx(14.0)


class TestSafeFloat:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1, 1.0),
            (1.5, 1.5),
            (-3, -3.0),
            ("1", 1.0),
            ("1.5", 1.5),
            ("  -2.25  ", -2.25),
            ("1e2", 100.0),
        ],
    )
    def test_accepts_numerics(self, value, expected):
        assert safe_float(value) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "   ",
            "M",
            "m",
            "NA",
            "n/a",
            "NaN",
            "None",
            "NULL",
            "abc",
            "1.2.3",
            [1.0],
            {"a": 1},
            object(),
        ],
    )
    def test_rejects_bad(self, value):
        assert safe_float(value) is None

    def test_rejects_booleans(self):
        # bools are an int subclass in Python — we deliberately reject them
        # because True/False in a weather payload is meaningless.
        assert safe_float(True) is None
        assert safe_float(False) is None


class TestIsoDatePrefixMatches:
    def test_iso_date_only(self):
        assert iso_date_prefix_matches("2025-01-02", date(2025, 1, 2))

    def test_iso_datetime_with_t_separator(self):
        assert iso_date_prefix_matches(
            "2025-01-02T13:53:00-05:00", date(2025, 1, 2)
        )

    def test_iso_datetime_with_space_separator(self):
        assert iso_date_prefix_matches("2025-01-02 13:53", date(2025, 1, 2))

    def test_different_day(self):
        assert not iso_date_prefix_matches(
            "2025-01-03T00:00:00", date(2025, 1, 2)
        )

    def test_non_string_inputs(self):
        assert not iso_date_prefix_matches(None, date(2025, 1, 2))
        assert not iso_date_prefix_matches(20250102, date(2025, 1, 2))
        assert not iso_date_prefix_matches("", date(2025, 1, 2))

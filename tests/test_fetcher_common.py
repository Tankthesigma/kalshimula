"""Tests for src.fetchers.common helpers — pure, deterministic, offline."""

from __future__ import annotations

from datetime import date

import pytest

from src.fetchers.common import (
    c_tenths_to_f,
    c_to_f,
    compact_error,
    is_missing_value,
    iso_date_prefix_matches,
    normalize_station,
    safe_float,
    safe_int,
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
        assert c_tenths_to_f(0) == pytest.approx(32.0)

    def test_tenths_typical_summer_high(self):
        # 300 tenths C = 30 C = 86 F
        assert c_tenths_to_f(300) == pytest.approx(86.0)

    def test_tenths_negative(self):
        # -100 tenths C = -10 C = 14 F
        assert c_tenths_to_f(-100) == pytest.approx(14.0)


class TestIsMissingValue:
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
            "NAN",
            "None",
            "NULL",
            "missing",
            "MISSING",
            True,
            False,
        ],
    )
    def test_missing(self, value):
        assert is_missing_value(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            0,
            0.0,
            -1,
            "0",
            "1.5",
            "abc",  # not a missing marker, just bad
            [],
            {},
            object(),
        ],
    )
    def test_present(self, value):
        assert is_missing_value(value) is False


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
            "missing",
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


class TestSafeInt:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (12, 12),
            (-7, -7),
            (0, 0),
            (12.0, 12),
            (-7.0, -7),
            ("12", 12),
            ("  12  ", 12),
            ("12.0", 12),
            ("-7", -7),
        ],
    )
    def test_accepts_whole_numbers(self, value, expected):
        assert safe_int(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            12.3,
            "12.3",
            "abc",
            "1.2.3",
            None,
            "",
            "M",
            "NA",
            "NaN",
            float("nan"),
            True,
            False,
            [12],
            object(),
        ],
    )
    def test_rejects_bad(self, value):
        assert safe_int(value) is None


class TestNormalizeStation:
    def test_uppercases(self):
        assert normalize_station("kord") == "KORD"

    def test_strips_whitespace(self):
        assert normalize_station("  kord  ") == "KORD"

    def test_already_normalized(self):
        assert normalize_station("KORD") == "KORD"

    def test_mixed_case(self):
        assert normalize_station("KoRd") == "KORD"

    def test_non_string(self):
        assert normalize_station(None) == ""  # type: ignore[arg-type]
        assert normalize_station(123) == ""  # type: ignore[arg-type]

    def test_empty_string(self):
        assert normalize_station("") == ""


class TestIsoDatePrefixMatches:
    def test_iso_date_only(self):
        assert iso_date_prefix_matches("2025-01-02", date(2025, 1, 2))

    def test_iso_datetime_with_t_separator(self):
        assert iso_date_prefix_matches(
            "2025-01-02T13:53:00-05:00", date(2025, 1, 2)
        )

    def test_iso_datetime_with_space_separator(self):
        assert iso_date_prefix_matches("2025-01-02 13:53", date(2025, 1, 2))

    def test_iso_datetime_with_z_suffix(self):
        assert iso_date_prefix_matches("2025-01-02T00:00:00Z", date(2025, 1, 2))

    def test_different_day(self):
        assert not iso_date_prefix_matches(
            "2025-01-03T00:00:00", date(2025, 1, 2)
        )

    def test_non_string_inputs(self):
        assert not iso_date_prefix_matches(None, date(2025, 1, 2))
        assert not iso_date_prefix_matches(20250102, date(2025, 1, 2))
        assert not iso_date_prefix_matches("", date(2025, 1, 2))


class TestCompactError:
    def test_basic(self):
        try:
            raise ValueError("bad input")
        except ValueError as exc:
            assert compact_error(exc) == "ValueError: bad input"

    def test_no_message(self):
        try:
            raise RuntimeError
        except RuntimeError as exc:
            assert compact_error(exc) == "RuntimeError"

    def test_newlines_collapsed(self):
        try:
            raise ValueError("line one\nline two\r\nline three")
        except ValueError as exc:
            rendered = compact_error(exc)
            assert "\n" not in rendered
            assert "\r" not in rendered
            assert "line one" in rendered and "line two" in rendered

    def test_truncation(self):
        long_msg = "x" * 500
        try:
            raise RuntimeError(long_msg)
        except RuntimeError as exc:
            rendered = compact_error(exc, max_len=50)
            assert len(rendered) == 50
            assert rendered.endswith("…")

    def test_default_max_len_is_160(self):
        long_msg = "x" * 500
        try:
            raise RuntimeError(long_msg)
        except RuntimeError as exc:
            rendered = compact_error(exc)
            assert len(rendered) == 160

    def test_short_message_unchanged(self):
        try:
            raise KeyError("a")
        except KeyError as exc:
            # KeyError's str() wraps the message in quotes — that's expected.
            rendered = compact_error(exc)
            assert rendered.startswith("KeyError")
            assert len(rendered) < 50

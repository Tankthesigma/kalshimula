"""Regression tests against snapshotted live NCEI Access Data Service responses.

Each fixture under ``tests/fixtures/ncei_*.json`` was captured directly from a
live ``GET https://www.ncei.noaa.gov/access/services/data/v1`` call with
``dataset=daily-summaries&dataTypes=TMAX&format=json&units=metric``. The
parser must keep handling these exact shapes — if NCEI renames a field or
flips a casing convention again, these tests will catch it on the PR that
imports the broken parser instead of a week later via the scheduled live
smoke workflow.

To refresh: re-run the throwaway capture script alongside this file (it lives
in repo root during development; not committed) and re-validate the expected
``high_f`` values below.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.fetchers.ncei import parse_daily_high

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_fixture_dir_present():
    # If this fails, the fixtures dir was renamed/moved — fail loudly so the
    # cases below don't silently file-not-found into a wrong shape.
    assert FIXTURES.exists() and FIXTURES.is_dir()


class TestLiveNceiPayloads:
    """One test per captured live response. Real shape, frozen in fixture."""

    def test_nyc_january_first_2025_returns_celsius_high(self):
        # 10.6 °C → 51.08 °F. Cold January NYC day.
        payload = _load("ncei_nyc_2025_01_01.json")
        result = parse_daily_high(payload, date(2025, 1, 1), "USW00094728")
        assert result.high_f == pytest.approx(51.08, abs=0.01)
        assert result.source == "ncei"
        assert result.target_date == date(2025, 1, 1)

    def test_denver_july_hot_day(self):
        # 35.0 °C → 95.0 °F. Hot Denver summer day, exercises high-end values.
        payload = _load("ncei_denver_2025_07_15.json")
        result = parse_daily_high(payload, date(2025, 7, 15), "USW00003017")
        assert result.high_f == pytest.approx(95.0, abs=0.01)

    def test_miami_january_warm_day(self):
        # 27.2 °C → 80.96 °F. Miami is hot even in January — exercises a
        # mid-tropical TMAX through the same parser path as winter NYC.
        payload = _load("ncei_miami_2025_01_01.json")
        result = parse_daily_high(payload, date(2025, 1, 1), "USW00012839")
        assert result.high_f == pytest.approx(80.96, abs=0.01)

    def test_chicago_january_negative_celsius(self):
        # -6.6 °C → 20.12 °F. Negative Celsius round-trip — guards against
        # someone "fixing" the parser to strip leading '-' on string values.
        payload = _load("ncei_chicago_2025_01_15.json")
        result = parse_daily_high(payload, date(2025, 1, 15), "USW00014819")
        assert result.high_f == pytest.approx(20.12, abs=0.01)

    def test_far_future_date_returns_no_high(self):
        # 2030-01-01 has no published observations yet — Access Data Service
        # returns a literal bare empty list ``[]``. Parser must surface that
        # as high_f=None, not crash and not invent a value.
        payload = _load("ncei_nyc_2030_01_01_empty.json")
        assert payload == []
        result = parse_daily_high(payload, date(2030, 1, 1), "USW00094728")
        assert result.high_f is None
        assert result.source == "ncei"

    def test_all_fixtures_are_bare_lists(self):
        # NCEI's `units=metric&format=json` reliably returns a top-level
        # JSON array. If a future fixture comes back as `{"results": [...]}`
        # (the legacy CDO shape) we want a visible heads-up here so the
        # parser's dual-shape support is intentionally tested both ways.
        for fixture in FIXTURES.glob("ncei_*.json"):
            payload = json.loads(fixture.read_text(encoding="utf-8"))
            assert isinstance(payload, list), (
                f"{fixture.name} is not a bare list — NCEI may have changed "
                "response shape; update parser + tests."
            )

    def test_field_keys_remain_uppercase(self):
        # Catches a future API change where NCEI flips back to lowercase
        # `date`/`station`/`tmax` (the legacy CDO casing). The parser handles
        # both shapes, but losing the uppercase variant would mean every
        # production NCEI call silently changed code paths.
        for fixture in FIXTURES.glob("ncei_*.json"):
            payload = json.loads(fixture.read_text(encoding="utf-8"))
            for row in payload:
                assert "DATE" in row, f"{fixture.name}: missing DATE field"
                assert "TMAX" in row, f"{fixture.name}: missing TMAX field"
                # STATION is informational — not strictly required.

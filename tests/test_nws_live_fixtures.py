"""Regression tests against snapshotted live NWS forecast responses.

Each fixture under ``tests/fixtures/nws_*.json`` wraps the live response
from ``api.weather.gov/points/{lat},{lon}`` followed by
``api.weather.gov/gridpoints/.../forecast`` at the time of capture. The
expected daily-high (computed at capture) is embedded in the fixture so
the test is fully deterministic on the snapshot.

If NWS changes the period field names, casing, or structure, these tests
fail at PR-gate time instead of a week later via the scheduled live
smoke workflow.

To refresh: re-run the throwaway capture script that lived next to this
file during development (it computes ``expected_high_f`` from the
captured payload and embeds it). Refresh requires nothing more than
running the script — values self-update on each capture.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.fetchers.nws import (
    forecast_url_from_points_payload,
    parse_daily_high_forecast,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _all_nws_fixtures() -> list[Path]:
    return sorted(FIXTURES.glob("nws_*.json"))


def test_fixture_dir_present():
    assert FIXTURES.exists() and FIXTURES.is_dir()


def test_at_least_one_nws_fixture():
    # Guard against an empty capture run silently leaving the regression
    # test pool empty.
    assert len(_all_nws_fixtures()) >= 1


@pytest.mark.parametrize(
    "name",
    [f.name for f in _all_nws_fixtures()],
)
class TestLiveNwsFixture:
    """One pytest case per captured fixture. New fixtures auto-extend."""

    def test_forecast_url_resolves_from_points_response(self, name: str):
        fixture = _load(name)
        points = fixture["points_response"]
        url = forecast_url_from_points_payload(points)
        assert url.startswith("https://api.weather.gov/gridpoints/"), (
            f"{name}: forecast URL no longer matches the gridpoint URL "
            "shape; NWS may have changed its routing convention."
        )

    def test_forecast_payload_has_periods_list(self, name: str):
        fixture = _load(name)
        periods = (
            fixture["forecast_response"]
            .get("properties", {})
            .get("periods")
        )
        assert isinstance(periods, list), (
            f"{name}: periods is missing or not a list; NWS may have "
            "changed the forecast payload shape."
        )
        assert len(periods) > 0, f"{name}: periods list is empty"

    def test_period_required_fields_present(self, name: str):
        fixture = _load(name)
        periods = fixture["forecast_response"]["properties"]["periods"]
        # Fields the parser reads — drift here is the silent-bug class
        # we want to catch at PR time.
        required = {
            "startTime",
            "isDaytime",
            "temperature",
            "temperatureUnit",
        }
        for i, period in enumerate(periods[:6]):  # spot-check first few
            missing = required - period.keys()
            assert not missing, (
                f"{name}: period {i} missing fields {missing}; "
                "NWS forecast schema drift."
            )

    def test_parser_returns_captured_high(self, name: str):
        # Deterministic: the fixture itself records the expected high_f
        # at capture time, so this assertion locks the parser to the
        # exact response shape that was live when the fixture was made.
        fixture = _load(name)
        target = date.fromisoformat(fixture["captured_target_date"])
        station = fixture["captured_for_station"]
        expected = fixture["expected_high_f"]

        result = parse_daily_high_forecast(
            fixture["forecast_response"], target, station
        )

        assert result.station == station
        assert result.target_date == target
        if expected is None:
            assert result.high_f is None
        else:
            assert result.high_f == pytest.approx(expected, abs=0.01)

    def test_temperature_unit_is_fahrenheit(self, name: str):
        # NWS contract has been "always F" forever, but we explicitly
        # handle C in the parser. If NWS ever flips a station to C this
        # test will fail loudly so we notice instead of silently
        # converting twice through the unit branch.
        fixture = _load(name)
        units = {
            p.get("temperatureUnit")
            for p in fixture["forecast_response"]["properties"]["periods"]
        }
        assert units <= {"F"}, (
            f"{name}: unexpected temperatureUnit values {units}; "
            "review the C-branch in parse_daily_high_forecast."
        )

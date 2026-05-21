"""Regression tests against snapshotted live NASA POWER responses.

Each fixture under ``tests/fixtures/power_*.json`` wraps a real response
from ``power.larc.nasa.gov/api/temporal/daily/point`` and embeds the
parser-computed ``expected_high_f`` from capture time so the test is
fully deterministic on the snapshot.

POWER's shape lock matters because the legacy CDO-like response was
already different from the modern Access Data Service (see NCEI for the
same class of bug). Pinning the live shape catches a future field
rename or units flip at PR-gate time.

To refresh: run ``python -m scripts.refresh_source_fixtures`` from repo root.
The script writes fixtures with explicit LF newlines so Windows captures
don't need a CRLF cleanup pass.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.fetchers.power import parse_daily_high

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _all_power_fixtures() -> list[Path]:
    return sorted(FIXTURES.glob("power_*.json"))


def test_at_least_one_power_fixture():
    assert len(_all_power_fixtures()) >= 1


@pytest.mark.parametrize(
    "name",
    [f.name for f in _all_power_fixtures()],
)
class TestLivePowerFixture:
    """One pytest case per captured fixture. New fixtures auto-extend."""

    def test_payload_has_t2m_max_path(self, name: str):
        fixture = _load(name)
        t2m = (
            fixture["payload"]
            .get("properties", {})
            .get("parameter", {})
            .get("T2M_MAX")
        )
        assert isinstance(t2m, dict), (
            f"{name}: properties.parameter.T2M_MAX is missing or not a "
            "dict; NASA POWER may have changed the payload shape."
        )

    def test_target_key_format_is_yyyymmdd(self, name: str):
        # POWER keys daily values by YYYYMMDD string. The parser depends
        # on this exact format — any change here breaks every call.
        fixture = _load(name)
        target = date.fromisoformat(fixture["captured_target_date"])
        t2m = fixture["payload"]["properties"]["parameter"]["T2M_MAX"]
        expected_key = target.strftime("%Y%m%d")
        assert expected_key in t2m, (
            f"{name}: expected {expected_key!r} in T2M_MAX keys; got {list(t2m.keys())[:5]}"
        )

    def test_value_is_celsius_not_fill(self, name: str):
        # POWER uses -999 / -9999 for missing. A healthy capture should
        # never produce a fill value, so this guards against the day a
        # quirky source suddenly returns -999 and we silently convert.
        fixture = _load(name)
        target = date.fromisoformat(fixture["captured_target_date"])
        value = fixture["payload"]["properties"]["parameter"]["T2M_MAX"][
            target.strftime("%Y%m%d")
        ]
        # The value can be int, float, or numeric string — all should
        # parse as a real number well above the -999 fill threshold.
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            pytest.fail(f"{name}: T2M_MAX value {value!r} is not numeric")
        assert numeric > -100, (
            f"{name}: T2M_MAX value {numeric} looks like a fill value; "
            "re-capture with a different date."
        )

    def test_parser_returns_captured_high(self, name: str):
        fixture = _load(name)
        target = date.fromisoformat(fixture["captured_target_date"])
        station = fixture["captured_for_station"]
        expected = fixture["expected_high_f"]

        result = parse_daily_high(fixture["payload"], target, station)

        assert result.station == station
        assert result.target_date == target
        assert result.source == "power"
        if expected is None:
            assert result.high_f is None
        else:
            assert result.high_f == pytest.approx(expected, abs=0.001)


def test_no_crlf_in_fixtures():
    # Fixtures must be LF-only so git diff --check stays clean and codex's
    # post-merge normalization step isn't needed every refresh.
    for fixture in _all_power_fixtures():
        raw = fixture.read_bytes()
        assert b"\r\n" not in raw, (
            f"{fixture.name} contains CRLF; re-capture using the script's "
            "explicit newline='\\n' open mode."
        )

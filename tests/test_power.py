"""Tests for src.fetchers.power — parser- and fetcher-focused, fully offline."""

from __future__ import annotations

from datetime import date

import pytest

from src.fetchers import power
from src.fetchers.power import PowerDailyHigh, fetch_daily_high, parse_daily_high

STATION = "nyc_lga"
TARGET = date(2025, 1, 2)
TARGET_KEY = "20250102"


def _payload(t2m_max: dict | None) -> dict:
    if t2m_max is None:
        return {"properties": {"parameter": {}}}
    return {"properties": {"parameter": {"T2M_MAX": t2m_max}}}


class TestParseDailyHigh:
    def test_celsius_converts_to_fahrenheit(self):
        result = parse_daily_high(_payload({TARGET_KEY: 20.0}), TARGET, STATION)
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
                    TARGET_KEY: 15.0,
                    "20250103": 12.0,
                }
            ),
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(59.0)

    def test_missing_target_returns_none(self):
        result = parse_daily_high(_payload({"20250103": 15.0}), TARGET, STATION)
        assert result.high_f is None

    def test_missing_parameter_shape_returns_none(self):
        assert parse_daily_high({}, TARGET, STATION).high_f is None
        assert parse_daily_high({"properties": {}}, TARGET, STATION).high_f is None
        assert (
            parse_daily_high({"properties": {"parameter": {}}}, TARGET, STATION).high_f is None
        )
        assert (
            parse_daily_high(
                {"properties": {"parameter": {"T2M_MAX": "oops"}}}, TARGET, STATION
            ).high_f
            is None
        )

    def test_fill_value_returns_none(self):
        assert (
            parse_daily_high(_payload({TARGET_KEY: -999.0}), TARGET, STATION).high_f
            is None
        )

    def test_extreme_fill_value_returns_none(self):
        # POWER occasionally uses -9999 in older payloads.
        assert (
            parse_daily_high(_payload({TARGET_KEY: -9999.0}), TARGET, STATION).high_f
            is None
        )

    def test_missing_markers_return_none(self):
        for marker in (None, "M", "NA", "NaN", "None"):
            assert (
                parse_daily_high(_payload({TARGET_KEY: marker}), TARGET, STATION).high_f
                is None
            )

    def test_bad_string_returns_none(self):
        assert parse_daily_high(_payload({TARGET_KEY: "abc"}), TARGET, STATION).high_f is None

    def test_numeric_string_value_accepted(self):
        result = parse_daily_high(_payload({TARGET_KEY: "20.0"}), TARGET, STATION)
        assert result.high_f == pytest.approx(68.0)

    def test_properties_not_dict_returns_none(self):
        assert parse_daily_high({"properties": "oops"}, TARGET, STATION).high_f is None

    def test_non_dict_payload_returns_none(self):
        for bad in (None, "oops", 42, []):
            result = parse_daily_high(bad, TARGET, STATION)  # type: ignore[arg-type]
            assert result.high_f is None

    def test_dataclass_metadata_preserved(self):
        result = parse_daily_high(_payload({TARGET_KEY: 0.0}), TARGET, STATION)
        assert result.target_date == TARGET
        assert result.station == STATION
        assert result.source == "power"


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self.calls: list[tuple[str, dict | None]] = []
        self._payload = payload

    def __call__(self, timeout):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None, headers=None):
        self.calls.append((url, params))
        return _FakeResponse(self._payload)


class TestFetchDailyHigh:
    def test_validates_params(self, monkeypatch):
        client = _FakeClient(_payload({TARGET_KEY: 20.0}))
        monkeypatch.setattr(power.httpx, "Client", client)

        result = fetch_daily_high(40.0, -75.0, TARGET, STATION)
        assert result.high_f == pytest.approx(68.0)

        url, params = client.calls[0]
        assert url == "https://power.larc.nasa.gov/api/temporal/daily/point"
        assert params["parameters"] == "T2M_MAX"
        assert params["start"] == TARGET_KEY
        assert params["end"] == TARGET_KEY
        assert params["latitude"] == 40.0
        assert params["longitude"] == -75.0
        assert params["format"] == "JSON"

    def test_http_error_propagates(self, monkeypatch):
        class _BoomResponse:
            def raise_for_status(self):
                raise RuntimeError("power 500")

            def json(self):
                raise AssertionError

        class _BoomClient:
            def __call__(self, timeout):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, *a, **k):
                return _BoomResponse()

        monkeypatch.setattr(power.httpx, "Client", _BoomClient())
        with pytest.raises(RuntimeError, match="power 500"):
            fetch_daily_high(40.0, -75.0, TARGET, STATION)

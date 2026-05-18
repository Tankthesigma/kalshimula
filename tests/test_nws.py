"""Tests for src.fetchers.nws — parser- and fetcher-focused, fully offline."""

from __future__ import annotations

from datetime import date

import pytest

from src.config import Station
from src.fetchers import nws
from src.fetchers.nws import (
    NwsDailyHighForecast,
    fetch_daily_high_forecast,
    forecast_url_from_points_payload,
    parse_daily_high_forecast,
    resolve_forecast_url,
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
    unit: str | None = "F",
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


def _station(*, nws_station: str = "KDEN") -> Station:
    return Station(
        slug="denver",
        name="Denver",
        nws_station=nws_station,
        ghcnd_id="GHCND:USW00003017",
        lat=39.8328,
        lon=-104.6575,
        tz="America/Denver",
        lst_offset_hours=-7,
    )


class _FakeResponse:
    def __init__(self, payload: object):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _FakeClient:
    """Records every GET, returns a queue of payloads in FIFO order."""

    def __init__(self, payloads: list[object]):
        self.calls: list[tuple[str, dict | None]] = []
        self._payloads = list(payloads)

    def __call__(self, timeout):  # used as httpx.Client(timeout=...)
        self.timeout = timeout
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None, params=None):
        self.calls.append((url, headers))
        return _FakeResponse(self._payloads.pop(0))


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

    def test_missing_unit_treated_as_fahrenheit(self):
        result = parse_daily_high_forecast(
            _payload(_period(temperature=66, unit=None)),
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(66.0)

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
        assert parse_daily_high_forecast({}, TARGET, STATION).high_f is None
        assert parse_daily_high_forecast({"properties": {}}, TARGET, STATION).high_f is None
        assert (
            parse_daily_high_forecast(
                {"properties": {"periods": []}}, TARGET, STATION
            ).high_f
            is None
        )

    def test_non_dict_payload_returns_none(self):
        # Defensive: a string, list, None, or arbitrary object must not raise.
        for bad in (None, "oops", 42, [1, 2, 3]):
            result = parse_daily_high_forecast(bad, TARGET, STATION)  # type: ignore[arg-type]
            assert result.high_f is None
            assert result.station == STATION
            assert result.target_date == TARGET

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


class TestForecastUrlFromPointsPayload:
    def test_extracts_url(self):
        payload = {"properties": {"forecast": "https://example.test/forecast"}}
        assert forecast_url_from_points_payload(payload) == "https://example.test/forecast"

    def test_missing_properties_raises(self):
        with pytest.raises(ValueError):
            forecast_url_from_points_payload({})

    def test_missing_forecast_field_raises(self):
        with pytest.raises(ValueError):
            forecast_url_from_points_payload({"properties": {}})

    def test_non_http_url_raises(self):
        with pytest.raises(ValueError):
            forecast_url_from_points_payload({"properties": {"forecast": "/relative/path"}})

    def test_non_dict_payload_raises(self):
        with pytest.raises(ValueError):
            forecast_url_from_points_payload("oops")  # type: ignore[arg-type]


class TestResolveForecastUrl:
    def test_calls_points_endpoint(self, monkeypatch):
        client = _FakeClient([{"properties": {"forecast": "https://example.test/fc"}}])
        monkeypatch.setattr(nws.httpx, "Client", client)
        url = resolve_forecast_url(_station())
        assert url == "https://example.test/fc"
        assert client.calls[0][0] == "https://api.weather.gov/points/39.8328,-104.6575"
        headers = client.calls[0][1]
        assert "User-Agent" in headers and "Accept" in headers


class TestFetchDailyHighForecast:
    def test_full_flow_uses_points_then_forecast(self, monkeypatch):
        points_payload = {"properties": {"forecast": "https://example.test/forecast"}}
        forecast_payload = _payload(_period(temperature=72))
        client = _FakeClient([points_payload, forecast_payload])
        monkeypatch.setattr(nws.httpx, "Client", client)

        result = fetch_daily_high_forecast(_station(nws_station="KDEN"), TARGET)

        assert result.high_f == pytest.approx(72.0)
        # First call: /points. Second call: the resolved forecast URL.
        assert len(client.calls) == 2
        assert client.calls[0][0].startswith("https://api.weather.gov/points/")
        assert client.calls[1][0] == "https://example.test/forecast"

    def test_skips_points_when_station_is_full_url(self, monkeypatch):
        forecast_payload = _payload(_period(temperature=72))
        client = _FakeClient([forecast_payload])
        monkeypatch.setattr(nws.httpx, "Client", client)

        url_station = _station(
            nws_station="https://api.weather.gov/gridpoints/BOU/1,2/forecast"
        )
        result = fetch_daily_high_forecast(url_station, TARGET)

        assert result.high_f == pytest.approx(72.0)
        assert len(client.calls) == 1
        assert client.calls[0][0] == url_station.nws_station

    def test_http_error_propagates(self, monkeypatch):
        class _BoomResponse:
            def raise_for_status(self):
                raise RuntimeError("nws 500")

            def json(self):
                raise AssertionError("should not be called")

        class _BoomClient:
            def __call__(self, timeout):
                return self

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get(self, *a, **k):
                return _BoomResponse()

        boom = _BoomClient()
        monkeypatch.setattr(nws.httpx, "Client", boom)
        with pytest.raises(RuntimeError, match="nws 500"):
            fetch_daily_high_forecast(_station(nws_station=STATION), TARGET)

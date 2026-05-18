from datetime import date

import pytest

from src.config import Station
from src.fetchers import nws
from src.fetchers.nws import (
    forecast_url_from_points_payload,
    parse_daily_high_forecast,
    resolve_forecast_url,
)


def test_parse_daily_high_forecast_returns_matching_daytime_high() -> None:
    payload = {
        "properties": {
            "periods": [
                {
                    "startTime": "2025-01-01T06:00:00-07:00",
                    "isDaytime": True,
                    "temperature": 70,
                    "temperatureUnit": "F",
                }
            ]
        }
    }

    forecast = parse_daily_high_forecast(payload, date(2025, 1, 1), "KDEN")

    assert forecast.high_f == 70


def test_forecast_url_from_points_payload_extracts_url() -> None:
    payload = {"properties": {"forecast": "https://api.weather.gov/gridpoints/BOU/1,2/forecast"}}

    assert forecast_url_from_points_payload(payload) == "https://api.weather.gov/gridpoints/BOU/1,2/forecast"


def test_forecast_url_from_points_payload_rejects_missing_url() -> None:
    with pytest.raises(ValueError):
        forecast_url_from_points_payload({"properties": {}})


def test_resolve_forecast_url_uses_points_endpoint(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"properties": {"forecast": "https://example.test/forecast"}}

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def get(self, url, headers):
            calls.append((url, headers))
            return FakeResponse()

    monkeypatch.setattr(nws.httpx, "Client", FakeClient)
    station = Station(
        slug="denver",
        name="Denver",
        nws_station="KDEN",
        ghcnd_id="GHCND:USW00003017",
        lat=39.8328,
        lon=-104.6575,
        tz="America/Denver",
        lst_offset_hours=-7,
    )

    assert resolve_forecast_url(station) == "https://example.test/forecast"
    assert calls[0][0] == "https://api.weather.gov/points/39.8328,-104.6575"

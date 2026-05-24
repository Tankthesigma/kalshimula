from datetime import date

from src.config import Station
from src.models import nws_guidance
from src.models.nws_guidance import (
    fetch_nws_guidance_rows,
    guidance_rows_from_nws_forecast_payload,
)


def _payload() -> dict:
    return {
        "properties": {
            "generatedAt": "2026-05-24T06:30:00Z",
            "periods": [
                {
                    "name": "Today",
                    "startTime": "2026-05-24T06:00:00-04:00",
                    "endTime": "2026-05-24T18:00:00-04:00",
                    "isDaytime": True,
                    "temperature": 70,
                    "temperatureUnit": "F",
                },
                {
                    "name": "Late Afternoon",
                    "startTime": "2026-05-24T15:00:00-04:00",
                    "endTime": "2026-05-24T20:00:00-04:00",
                    "isDaytime": True,
                    "temperature": 73,
                    "temperatureUnit": "F",
                },
                {
                    "name": "Tonight",
                    "startTime": "2026-05-24T20:00:00-04:00",
                    "endTime": "2026-05-25T06:00:00-04:00",
                    "isDaytime": False,
                    "temperature": 55,
                    "temperatureUnit": "F",
                },
            ],
        }
    }


def test_guidance_rows_from_nws_forecast_payload_normalizes_row() -> None:
    rows = guidance_rows_from_nws_forecast_payload(
        _payload(),
        city="NYC",
        station_id="KNYC",
        target=date(2026, 5, 24),
        fetched_at="2026-05-24T07:00:00Z",
    )

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["city"] == "nyc"
    assert row["source"] == "nws_forecast"
    assert row["station_id"] == "KNYC"
    assert row["guidance_point_f"] == 73
    assert row["guidance_q50_f"] == 73
    assert row["issue_ts_utc"] == "2026-05-24T06:30:00+00:00"
    assert row["available_ts_utc"] == "2026-05-24T06:30:00+00:00"
    assert row["valid_ts_utc"] == "2026-05-25T00:00:00+00:00"
    assert isinstance(row["raw_payload_hash"], str)


def test_guidance_rows_from_nws_forecast_payload_empty_when_no_target_period() -> None:
    rows = guidance_rows_from_nws_forecast_payload(
        _payload(),
        city="nyc",
        station_id="KNYC",
        target=date(2026, 5, 25),
        fetched_at="2026-05-24T07:00:00Z",
    )

    assert rows.empty


def test_fetch_nws_guidance_rows_uses_configured_stations(monkeypatch) -> None:
    station = Station(
        slug="nyc",
        name="New York City",
        nws_station="KNYC",
        ghcnd_id="GHCND:USW00094728",
        lat=40.7,
        lon=-73.9,
        tz="America/New_York",
        lst_offset_hours=-5,
    )

    def fake_fetch_forecast_payload(received_station):
        assert received_station == station
        return _payload(), "https://example.test/forecast"

    monkeypatch.setattr(
        nws_guidance,
        "fetch_forecast_payload",
        fake_fetch_forecast_payload,
    )

    rows = fetch_nws_guidance_rows(
        {"nyc": station},
        target=date(2026, 5, 24),
        cities=["nyc"],
        fetched_at="2026-05-24T07:00:00Z",
    )

    assert rows["city"].tolist() == ["nyc"]
    assert rows["guidance_point_f"].tolist() == [73]

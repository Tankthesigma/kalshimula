from datetime import datetime

import pandas as pd

from src.fetchers.asos import AsosHourlyObservation
from src.models.nowcast_features import (
    build_nowcast_features,
    fetch_observations_for_rules,
    observations_to_frame,
    render_nowcast_feature_report,
)
from src.models.station_rules import station_rule_by_key


def test_observations_to_frame_uses_canonical_columns() -> None:
    frame = observations_to_frame(
        [
            AsosHourlyObservation(
                station="KMDW",
                valid_time=datetime(2026, 5, 24, 14, 53),
                temp_f=70.0,
                dewpoint_f=55.0,
                wind_speed_kt=8.0,
                cloud_cover="CLR",
            )
        ]
    )

    assert frame.loc[0, "station_id"] == "KMDW"
    assert frame.loc[0, "temperature_f"] == 70.0
    assert frame.loc[0, "dewpoint_f"] == 55.0
    assert frame.loc[0, "cloud_cover"] == "CLR"


def test_fetch_observations_for_rules_degrades_failed_station(monkeypatch) -> None:
    def fake_fetch(station, start, end):
        if station == "KMDW":
            raise RuntimeError("rate limited")
        return "station,valid,tmpf\nKNYC,2026-05-24 10:00,70\n"

    monkeypatch.setattr(
        "src.models.nowcast_features.fetch_asos_observation_csv",
        fake_fetch,
    )

    observations = fetch_observations_for_rules(
        [station_rule_by_key(city="chicago"), station_rule_by_key(city="nyc")],
        start=datetime(2026, 5, 24).date(),
        end=datetime(2026, 5, 24).date(),
    )

    assert observations["station_id"].tolist() == ["KNYC"]
    assert observations["temperature_f"].tolist() == [70.0]


def test_nowcast_features_use_only_observations_available_by_as_of() -> None:
    rule = station_rule_by_key(city="chicago")
    observations = pd.DataFrame(
        [
            _obs("KMDW", "2026-05-24T13:00:00", 70),
            _obs("KMDW", "2026-05-24T14:00:00", 72),
            _obs("KMDW", "2026-05-24T16:00:00", 99),
        ]
    )

    features = build_nowcast_features(
        observations,
        [rule],
        target_date=datetime(2026, 5, 24).date(),
        as_of_ts=datetime(2026, 5, 24, 14, 30),
        decision_time_label="10",
    )

    row = features.iloc[0]
    assert row["latest_temp_f"] == 72
    assert row["high_so_far_f"] == 72
    assert row["low_so_far_f"] == 70
    assert row["latest_obs_ts_utc"] == "2026-05-24T14:00:00"


def test_nowcast_features_flags_missing_observations() -> None:
    rule = station_rule_by_key(city="chicago")

    features = build_nowcast_features(
        pd.DataFrame(columns=["station_id", "obs_ts_utc", "available_ts_utc", "temperature_f"]),
        [rule],
        target_date=datetime(2026, 5, 24).date(),
        as_of_ts=datetime(2026, 5, 24, 14, 30),
        decision_time_label="10",
    )

    row = features.iloc[0]
    assert row["nowcast_veto_flag"]
    assert "missing_observations" in row["weather_reason_codes"]


def test_nowcast_report_is_weather_only() -> None:
    rule = station_rule_by_key(city="chicago")
    features = build_nowcast_features(
        pd.DataFrame([_obs("KMDW", "2026-05-24T14:00:00", 72)]),
        [rule],
        target_date=datetime(2026, 5, 24).date(),
        as_of_ts=datetime(2026, 5, 24, 14, 30),
        decision_time_label="10",
    )

    report = render_nowcast_feature_report(features)

    assert "Weather-only" in report
    assert "market prices" in report
    assert "chicago" in report


def _obs(station: str, timestamp: str, temp: float) -> dict:
    return {
        "station_id": station,
        "obs_ts_utc": timestamp,
        "available_ts_utc": timestamp,
        "temperature_f": temp,
        "dewpoint_f": 55.0,
        "wind_speed_kt": 8.0,
        "wind_direction_deg": 180.0,
        "gust_kt": None,
        "cloud_cover": "CLR",
        "pressure_mb": 1012.0,
        "precip_in": 0.0,
        "source": "asos",
    }

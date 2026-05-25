from datetime import datetime

import pandas as pd

from src.fetchers.asos import AsosHourlyObservation
from src.models.nowcast_features import (
    build_nowcast_features,
    build_observation_coverage,
    fetch_observations_for_rules,
    load_observation_store,
    merge_observation_store,
    observations_to_frame,
    render_nowcast_feature_report,
    write_nowcast_features,
    write_observation_store,
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
    assert frame.loc[0, "obs_ts_utc"] == "2026-05-24T14:53:00"
    assert frame.loc[0, "available_ts_utc"] == "2026-05-24T15:03:00"
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


def test_observation_store_merges_and_deduplicates(tmp_path) -> None:
    existing = pd.DataFrame(
        [
            _obs("KMDW", "2026-05-24T13:00:00", 70),
            _obs("KMDW", "2026-05-24T14:00:00", 71),
        ]
    )
    newer = pd.DataFrame(
        [
            _obs("KMDW", "2026-05-24T14:00:00", 72),
            _obs("KNYC", "2026-05-24T14:00:00", 65),
        ]
    )

    merged = merge_observation_store(existing, newer)

    assert len(merged) == 3
    chicago_14 = merged[
        (merged["station_id"] == "KMDW")
        & (merged["obs_ts_utc"] == "2026-05-24T14:00:00")
    ].iloc[0]
    assert chicago_14["temperature_f"] == 72

    store = tmp_path / "observations.csv"
    write_observation_store(store, merged)
    loaded = load_observation_store(store)
    assert loaded["station_id"].tolist() == ["KMDW", "KMDW", "KNYC"]


def test_write_nowcast_features_can_read_and_update_observation_store(tmp_path) -> None:
    store = tmp_path / "observations.csv"
    write_observation_store(store, pd.DataFrame([_obs("KMDW", "2026-05-24T14:00:00", 72)]))

    result = write_nowcast_features(
        output_dir=tmp_path / "out",
        target_date=datetime(2026, 5, 24).date(),
        as_of_ts=datetime(2026, 5, 24, 14, 30),
        decision_time_label="10",
        observation_store_path=store,
        update_observation_store=True,
    )

    chicago = result.features[result.features["city"] == "chicago"].iloc[0]
    assert chicago["latest_temp_f"] == 72
    assert result.manifest["observation_store_updated"] is True
    assert store.exists()


def test_write_nowcast_features_can_target_low_market_rules(tmp_path) -> None:
    store = tmp_path / "observations.csv"
    write_observation_store(store, pd.DataFrame([_obs("KMDW", "2026-05-24T14:00:00", 72)]))

    result = write_nowcast_features(
        output_dir=tmp_path / "out",
        target_date=datetime(2026, 5, 24).date(),
        as_of_ts=datetime(2026, 5, 24, 14, 30),
        decision_time_label="10",
        observation_store_path=store,
        market_types=["low"],
    )

    chicago = result.features[result.features["city"] == "chicago"].iloc[0]
    assert chicago["market_type"] == "low"
    assert chicago["station_id"] == "KMDW"


def test_write_nowcast_features_can_filter_cities(tmp_path) -> None:
    store = tmp_path / "observations.csv"
    write_observation_store(
        store,
        pd.DataFrame(
            [
                _obs("KMDW", "2026-05-24T14:00:00", 72),
                _obs("KNYC", "2026-05-24T14:00:00", 65),
            ]
        ),
    )

    result = write_nowcast_features(
        output_dir=tmp_path / "out",
        target_date=datetime(2026, 5, 24).date(),
        as_of_ts=datetime(2026, 5, 24, 14, 30),
        decision_time_label="10",
        observation_store_path=store,
        cities=["nyc"],
    )

    assert result.features["city"].tolist() == ["nyc"]
    assert result.features["station_id"].tolist() == ["KNYC"]
    assert result.manifest["cities"] == ["nyc"]


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
    assert row["as_of_ts_utc"] == "2026-05-24T14:30:00+00:00"
    assert row["latest_obs_ts_utc"] == "2026-05-24T14:00:00"


def test_nowcast_features_filter_target_day_by_station_settlement_date() -> None:
    rule = station_rule_by_key(city="phoenix")
    observations = pd.DataFrame(
        [
            _obs("KPHX", "2026-05-04T00:51:00", 94),  # May 3 17:51 Phoenix time.
            _obs("KPHX", "2026-05-04T14:51:00", 75),
            _obs("KPHX", "2026-05-04T20:51:00", 79),
        ]
    )

    features = build_nowcast_features(
        observations,
        [rule],
        target_date=datetime(2026, 5, 4).date(),
        as_of_ts=datetime(2026, 5, 4, 21, 30),
        decision_time_label="15",
    )
    coverage = build_observation_coverage(
        observations,
        [rule],
        target_date=datetime(2026, 5, 4).date(),
        as_of_ts=datetime(2026, 5, 4, 21, 30),
        decision_time_label="15",
    )

    assert features.iloc[0]["latest_temp_f"] == 79
    assert features.iloc[0]["high_so_far_f"] == 79
    assert coverage.iloc[0]["high_so_far_f"] == 79


def test_observation_coverage_flags_sparse_or_stale_store() -> None:
    rule = station_rule_by_key(city="chicago")
    observations = pd.DataFrame([_obs("KMDW", "2026-05-24T06:00:00", 60)])

    coverage = build_observation_coverage(
        observations,
        [rule],
        target_date=datetime(2026, 5, 24).date(),
        as_of_ts=datetime(2026, 5, 24, 14, 30),
        decision_time_label="10",
    )

    row = coverage.iloc[0]
    assert row["obs_count_available"] == 1
    assert row["temp_obs_count_available"] == 1
    assert not bool(row["coverage_ok"])
    assert "stale_observation" in row["coverage_reason_codes"]
    assert "thin_temperature_coverage" in row["coverage_reason_codes"]


def test_write_nowcast_features_writes_observation_coverage(tmp_path) -> None:
    store = tmp_path / "observations.csv"
    write_observation_store(
        store,
        pd.DataFrame(
            [
                _obs("KNYC", "2026-05-24T12:00:00", 60),
                _obs("KNYC", "2026-05-24T13:00:00", 62),
                _obs("KNYC", "2026-05-24T14:00:00", 64),
            ]
        ),
    )

    result = write_nowcast_features(
        output_dir=tmp_path / "out",
        target_date=datetime(2026, 5, 24).date(),
        as_of_ts=datetime(2026, 5, 24, 14, 30),
        decision_time_label="10",
        observation_store_path=store,
        cities=["nyc"],
    )

    coverage_path = tmp_path / "out" / "observation_coverage.csv"
    assert coverage_path.exists()
    assert result.coverage["city"].tolist() == ["nyc"]
    assert result.manifest["row_counts"]["coverage"] == 1
    written = pd.read_csv(coverage_path)
    assert written["station_id"].tolist() == ["KNYC"]


def test_nowcast_features_accept_epoch_second_observation_timestamps() -> None:
    rule = station_rule_by_key(city="chicago")
    observations = pd.DataFrame(
        [
            {
                **_obs("KMDW", "2026-05-24T14:00:00", 72),
                "obs_ts_utc": 1779631200,
                "available_ts_utc": 1779631200,
            }
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
    assert row["latest_obs_ts_utc"] == "2026-05-24T14:00:00"


def test_nowcast_features_accept_epoch_millisecond_observation_timestamps() -> None:
    rule = station_rule_by_key(city="chicago")
    observations = pd.DataFrame(
        [
            {
                **_obs("KMDW", "2026-05-24T14:00:00", 72),
                "obs_ts_utc": 1779631200000,
                "available_ts_utc": 1779631200000,
            }
        ]
    )

    features = build_nowcast_features(
        observations,
        [rule],
        target_date=datetime(2026, 5, 24).date(),
        as_of_ts=datetime(2026, 5, 24, 14, 30),
        decision_time_label="10",
    )

    assert features.iloc[0]["latest_obs_ts_utc"] == "2026-05-24T14:00:00"


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

import json

import pandas as pd

from src.models.nowcast_adjustment import apply_nowcast_adjustments


def _prediction_rows() -> pd.DataFrame:
    rows = []
    pmf = {"69": 0.25, "70": 0.5, "71": 0.25}
    for degree, probability in [(69, 0.25), (70, 0.5), (71, 0.25)]:
        rows.append(
            {
                "model_version": "test",
                "city": "nyc",
                "platform": "kalshi",
                "market_type": "high",
                "station_id": "KNYC",
                "target_date": "2026-05-24",
                "prediction_ts_utc": "2026-05-24T14:00:00+00:00",
                "prediction_time_local": "2026-05-24T10:00:00-04:00",
                "decision_time_label": "10",
                "as_of_ts_utc": "2026-05-24T14:00:00+00:00",
                "bin_lower_f": degree,
                "bin_upper_f": degree,
                "bin_label": str(degree),
                "model_probability": probability,
                "calibrated_probability": probability,
                "point_f": 70,
                "q05_f": 69,
                "q10_f": 69,
                "q20_f": 69,
                "q25_f": 69,
                "q30_f": 70,
                "q40_f": 70,
                "q50_f": 70,
                "q60_f": 70,
                "q70_f": 70,
                "q75_f": 70,
                "q80_f": 71,
                "q90_f": 71,
                "q95_f": 71,
                "pmf_degree_json": json.dumps(pmf),
                "source_policy": "gfs_ens",
                "nowcast_veto_flag": False,
                "weather_reason_codes": "",
                "station_rule_confidence": "high",
                "source_independence_score": 1.0,
                "feature_hash": "feature",
            }
        )
    return pd.DataFrame(rows)


def _features(high_so_far: float | None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": "nyc",
                "platform": "kalshi",
                "market_type": "high",
                "station_id": "KNYC",
                "target_date": "2026-05-24",
                "prediction_ts_utc": "2026-05-24T14:00:00+00:00",
                "prediction_time_local": "2026-05-24T10:00:00-04:00",
                "decision_time_label": "10",
                "as_of_ts_utc": "2026-05-24T14:00:00+00:00",
                "latest_obs_ts_utc": "2026-05-24T13:55:00+00:00",
                "latest_temp_f": high_so_far,
                "latest_dewpoint_f": 55,
                "high_so_far_f": high_so_far,
                "low_so_far_f": 60,
                "temp_1h_slope_f": 1,
                "temp_3h_slope_f": 3,
                "dewpoint_depression_f": 17,
                "wind_speed_kt": 10,
                "cloud_cover": "CLR",
                "hours_since_sunrise": 4,
                "hours_to_solar_noon": 2,
                "hours_to_sunset": 8,
                "radiative_cooling_index": 0.1,
                "remaining_heating_estimate_f": 3,
                "remaining_cooling_estimate_f": 0,
                "nowcast_veto_flag": False,
                "weather_reason_codes": "",
                "station_rule_confidence": "high",
                "feature_hash": "feature",
            }
        ]
    )


def test_adjustment_truncates_high_pmf_below_high_so_far() -> None:
    adjusted = apply_nowcast_adjustments(_prediction_rows(), _features(70.6))

    assert adjusted["bin_lower_f"].tolist() == [71]
    assert adjusted["calibrated_probability"].tolist() == [1.0]
    assert adjusted.iloc[0]["point_f"] == 71.0
    assert "pmf_truncated_below_high_so_far:71" in adjusted.iloc[0]["weather_reason_codes"]
    assert json.loads(adjusted.iloc[0]["pmf_degree_json"]) == {"71": 1.0}


def test_adjustment_collapses_when_high_so_far_above_support() -> None:
    adjusted = apply_nowcast_adjustments(_prediction_rows(), _features(73.0))

    assert adjusted["bin_lower_f"].tolist() == [73]
    assert adjusted["calibrated_probability"].tolist() == [1.0]
    assert adjusted["model_probability"].tolist() == [0.0]


def test_adjustment_keeps_rows_without_high_so_far() -> None:
    adjusted = apply_nowcast_adjustments(_prediction_rows(), _features(None))

    assert adjusted["bin_lower_f"].tolist() == [69, 70, 71]
    assert adjusted["calibrated_probability"].tolist() == [0.25, 0.5, 0.25]

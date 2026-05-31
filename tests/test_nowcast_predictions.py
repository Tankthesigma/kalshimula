import json

import pandas as pd
import pytest

from src.models.nowcast_predictions import build_nowcast_prediction_rows
from src.models.station_rules import StationRule


def _rule(market_type: str = "high") -> StationRule:
    return StationRule(
        city="nyc",
        platform="kalshi",
        market_type=market_type,
        settlement_station="KNYC",
        station_name="Central Park",
        timezone="America/New_York",
        lst_offset=-5,
        dst_policy="lst_year_round",
        unit="F",
        rounding_rule="whole_degree",
        settlement_source="test",
        rule_confidence="high",
    )


def _payload() -> dict:
    return {
        "predictions": [
            {
                "city": "nyc",
                "target_date": "2026-05-24",
                "generated_at": "2026-05-24T14:00:00+00:00",
                "selected_source": "gfs_ens",
                "selected_source_applied": True,
                "forecast": {
                    "point_f": 70.0,
                    "bin_probabilities": {"69": 0.25, "70": 0.5, "71": 0.25},
                },
                "calibration": {
                    "corrected_point_f": 71.0,
                    "bias_correction_f": 1.0,
                },
            }
        ]
    }


def test_build_nowcast_prediction_rows_uses_corrected_pmf() -> None:
    rows = build_nowcast_prediction_rows(
        _payload(),
        station_rules=[_rule()],
        decision_time_label="10",
        as_of_ts_utc="2026-05-24T14:00:00Z",
    )

    assert list(rows["bin_lower_f"]) == [70, 71, 72]
    assert rows["calibrated_probability"].tolist() == [0.25, 0.5, 0.25]
    assert rows.loc[rows["bin_lower_f"] == 70, "model_probability"].item() == 0.5
    assert set(rows["source_policy"]) == {"gfs_ens"}
    pmf = json.loads(rows.iloc[0]["pmf_degree_json"])
    assert pmf == {"70": 0.25, "71": 0.5, "72": 0.25}


def test_build_nowcast_prediction_rows_carries_weather_only_feature_flags() -> None:
    features = pd.DataFrame(
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
                "latest_temp_f": 72,
                "latest_dewpoint_f": 55,
                "high_so_far_f": 72,
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
                "nowcast_veto_flag": True,
                "weather_reason_codes": "high_so_far_exceeds_model_point",
                "station_rule_confidence": "high",
                "feature_hash": "abc123",
            }
        ]
    )

    rows = build_nowcast_prediction_rows(
        _payload(),
        features=features,
        station_rules=[_rule()],
        decision_time_label="morning",
    )

    assert set(rows["decision_time_label"]) == {"10"}
    assert set(rows["nowcast_veto_flag"]) == {True}
    assert set(rows["weather_reason_codes"]) == {"high_so_far_exceeds_model_point"}
    assert set(rows["feature_hash"]) == {"abc123"}


def test_build_nowcast_prediction_rows_reports_fallback_source_when_selected_not_applied() -> None:
    payload = _payload()
    payload["predictions"][0]["selected_source_applied"] = False

    rows = build_nowcast_prediction_rows(
        payload,
        station_rules=[_rule()],
        decision_time_label="10",
        as_of_ts_utc="2026-05-24T14:00:00Z",
    )

    assert set(rows["source_policy"]) == {"openmeteo_naive"}
    assert set(rows["weather_reason_codes"]) == {"selected_source_fallback"}


def test_build_nowcast_prediction_rows_rejects_low_market_rows() -> None:
    with pytest.raises(ValueError, match="supports only high-temperature"):
        build_nowcast_prediction_rows(
            _payload(),
            station_rules=[_rule("low")],
            decision_time_label="evening",
            as_of_ts_utc="2026-05-24T23:00:00Z",
            market_type="low",
        )

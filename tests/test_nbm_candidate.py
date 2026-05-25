import pandas as pd

from src.models.nbm_candidate import build_nbm_candidate_predictions
from src.models.nowcast_predictions import NOWCAST_PREDICTION_COLUMNS


def test_build_nbm_candidate_predictions_emits_frozen_schema_pmf() -> None:
    base = {column: "" for column in NOWCAST_PREDICTION_COLUMNS}
    base.update(
        {
            "model_version": "raw",
            "city": "nyc",
            "platform": "kalshi",
            "market_type": "high",
            "station_id": "KNYC",
            "target_date": "2026-05-25",
            "prediction_ts_utc": "2026-05-25T18:00:00+00:00",
            "prediction_time_local": "2026-05-25T13:00:00-04:00",
            "decision_time_label": "13",
            "as_of_ts_utc": "2026-05-25T18:00:00+00:00",
            "bin_lower_f": 72,
            "bin_upper_f": 72,
            "bin_label": "72",
            "model_probability": 1.0,
            "calibrated_probability": 1.0,
            "point_f": 72,
            "pmf_degree_json": '{"72": 1.0}',
            "source_policy": "gfs_ens",
            "weather_reason_codes": "raw",
            "source_independence_score": 1.0,
        }
    )
    raw = pd.DataFrame([base], columns=NOWCAST_PREDICTION_COLUMNS)
    guidance = pd.DataFrame(
        [
            {
                "city": "nyc",
                "source": "nbm_text",
                "station_id": "KNYC",
                "market_type": "high",
                "target_date": "2026-05-25",
                "issue_ts_utc": "2026-05-25T13:00:00+00:00",
                "valid_ts_utc": "2026-05-26T01:00:00+00:00",
                "available_ts_utc": "2026-05-25T13:00:00+00:00",
                "guidance_point_f": 74.0,
                "guidance_q10_f": 70.0,
                "guidance_q50_f": 74.0,
                "guidance_q90_f": 78.0,
                "actual_high_f": pd.NA,
                "raw_payload_hash": "abc",
                "as_of_ts_utc": "2026-05-25T18:00:00+00:00",
            }
        ]
    )

    candidate = build_nbm_candidate_predictions(
        raw_predictions=raw,
        latest_guidance=guidance,
    )

    assert list(candidate.columns) == NOWCAST_PREDICTION_COLUMNS
    assert candidate["source_policy"].unique().tolist() == ["nbm_text"]
    assert candidate["model_version"].unique().tolist() == ["nbm-text-candidate-v1"]
    assert "nbm_guidance_candidate" in candidate.iloc[0]["weather_reason_codes"]
    assert abs(candidate["calibrated_probability"].sum() - 1.0) < 1e-9
    assert 73.0 <= candidate["point_f"].iloc[0] <= 75.0

import json
from pathlib import Path

import pandas as pd

from src.nowcast_predictions_cli import main


def test_nowcast_predictions_cli_writes_expected_files(tmp_path: Path, capsys) -> None:
    predictions = tmp_path / "predictions.json"
    features = tmp_path / "features.csv"
    out_dir = tmp_path / "out"
    predictions.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "city": "chicago",
                        "target_date": "2026-05-24",
                        "generated_at": "2026-05-24T15:00:00+00:00",
                        "selected_source": "gfs_ens",
                        "selected_source_applied": True,
                        "forecast": {
                            "point_f": 75.0,
                            "bin_probabilities": {"74": 0.4, "75": 0.6},
                        },
                        "calibration": {"corrected_point_f": 75.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "city": "chicago",
                "platform": "kalshi",
                "market_type": "high",
                "station_id": "KMDW",
                "target_date": "2026-05-24",
                "prediction_ts_utc": "2026-05-24T15:00:00+00:00",
                "prediction_time_local": "2026-05-24T10:00:00-05:00",
                "decision_time_label": "07",
                "as_of_ts_utc": "2026-05-24T15:00:00+00:00",
                "latest_obs_ts_utc": "2026-05-24T14:55:00+00:00",
                "latest_temp_f": 72,
                "latest_dewpoint_f": 55,
                "high_so_far_f": 72,
                "low_so_far_f": 60,
                "latest_minus_high_so_far_f": 0,
                "latest_minus_low_so_far_f": 12,
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
    ).to_csv(features, index=False)

    exit_code = main(
        [
            "--predictions-json",
            str(predictions),
            "--nowcast-features",
            str(features),
            "--decision-time-label",
            "fallback",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "predictions_nowcast.csv").exists()
    assert (out_dir / "predictions_nowcast_manifest.json").exists()
    rows = pd.read_csv(out_dir / "predictions_nowcast.csv")
    assert rows["station_id"].tolist() == ["KMDW", "KMDW"]
    raw_csv = (out_dir / "predictions_nowcast.csv").read_text(encoding="utf-8")
    assert ",07,2026-05-24T15:00:00+00:00," in raw_csv
    assert "Wrote 2 nowcast prediction rows" in capsys.readouterr().out

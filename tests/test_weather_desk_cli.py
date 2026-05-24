import json
from pathlib import Path

import pandas as pd

from src.weather_desk_cli import main


def test_weather_desk_cli_writes_end_to_end_packet(tmp_path: Path, capsys) -> None:
    predictions = tmp_path / "predictions.json"
    observations = tmp_path / "observations.csv"
    out_dir = tmp_path / "weather_desk"
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
                            "point_f": 74.0,
                            "bin_probabilities": {"73": 0.5, "75": 0.5},
                        },
                        "calibration": {"corrected_point_f": 74.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "station_id": "KMDW",
                "obs_ts_utc": "2026-05-24T14:00:00",
                "available_ts_utc": "2026-05-24T14:00:00",
                "temperature_f": 75,
                "dewpoint_f": 55,
                "wind_speed_kt": 8,
                "wind_direction_deg": 180,
                "gust_kt": None,
                "cloud_cover": "CLR",
                "pressure_mb": 1012,
                "precip_in": 0,
                "source": "asos",
            }
        ]
    ).to_csv(observations, index=False)

    exit_code = main(
        [
            "--predictions-json",
            str(predictions),
            "--target-date",
            "2026-05-24",
            "--as-of",
            "2026-05-24T15:00:00Z",
            "--decision-time-label",
            "10",
            "--observations-csv",
            str(observations),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "nowcast_features" / "nowcast_features.csv").exists()
    assert (out_dir / "predictions_nowcast_raw" / "predictions_nowcast.csv").exists()
    adjusted_path = out_dir / "predictions_nowcast_adjusted" / "predictions_nowcast.csv"
    assert adjusted_path.exists()
    assert (out_dir / "nowcast_report" / "nowcast_report.md").exists()
    assert (out_dir / "weather_desk_manifest.json").exists()
    adjusted = pd.read_csv(adjusted_path)
    assert adjusted["bin_lower_f"].tolist() == [75]
    assert adjusted["calibrated_probability"].tolist() == [1.0]
    assert "Wrote weather desk packet" in capsys.readouterr().out

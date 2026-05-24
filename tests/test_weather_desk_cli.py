import json
from pathlib import Path

import pandas as pd

from src.weather_desk_cli import main


def test_weather_desk_cli_writes_end_to_end_packet(tmp_path: Path, capsys, monkeypatch) -> None:
    predictions = tmp_path / "predictions.json"
    observations = tmp_path / "observations.csv"
    out_dir = tmp_path / "weather_desk"

    def fake_write_nws_guidance_rows(*, output_path, target, cities=None, market_types=None, fetched_at=None):
        assert target.isoformat() == "2026-05-24"
        assert cities is None
        assert market_types == ["high"]
        pd.DataFrame(
            [
                {
                    "city": "chicago",
                    "source": "nws_forecast",
                    "station_id": "KMDW",
                    "market_type": "high",
                    "target_date": "2026-05-24",
                    "issue_ts_utc": "2026-05-24T14:30:00+00:00",
                    "valid_ts_utc": "2026-05-25T00:00:00+00:00",
                    "available_ts_utc": "2026-05-24T14:30:00+00:00",
                    "guidance_point_f": 72,
                    "guidance_q10_f": None,
                    "guidance_q50_f": 74,
                    "guidance_q90_f": None,
                    "actual_high_f": None,
                    "raw_payload_hash": "abc",
                }
            ]
        ).to_csv(output_path, index=False)
        return pd.read_csv(output_path)

    monkeypatch.setattr(
        "src.weather_desk_cli.write_nws_guidance_rows",
        fake_write_nws_guidance_rows,
    )
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
            "--include-nws-guidance",
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
    assert (out_dir / "weather_analyst" / "weather_analyst_packet.md").exists()
    assert (out_dir / "guidance" / "nws_guidance_rows.csv").exists()
    assert (out_dir / "guidance_diagnostics" / "guidance_report.md").exists()
    comparison_path = out_dir / "guidance" / "model_vs_nws_guidance.csv"
    assert comparison_path.exists()
    assert (out_dir / "weather_desk_manifest.json").exists()
    adjusted = pd.read_csv(adjusted_path)
    assert adjusted["bin_lower_f"].tolist() == [75]
    assert adjusted["calibrated_probability"].tolist() == [1.0]
    comparison = pd.read_csv(comparison_path)
    assert comparison["model_minus_nws_f"].tolist() == [3.0]
    assert comparison["abs_model_minus_nws_f"].tolist() == [3.0]
    assert comparison["model_vs_nws_direction"].tolist() == ["model_hotter"]
    assert comparison["guidance_agreement"].tolist() == ["divergent"]
    comparison_md = (out_dir / "guidance" / "model_vs_nws_guidance.md").read_text(
        encoding="utf-8"
    )
    assert "Weather-only guidance comparison" in comparison_md
    assert "divergent" in comparison_md
    analyst = pd.read_csv(out_dir / "weather_analyst" / "weather_analyst_packet.csv")
    assert analyst["desk_priority"].tolist() == ["review"]
    assert "nws_divergent" in analyst.iloc[0]["risk_flags"]
    assert "Wrote weather desk packet" in capsys.readouterr().out

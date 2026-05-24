import pandas as pd

from src.nowcast_features_cli import main


def test_nowcast_features_cli_writes_outputs(tmp_path, capsys) -> None:
    observations = tmp_path / "observations.csv"
    out_dir = tmp_path / "nowcast"
    pd.DataFrame(
        [
            {
                "station_id": "KMDW",
                "obs_ts_utc": "2026-05-24T14:00:00",
                "available_ts_utc": "2026-05-24T14:00:00",
                "temperature_f": 72,
                "dewpoint_f": 55,
                "wind_speed_kt": 8,
                "cloud_cover": "CLR",
            }
        ]
    ).to_csv(observations, index=False)

    exit_code = main(
        [
            "--target-date",
            "2026-05-24",
            "--as-of",
            "2026-05-24T14:30:00Z",
            "--decision-time-label",
            "10",
            "--observations-csv",
            str(observations),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "asos_observations.csv").exists()
    assert (out_dir / "nowcast_features.csv").exists()
    assert (out_dir / "nowcast_features_report.md").exists()
    assert (out_dir / "nowcast_features_manifest.json").exists()
    assert "Wrote nowcast features" in capsys.readouterr().out


def test_nowcast_features_cli_uses_observation_store(tmp_path, capsys) -> None:
    store = tmp_path / "store.csv"
    out_dir = tmp_path / "nowcast"
    pd.DataFrame(
        [
            {
                "station_id": "KMDW",
                "obs_ts_utc": "2026-05-24T14:00:00",
                "available_ts_utc": "2026-05-24T14:00:00",
                "temperature_f": 72,
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
    ).to_csv(store, index=False)

    exit_code = main(
        [
            "--target-date",
            "2026-05-24",
            "--as-of",
            "2026-05-24T14:30:00Z",
            "--decision-time-label",
            "10",
            "--observation-store",
            str(store),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    features = pd.read_csv(out_dir / "nowcast_features.csv")
    assert features.loc[features["city"] == "chicago", "latest_temp_f"].item() == 72
    assert "Wrote nowcast features" in capsys.readouterr().out

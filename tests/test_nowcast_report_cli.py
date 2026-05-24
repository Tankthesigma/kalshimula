from pathlib import Path

import pandas as pd

from src.nowcast_report_cli import main


def test_nowcast_report_cli_writes_outputs(tmp_path: Path, capsys) -> None:
    predictions = tmp_path / "predictions_nowcast.csv"
    out_dir = tmp_path / "report"
    pd.DataFrame(
        [
            {
                "city": "la",
                "platform": "kalshi",
                "market_type": "high",
                "station_id": "KLAX",
                "target_date": "2026-05-24",
                "decision_time_label": "10",
                "source_policy": "gfs_ens",
                "point_f": 68,
                "q10_f": 66,
                "q50_f": 68,
                "q90_f": 70,
                "bin_label": "68",
                "calibrated_probability": 0.6,
                "nowcast_veto_flag": False,
                "weather_reason_codes": "",
                "station_rule_confidence": "high",
                "source_independence_score": 1.0,
            }
        ]
    ).to_csv(predictions, index=False)

    exit_code = main(
        [
            "--predictions-nowcast",
            str(predictions),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "nowcast_report_summary.csv").exists()
    assert (out_dir / "nowcast_report.md").exists()
    assert (out_dir / "nowcast_report_manifest.json").exists()
    assert "Wrote 1 city rows" in capsys.readouterr().out

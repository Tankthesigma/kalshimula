import json

import pandas as pd

from src.walkforward_eval_cli import main


def test_walkforward_eval_cli_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "walkforward"
    _rows(days=50).to_csv(input_path, index=False)

    exit_code = main(
        [
            "--rows",
            str(input_path),
            "--out-dir",
            str(output_dir),
            "--cities",
            "denver",
            "--sources",
            "gfs_ens,openmeteo_naive",
            "--train-window-days",
            "30",
            "--test-window-days",
            "5",
            "--step-days",
            "5",
            "--threshold-offsets",
            "-2,0,2",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "walkforward_events.csv").exists()
    assert (output_dir / "walkforward_predictions.csv").exists()
    assert (output_dir / "walkforward_window_summary.csv").exists()
    assert (output_dir / "walkforward_city_source_summary.csv").exists()
    assert (output_dir / "walkforward_threshold_summary.csv").exists()
    assert (output_dir / "walkforward_policy_leaderboard.csv").exists()
    assert (output_dir / "walkforward_report.md").exists()
    manifest = json.loads((output_dir / "walkforward_manifest.json").read_text())
    assert manifest["command_args"]["cities"] == ["denver"]
    assert "Wrote walk-forward evaluation" in capsys.readouterr().out


def _rows(days: int) -> pd.DataFrame:
    out = []
    for day, target_date in enumerate(pd.date_range("2025-01-01", periods=days)):
        actual = 70 + (day % 5)
        out.append(
            {
                "city": "denver",
                "target_date": target_date.date().isoformat(),
                "source": "gfs_ens",
                "point_f": actual - 1,
                "actual_high_f": actual,
            }
        )
        out.append(
            {
                "city": "denver",
                "target_date": target_date.date().isoformat(),
                "source": "openmeteo_naive",
                "point_f": actual,
                "actual_high_f": actual,
            }
        )
    return pd.DataFrame(out)

import pandas as pd

from src.threshold_calibration_cli import main


def test_threshold_calibration_cli_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    source_path = tmp_path / "recommended_sources.csv"
    bias_path = tmp_path / "bias_table.csv"
    output_dir = tmp_path / "probability_calibration"
    pd.DataFrame(
        [
            _row("2025-01-01", 70, 68),
            _row("2025-01-02", 70, 70),
            _row("2025-01-03", 70, 72),
            _row("2025-01-04", 70, 74),
            _row("2025-01-05", 70, 72),
            _row("2025-01-06", 70, 74),
            _row("2025-01-07", 70, 72),
            _row("2025-01-08", 70, 74),
        ]
    ).to_csv(input_path, index=False)
    pd.DataFrame([{"city": "denver", "selected_source": "gfs_ens"}]).to_csv(
        source_path, index=False
    )
    pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "n": 4,
                "mean_error_f": 0.0,
                "bias_correction_f": 0.0,
            }
        ]
    ).to_csv(bias_path, index=False)

    code = main(
        [
            "--input",
            str(input_path),
            "--recommended-sources",
            str(source_path),
            "--bias-table",
            str(bias_path),
            "--out-dir",
            str(output_dir),
            "--validation-start",
            "2025-01-05",
            "--test-start",
            "2025-01-07",
            "--offsets",
            "0,2",
            "--buckets",
            "5",
        ]
    )

    assert code == 0
    assert "test events" in capsys.readouterr().out
    assert (output_dir / "threshold_calibration_summary.csv").exists()


def _row(target_date: str, point_f: float, actual_high_f: float) -> dict:
    return {
        "city": "denver",
        "target_date": target_date,
        "source": "gfs_ens",
        "point_f": point_f,
        "actual_high_f": actual_high_f,
        "absolute_error_f": abs(point_f - actual_high_f),
    }

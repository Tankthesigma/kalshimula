import pandas as pd

from src.validation_grid_cli import main


def test_validation_grid_cli_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "grid"
    rows = pd.DataFrame(
        [
            _row("2025-01-01", 100, 50),
            _row("2025-01-02", 70, 72),
            _row("2025-01-03", 70, 72),
            _row("2025-01-04", 70, 72),
            _row("2025-01-05", 70, 72),
        ]
    )
    rows.to_csv(input_path, index=False)

    code = main(
        [
            "--input",
            str(input_path),
            "--out-dir",
            str(output_dir),
            "--validation-start",
            "2025-01-03",
            "--test-start",
            "2025-01-05",
            "--recent-days",
            "2,3",
            "--alphas",
            "0.2",
            "--policy-out-dir",
            str(output_dir / "model_policy"),
        ]
    )

    assert code == 0
    assert "selected recent_" in capsys.readouterr().out
    assert (output_dir / "validation_grid.csv").exists()
    assert (output_dir / "test_grid.csv").exists()
    assert (output_dir / "selected_config.csv").exists()
    assert (output_dir / "model_policy" / "bias_table.csv").exists()


def _row(target_date: str, point_f: float, actual_high_f: float) -> dict:
    return {
        "city": "denver",
        "target_date": target_date,
        "source": "gfs_ens",
        "point_f": point_f,
        "actual_high_f": actual_high_f,
        "absolute_error_f": abs(point_f - actual_high_f),
    }

import pandas as pd

from src.residual_diagnostics_cli import main


def test_residual_diagnostics_cli_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "diagnostics"
    pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "openmeteo",
                "target_date": "2025-01-01",
                "point_f": 70,
                "actual_high_f": 68,
            }
        ]
    ).to_csv(input_path, index=False)

    exit_code = main(["--input", str(input_path), "--out-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "source_residuals.csv").exists()
    assert (output_dir / "monthly_residuals.csv").exists()
    assert "Wrote residual diagnostics" in capsys.readouterr().out

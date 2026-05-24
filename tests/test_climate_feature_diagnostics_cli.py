import pandas as pd

from src.climate_feature_diagnostics_cli import main


def test_climate_feature_diagnostics_cli_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "climate"
    pd.DataFrame(
        [
            {
                "city": "denver",
                "target_date": target_date.date().isoformat(),
                "actual_high_f": float(index),
            }
            for index, target_date in enumerate(pd.date_range("2025-01-01", periods=40))
        ]
    ).to_csv(input_path, index=False)

    exit_code = main(["--input", str(input_path), "--out-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "climate_features.csv").exists()
    assert (output_dir / "climate_feature_summary.csv").exists()
    assert (output_dir / "climate_feature_diagnostics.md").exists()
    assert "Wrote climate feature diagnostics" in capsys.readouterr().out

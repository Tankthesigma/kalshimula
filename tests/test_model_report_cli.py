import pandas as pd

from src.model_report_cli import main


def test_model_report_cli_writes_report(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "report"
    pd.DataFrame(
        [
            {"city": "denver", "target_date": "2025-01-01", "source": "openmeteo", "point_f": 70, "actual_high_f": 68, "absolute_error_f": 2},
            {"city": "denver", "target_date": "2025-01-02", "source": "openmeteo", "point_f": 72, "actual_high_f": 71, "absolute_error_f": 1},
        ]
    ).to_csv(input_path, index=False)

    code = main(["--input", str(input_path), "--out-dir", str(output_dir)])

    assert code == 0
    assert (output_dir / "raw_summary.csv").exists()
    assert "Wrote report" in capsys.readouterr().out

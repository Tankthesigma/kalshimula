import pandas as pd

from src.train_eval_split_cli import main


def test_train_eval_split_cli_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "eval"
    pd.DataFrame(
        [
            {"city": "denver", "target_date": "2025-01-01", "source": "openmeteo", "point_f": 70, "actual_high_f": 68, "absolute_error_f": 2},
            {"city": "denver", "target_date": "2025-01-02", "source": "openmeteo", "point_f": 72, "actual_high_f": 71, "absolute_error_f": 1},
            {"city": "denver", "target_date": "2025-01-03", "source": "openmeteo", "point_f": 73, "actual_high_f": 73, "absolute_error_f": 0},
        ]
    ).to_csv(input_path, index=False)

    code = main(
        [
            "--input",
            str(input_path),
            "--test-start",
            "2025-01-03",
            "--out-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    assert (output_dir / "evaluation.csv").exists()
    assert "2 train rows, 1 test rows" in capsys.readouterr().out

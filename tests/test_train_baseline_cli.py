import pandas as pd

from src.train_baseline_cli import main


def test_train_baseline_cli_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    bias_out = tmp_path / "bias.csv"
    evaluation_out = tmp_path / "evaluation.csv"
    pd.DataFrame(
        [
            {"city": "denver", "source": "openmeteo", "point_f": 70, "actual_high_f": 68},
            {"city": "denver", "source": "openmeteo", "point_f": 72, "actual_high_f": 71},
        ]
    ).to_csv(input_path, index=False)

    code = main(
        [
            "--input",
            str(input_path),
            "--bias-out",
            str(bias_out),
            "--evaluation-out",
            str(evaluation_out),
        ]
    )

    assert code == 0
    assert bias_out.exists()
    assert evaluation_out.exists()
    assert "Wrote 1 bias rows" in capsys.readouterr().out

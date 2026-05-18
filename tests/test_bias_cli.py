import pandas as pd

from src.bias_cli import main


def test_bias_cli_writes_table(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    output_path = tmp_path / "bias.csv"
    pd.DataFrame(
        [{"city": "denver", "source": "openmeteo", "point_f": 70, "actual_high_f": 68}]
    ).to_csv(input_path, index=False)

    code = main(["--input", str(input_path), "--out", str(output_path)])

    assert code == 0
    assert output_path.exists()
    assert "Wrote 1 bias rows" in capsys.readouterr().out

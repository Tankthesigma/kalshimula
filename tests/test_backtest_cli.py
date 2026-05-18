from pathlib import Path

import pandas as pd

from src.backtest_cli import main, summarize_backtest_csv


def test_summarize_backtest_csv_writes_summary(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    output_path = tmp_path / "summary" / "summary.csv"
    pd.DataFrame(
        [
            {
                "city": "denver",
                "target_date": "2025-01-01",
                "source": "nws",
                "point_f": 70,
                "actual_high_f": 68,
                "absolute_error_f": 2,
            },
            {
                "city": "denver",
                "target_date": "2025-01-02",
                "source": "nws",
                "point_f": 71,
                "actual_high_f": 74,
                "absolute_error_f": 3,
            },
        ]
    ).to_csv(input_path, index=False)

    summary = summarize_backtest_csv(input_path, output_path)
    written = pd.read_csv(output_path)

    assert list(summary.columns) == ["city", "source", "n", "mae", "rmse", "bias"]
    assert summary.iloc[0]["mae"] == 2.5
    assert written.iloc[0]["city"] == "denver"


def test_backtest_cli_main_writes_output(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    output_path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "city": "denver",
                "target_date": "2025-01-01",
                "source": "nws",
                "point_f": 70,
                "actual_high_f": 68,
                "absolute_error_f": 2,
            }
        ]
    ).to_csv(input_path, index=False)

    code = main(["--input", str(input_path), "--out", str(output_path)])

    assert code == 0
    assert output_path.exists()
    assert "Wrote 1 summary rows" in capsys.readouterr().out


def test_backtest_cli_accepts_path_arguments(tmp_path) -> None:
    input_path = Path(tmp_path / "rows.csv")
    output_path = Path(tmp_path / "summary.csv")
    pd.DataFrame(
        columns=["city", "target_date", "source", "point_f", "actual_high_f"]
    ).to_csv(input_path, index=False)

    code = main(["--input", str(input_path), "--out", str(output_path)])

    assert code == 0
    assert output_path.exists()

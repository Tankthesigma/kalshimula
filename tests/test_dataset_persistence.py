from datetime import date

from src.datasets.backtest import make_backtest_row
from src.datasets.persistence import (
    load_backtest_rows,
    load_training_examples,
    save_backtest_rows,
    save_training_examples,
)
from src.datasets.training import make_training_example


def test_save_and_load_training_examples(tmp_path) -> None:
    path = tmp_path / "datasets" / "training.csv"
    example = make_training_example(
        city="denver",
        target_date=date(2025, 1, 1),
        source="nws",
        forecast_high_f=70,
        actual_high_f=68,
    )

    save_training_examples(path, [example])
    df = load_training_examples(path)

    assert list(df.columns) == [
        "city",
        "target_date",
        "source",
        "forecast_high_f",
        "actual_high_f",
        "error_f",
    ]
    assert df.iloc[0]["city"] == "denver"
    assert df.iloc[0]["error_f"] == 2.0


def test_save_and_load_backtest_rows(tmp_path) -> None:
    path = tmp_path / "datasets" / "backtest.csv"
    row = make_backtest_row(
        city="chicago",
        target_date=date(2025, 1, 2),
        source="naive",
        point_f=31,
        actual_high_f=35,
    )

    save_backtest_rows(path, [row])
    df = load_backtest_rows(path)

    assert list(df.columns) == [
        "city",
        "target_date",
        "source",
        "point_f",
        "actual_high_f",
        "absolute_error_f",
    ]
    assert df.iloc[0]["absolute_error_f"] == 4.0

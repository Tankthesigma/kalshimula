from datetime import date

from src.datasets.backtest import backtest_rows_to_dataframe, make_backtest_row


def test_make_backtest_row_computes_absolute_error() -> None:
    row = make_backtest_row(
        city="miami",
        target_date=date(2025, 3, 4),
        source="naive",
        point_f=88,
        actual_high_f=91.5,
    )

    assert row.absolute_error_f == 3.5


def test_backtest_rows_to_dataframe_has_stable_columns() -> None:
    row = make_backtest_row(
        city="austin",
        target_date=date(2025, 4, 5),
        source="naive",
        point_f=80,
        actual_high_f=79,
    )

    df = backtest_rows_to_dataframe([row])

    assert list(df.columns) == [
        "city",
        "target_date",
        "source",
        "point_f",
        "actual_high_f",
        "absolute_error_f",
    ]
    assert df.iloc[0]["absolute_error_f"] == 1.0

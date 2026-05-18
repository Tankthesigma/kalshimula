from datetime import date

import pytest

from src.datasets.backtest import backtest_rows_to_dataframe, make_backtest_row
from src.models.backtest import summarize_backtest


def test_summarize_backtest_groups_by_city_and_source() -> None:
    rows = [
        make_backtest_row(
            city="denver",
            target_date=date(2025, 1, 1),
            source="naive",
            point_f=70,
            actual_high_f=68,
        ),
        make_backtest_row(
            city="denver",
            target_date=date(2025, 1, 2),
            source="naive",
            point_f=71,
            actual_high_f=74,
        ),
    ]
    df = backtest_rows_to_dataframe(rows)

    summary = summarize_backtest(df)

    assert list(summary.columns) == ["city", "source", "n", "mae", "rmse", "bias"]
    assert summary.iloc[0]["city"] == "denver"
    assert summary.iloc[0]["n"] == 2
    assert summary.iloc[0]["mae"] == pytest.approx(2.5)
    assert summary.iloc[0]["bias"] == pytest.approx(-0.5)


def test_summarize_backtest_returns_empty_stable_shape() -> None:
    summary = summarize_backtest(backtest_rows_to_dataframe([]))

    assert list(summary.columns) == ["city", "source", "n", "mae", "rmse", "bias"]
    assert summary.empty


def test_summarize_backtest_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        summarize_backtest(backtest_rows_to_dataframe([]).drop(columns=["point_f"]))

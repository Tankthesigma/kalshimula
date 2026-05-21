import pandas as pd
import pytest

from src.models.train_eval import (
    split_rows_by_date,
    split_rows_by_month_stratified,
    train_eval_split,
    write_train_eval_outputs,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"city": "denver", "target_date": "2025-01-01", "source": "openmeteo", "point_f": 70, "actual_high_f": 68, "absolute_error_f": 2},
            {"city": "denver", "target_date": "2025-01-02", "source": "openmeteo", "point_f": 72, "actual_high_f": 71, "absolute_error_f": 1},
            {"city": "denver", "target_date": "2025-01-03", "source": "openmeteo", "point_f": 73, "actual_high_f": 73, "absolute_error_f": 0},
        ]
    )


def test_split_rows_by_date_uses_inclusive_test_start() -> None:
    train, test = split_rows_by_date(_rows(), test_start="2025-01-03")

    assert len(train) == 2
    assert len(test) == 1
    assert str(test.iloc[0]["target_date"].date()) == "2025-01-03"


def test_split_rows_by_month_stratified_holds_out_each_month() -> None:
    rows = pd.DataFrame(
        [
            {"city": "denver", "target_date": "2025-01-01", "source": "openmeteo", "point_f": 70, "actual_high_f": 68},
            {"city": "denver", "target_date": "2025-01-02", "source": "openmeteo", "point_f": 71, "actual_high_f": 69},
            {"city": "denver", "target_date": "2025-02-01", "source": "openmeteo", "point_f": 40, "actual_high_f": 41},
            {"city": "denver", "target_date": "2025-02-02", "source": "openmeteo", "point_f": 42, "actual_high_f": 43},
        ]
    )

    train, test = split_rows_by_month_stratified(rows, test_fraction=0.5)

    assert len(train) == 2
    assert len(test) == 2
    assert {d.month for d in test["target_date"]} == {1, 2}
    assert list(test["target_date"].dt.day) == [2, 2]


def test_train_eval_split_fits_on_train_and_evaluates_test() -> None:
    result = train_eval_split(_rows(), test_start="2025-01-03")

    assert len(result.train_rows) == 2
    assert len(result.test_rows) == 1
    assert len(result.bias_table) == 2
    assert "month" in result.bias_table.columns
    assert "corrected_point_f" in result.corrected_test_rows.columns
    assert "interval_lower_f" in result.corrected_test_rows.columns
    assert "interval_lower_raw_f" in result.corrected_test_rows.columns
    assert "interval_lower_corrected_f" in result.corrected_test_rows.columns
    assert "interval_coverage_corrected" in result.evaluation.columns
    assert len(result.source_residuals) == 1
    assert len(result.monthly_residuals) == 1
    assert len(result.evaluation) == 1


def test_train_eval_split_supports_month_stratified_strategy() -> None:
    rows = pd.DataFrame(
        [
            {"city": "denver", "target_date": "2025-01-01", "source": "openmeteo", "point_f": 70, "actual_high_f": 68},
            {"city": "denver", "target_date": "2025-01-02", "source": "openmeteo", "point_f": 72, "actual_high_f": 70},
            {"city": "denver", "target_date": "2025-02-01", "source": "openmeteo", "point_f": 40, "actual_high_f": 43},
            {"city": "denver", "target_date": "2025-02-02", "source": "openmeteo", "point_f": 42, "actual_high_f": 45},
        ]
    )

    result = train_eval_split(
        rows,
        split_strategy="month-stratified",
        test_fraction=0.5,
    )

    assert len(result.train_rows) == 2
    assert len(result.test_rows) == 2
    assert set(result.bias_table["month"].dropna().astype(int)) == {1, 2}


def test_train_eval_split_rejects_empty_train_or_test() -> None:
    with pytest.raises(ValueError):
        train_eval_split(_rows(), test_start="2025-01-01")
    with pytest.raises(ValueError):
        train_eval_split(_rows(), test_start="2026-01-01")
    with pytest.raises(ValueError, match="test_start is required"):
        train_eval_split(_rows())


def test_write_train_eval_outputs(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "eval"
    _rows().to_csv(input_path, index=False)

    result = write_train_eval_outputs(
        input_path=input_path,
        output_dir=output_dir,
        test_start="2025-01-03",
    )

    assert len(result.evaluation) == 1
    assert (output_dir / "train_rows.csv").exists()
    assert (output_dir / "test_rows.csv").exists()
    assert (output_dir / "bias_table.csv").exists()
    assert (output_dir / "interval_table.csv").exists()
    assert (output_dir / "corrected_test_rows.csv").exists()
    assert (output_dir / "evaluation.csv").exists()
    assert (output_dir / "source_residuals.csv").exists()
    assert (output_dir / "monthly_residuals.csv").exists()

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


def _selection_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("nyc", "2024-03-01", 75, 70),
            _row("nyc", "2025-02-20", 71, 70),
            _row("nyc", "2025-03-15", 75, 70),
            _row("nyc", "2025-04-15", 75, 70),
            _row("austin", "2024-03-01", 78, 70),
            _row("austin", "2025-02-20", 71, 70),
            _row("austin", "2025-03-15", 71, 70),
            _row("austin", "2025-04-15", 71, 70),
            _row("miami", "2025-02-20", 72, 70),
            _row("miami", "2025-04-15", 72, 70),
        ]
    )


def _row(city: str, target_date: str, point_f: float, actual_high_f: float) -> dict:
    return {
        "city": city,
        "target_date": target_date,
        "source": "openmeteo",
        "point_f": point_f,
        "actual_high_f": actual_high_f,
        "absolute_error_f": abs(point_f - actual_high_f),
    }


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
    assert result.validation_rows.empty
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
    assert result.evaluation.iloc[0]["selected_bias_method"] == "prior_same_month"


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


def test_train_eval_split_supports_recent_bias_strategy() -> None:
    rows = pd.DataFrame(
        [
            {
                "city": "denver",
                "target_date": "2025-01-01",
                "source": "openmeteo",
                "point_f": 100,
                "actual_high_f": 50,
            },
            {
                "city": "denver",
                "target_date": "2025-01-02",
                "source": "openmeteo",
                "point_f": 70,
                "actual_high_f": 72,
            },
            {
                "city": "denver",
                "target_date": "2025-01-03",
                "source": "openmeteo",
                "point_f": 71,
                "actual_high_f": 73,
            },
            {
                "city": "denver",
                "target_date": "2025-01-04",
                "source": "openmeteo",
                "point_f": 70,
                "actual_high_f": 72,
            },
        ]
    )

    result = train_eval_split(
        rows,
        test_start="2025-01-04",
        bias_strategy="recent",
        bias_recent_days=2,
    )

    assert "month" not in result.bias_table.columns
    assert result.bias_table.iloc[0]["bias_correction_f"] == pytest.approx(2.0)
    assert result.corrected_test_rows.iloc[0]["corrected_point_f"] == pytest.approx(72.0)
    assert result.selected_methods.iloc[0]["selected_bias_method"] == "recent_2d"


def test_train_eval_split_selects_bias_method_per_city_from_validation() -> None:
    result = train_eval_split(
        _selection_rows(),
        validation_start="2025-03-01",
        test_start="2025-04-01",
    )

    nyc = result.selected_methods[result.selected_methods["city"] == "nyc"].iloc[0]
    austin = result.selected_methods[result.selected_methods["city"] == "austin"].iloc[0]
    miami = result.selected_methods[result.selected_methods["city"] == "miami"].iloc[0]
    nyc_scores = result.validation_scores[result.validation_scores["city"] == "nyc"]

    assert set(nyc_scores["method"]) == {
        "recent_180d",
        "prior_same_month",
        "recent_365d",
        "all_train",
    }
    assert nyc["selected_bias_method"] == "prior_same_month"
    assert nyc["selected_validation_mae"] == pytest.approx(0)
    assert austin["selected_bias_method"] == "recent_180d"
    assert miami["selected_bias_method"] == "prior_same_month"
    assert miami["selection_fallback"]
    assert "selected_bias_method" in result.corrected_test_rows.columns
    assert "selected_validation_mae" in result.evaluation.columns


def test_train_eval_split_rejects_recent_bias_without_window() -> None:
    with pytest.raises(ValueError, match="bias_recent_days is required"):
        train_eval_split(_rows(), test_start="2025-01-03", bias_strategy="recent")


def test_train_eval_split_rejects_invalid_recent_bias_window() -> None:
    with pytest.raises(ValueError, match="bias_recent_days must be at least 1"):
        train_eval_split(
            _rows(),
            test_start="2025-01-03",
            bias_strategy="recent",
            bias_recent_days=0,
        )


def test_train_eval_split_rejects_empty_validation_slice() -> None:
    with pytest.raises(ValueError, match="validation slice is empty"):
        train_eval_split(
            _rows(),
            test_start="2025-01-03",
            validation_start="2025-01-03",
        )


def test_train_eval_split_uses_all_train_when_recent_window_is_larger_than_train() -> None:
    result = train_eval_split(
        _rows(),
        test_start="2025-01-03",
        bias_strategy="recent",
        bias_recent_days=1000,
    )

    assert result.bias_table.iloc[0]["n"] == 2
    assert result.bias_table.iloc[0]["bias_correction_f"] == pytest.approx(-1.5)


def test_train_eval_split_rejects_month_stratified_without_test_rows() -> None:
    rows = pd.DataFrame(
        [
            {"city": "denver", "target_date": "2025-01-01", "source": "openmeteo", "point_f": 70, "actual_high_f": 68},
            {"city": "denver", "target_date": "2025-02-01", "source": "openmeteo", "point_f": 40, "actual_high_f": 43},
        ]
    )

    with pytest.raises(ValueError, match="test split is empty"):
        train_eval_split(rows, split_strategy="month-stratified")


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
    assert (output_dir / "validation_rows.csv").exists()
    assert (output_dir / "test_rows.csv").exists()
    assert (output_dir / "bias_table.csv").exists()
    assert (output_dir / "interval_table.csv").exists()
    assert (output_dir / "validation_scores.csv").exists()
    assert (output_dir / "selected_methods.csv").exists()
    assert (output_dir / "corrected_test_rows.csv").exists()
    assert (output_dir / "evaluation.csv").exists()
    assert (output_dir / "source_residuals.csv").exists()
    assert (output_dir / "monthly_residuals.csv").exists()

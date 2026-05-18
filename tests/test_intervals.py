import pandas as pd
import pytest

from src.models.intervals import (
    apply_empirical_intervals,
    fit_empirical_intervals,
    write_interval_table,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"city": "denver", "source": "openmeteo", "point_f": 70, "actual_high_f": 68},
            {"city": "denver", "source": "openmeteo", "point_f": 70, "actual_high_f": 70},
            {"city": "denver", "source": "openmeteo", "point_f": 70, "actual_high_f": 72},
        ]
    )


def test_fit_empirical_intervals_groups_errors() -> None:
    table = fit_empirical_intervals(_rows(), alpha=0.2)

    row = table.iloc[0]
    assert row["n"] == 3
    assert row["lower_error_f"] == pytest.approx(-1.6)
    assert row["upper_error_f"] == pytest.approx(1.6)
    assert row["alpha"] == pytest.approx(0.2)


def test_apply_empirical_intervals_adds_bounds() -> None:
    intervals = fit_empirical_intervals(_rows(), alpha=0.2)
    applied = apply_empirical_intervals(_rows(), intervals)

    assert applied.iloc[0]["interval_lower_f"] == pytest.approx(68.4)
    assert applied.iloc[0]["interval_upper_f"] == pytest.approx(71.6)


def test_apply_empirical_intervals_defaults_missing_group_to_point() -> None:
    rows = pd.DataFrame([{"city": "nyc", "source": "nws", "point_f": 50}])
    intervals = pd.DataFrame(columns=["city", "source", "lower_error_f", "upper_error_f"])

    applied = apply_empirical_intervals(rows, intervals)

    assert applied.iloc[0]["interval_lower_f"] == 50
    assert applied.iloc[0]["interval_upper_f"] == 50


def test_fit_empirical_intervals_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        fit_empirical_intervals(_rows(), alpha=1.0)


def test_write_interval_table(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    output_path = tmp_path / "intervals" / "intervals.csv"
    _rows().to_csv(input_path, index=False)

    table = write_interval_table(input_path, output_path, alpha=0.2)
    written = pd.read_csv(output_path)

    assert len(table) == 1
    assert len(written) == 1

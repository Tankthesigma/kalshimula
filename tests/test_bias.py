import pandas as pd
import pytest

from src.models.bias import apply_bias_correction, fit_bias_table, write_bias_table


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"city": "denver", "source": "openmeteo", "point_f": 70, "actual_high_f": 68},
            {"city": "denver", "source": "openmeteo", "point_f": 72, "actual_high_f": 71},
            {"city": "chicago", "source": "openmeteo", "point_f": 30, "actual_high_f": 34},
        ]
    )


def test_bias_correction_sign_round_trip_for_underforecast() -> None:
    rows = pd.DataFrame(
        [{"city": "denver", "source": "openmeteo", "point_f": 70, "actual_high_f": 72}]
    )

    table = fit_bias_table(rows)
    corrected = apply_bias_correction(rows, table)

    assert table.iloc[0]["mean_error_f"] == pytest.approx(-2.0)
    assert table.iloc[0]["bias_correction_f"] == pytest.approx(2.0)
    assert corrected.iloc[0]["corrected_point_f"] == pytest.approx(72.0)


def test_bias_correction_sign_round_trip_for_overforecast() -> None:
    rows = pd.DataFrame(
        [{"city": "denver", "source": "openmeteo", "point_f": 72, "actual_high_f": 70}]
    )

    table = fit_bias_table(rows)
    corrected = apply_bias_correction(rows, table)

    assert table.iloc[0]["mean_error_f"] == pytest.approx(2.0)
    assert table.iloc[0]["bias_correction_f"] == pytest.approx(-2.0)
    assert corrected.iloc[0]["corrected_point_f"] == pytest.approx(70.0)


def test_fit_bias_table_groups_by_city_source() -> None:
    table = fit_bias_table(_rows())

    denver = table[table["city"] == "denver"].iloc[0]
    chicago = table[table["city"] == "chicago"].iloc[0]
    assert denver["n"] == 2
    assert denver["mean_error_f"] == pytest.approx(1.5)
    assert denver["bias_correction_f"] == pytest.approx(-1.5)
    assert chicago["bias_correction_f"] == pytest.approx(4.0)


def test_apply_bias_correction_adds_corrected_point() -> None:
    table = fit_bias_table(_rows())
    corrected = apply_bias_correction(_rows(), table)

    assert corrected.iloc[0]["corrected_point_f"] == pytest.approx(68.5)


def test_apply_bias_correction_defaults_missing_group_to_zero() -> None:
    rows = pd.DataFrame([{"city": "nyc", "source": "nws", "point_f": 50}])
    table = pd.DataFrame(columns=["city", "source", "bias_correction_f"])

    corrected = apply_bias_correction(rows, table)

    assert corrected.iloc[0]["corrected_point_f"] == 50


def test_fit_seasonal_bias_table_groups_by_month_with_fallback() -> None:
    rows = pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "openmeteo",
                "target_date": f"2025-05-{day:02d}",
                "point_f": 72,
                "actual_high_f": 70,
            }
            for day in range(1, 7)
        ]
        + [
            {
                "city": "denver",
                "source": "openmeteo",
                "target_date": f"2025-11-{day:02d}",
                "point_f": 69,
                "actual_high_f": 70,
            }
            for day in range(1, 7)
        ]
    )

    table = fit_bias_table(rows, group_month=True)
    may = table[table["month"].eq(5)].iloc[0]
    november = table[table["month"].eq(11)].iloc[0]
    fallback = table[table["month"].isna()].iloc[0]

    assert may["bias_correction_f"] == pytest.approx(-2.0)
    assert november["bias_correction_f"] == pytest.approx(1.0)
    assert fallback["bias_correction_f"] == pytest.approx(-0.5)

    corrected = apply_bias_correction(
        pd.DataFrame(
            [
                {
                    "city": "denver",
                    "source": "openmeteo",
                    "target_date": "2026-05-15",
                    "point_f": 80,
                }
            ]
        ),
        table,
    )
    assert corrected.iloc[0]["corrected_point_f"] == pytest.approx(78.0)


def test_apply_seasonal_bias_correction_falls_back_for_missing_month() -> None:
    rows = pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "openmeteo",
                "target_date": "2025-05-01",
                "point_f": 72,
                "actual_high_f": 70,
            },
            {
                "city": "denver",
                "source": "openmeteo",
                "target_date": "2025-11-01",
                "point_f": 69,
                "actual_high_f": 70,
            },
        ]
    )
    table = fit_bias_table(rows, group_month=True)

    corrected = apply_bias_correction(
        pd.DataFrame(
            [
                {
                    "city": "denver",
                    "source": "openmeteo",
                    "target_date": "2026-02-01",
                    "point_f": 80,
                }
            ]
        ),
        table,
    )

    assert corrected.iloc[0]["bias_correction_f"] == pytest.approx(-0.5)
    assert corrected.iloc[0]["corrected_point_f"] == pytest.approx(79.5)


def test_apply_seasonal_bias_correction_defaults_missing_group_to_zero() -> None:
    rows = pd.DataFrame(
        [{"city": "nyc", "source": "nws", "target_date": "2026-02-01", "point_f": 50}]
    )
    table = pd.DataFrame(
        columns=["city", "source", "month", "bias_correction_f"]
    )

    corrected = apply_bias_correction(rows, table)

    assert corrected.iloc[0]["bias_correction_f"] == 0
    assert corrected.iloc[0]["corrected_point_f"] == 50


def test_fit_bias_table_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        fit_bias_table(pd.DataFrame({"city": ["denver"]}))


def test_write_bias_table(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    output_path = tmp_path / "bias" / "bias.csv"
    _rows().to_csv(input_path, index=False)

    table = write_bias_table(input_path, output_path)
    written = pd.read_csv(output_path)

    assert len(table) == 2
    assert len(written) == 2

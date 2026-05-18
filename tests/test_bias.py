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

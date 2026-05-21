import pandas as pd
import pytest

from src.models.diagnostics import (
    RESIDUAL_SUMMARY_COLUMNS,
    build_residual_diagnostics,
    summarize_residuals,
    write_residual_diagnostics,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "openmeteo",
                "target_date": "2025-01-01",
                "point_f": 70,
                "corrected_point_f": 69,
                "actual_high_f": 68,
            },
            {
                "city": "denver",
                "source": "openmeteo",
                "target_date": "2025-01-02",
                "point_f": 72,
                "corrected_point_f": 71,
                "actual_high_f": 71,
            },
            {
                "city": "denver",
                "source": "openmeteo",
                "target_date": "2025-02-01",
                "point_f": 40,
                "corrected_point_f": 42,
                "actual_high_f": 43,
            },
        ]
    )


def test_summarize_residuals_reports_raw_and_corrected_spread() -> None:
    summary = summarize_residuals(_rows())
    row = summary.iloc[0]

    assert list(summary.columns) == RESIDUAL_SUMMARY_COLUMNS
    assert pd.isna(row["month"])
    assert row["n"] == 3
    assert row["bias_raw"] == pytest.approx(0.0)
    assert row["mae_raw"] == pytest.approx(2.0)
    assert row["residual_std_raw"] == pytest.approx(2.160246899)
    assert row["mae_corrected"] == pytest.approx(2 / 3)
    assert row["residual_std_corrected"] == pytest.approx(0.8164965809)


def test_summarize_residuals_can_group_by_month() -> None:
    summary = summarize_residuals(_rows(), group_month=True)

    jan = summary[summary["month"] == 1].iloc[0]
    feb = summary[summary["month"] == 2].iloc[0]

    assert jan["n"] == 2
    assert jan["bias_raw"] == pytest.approx(1.5)
    assert feb["n"] == 1
    assert feb["bias_raw"] == pytest.approx(-3.0)


def test_summarize_residuals_keeps_corrected_metrics_optional() -> None:
    rows = _rows().drop(columns=["corrected_point_f"])
    summary = summarize_residuals(rows)

    assert summary.iloc[0]["mae_raw"] == pytest.approx(2.0)
    assert pd.isna(summary.iloc[0]["mae_corrected"])


def test_summarize_residuals_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError):
        summarize_residuals(pd.DataFrame({"city": ["denver"]}))


def test_build_residual_diagnostics_returns_source_and_monthly_tables() -> None:
    diagnostics = build_residual_diagnostics(_rows())

    assert len(diagnostics.source_summary) == 1
    assert len(diagnostics.monthly_summary) == 2


def test_write_residual_diagnostics_outputs_csvs(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "diagnostics"
    _rows().to_csv(input_path, index=False)

    diagnostics = write_residual_diagnostics(input_path=input_path, output_dir=output_dir)

    assert len(diagnostics.source_summary) == 1
    assert (output_dir / "source_residuals.csv").exists()
    assert (output_dir / "monthly_residuals.csv").exists()

import pandas as pd

from src.models.report import build_model_report, write_model_report


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"city": "denver", "target_date": "2025-01-01", "source": "openmeteo", "point_f": 70, "actual_high_f": 68, "absolute_error_f": 2},
            {"city": "denver", "target_date": "2025-01-02", "source": "openmeteo", "point_f": 72, "actual_high_f": 71, "absolute_error_f": 1},
            {"city": "chicago", "target_date": "2025-01-01", "source": "openmeteo", "point_f": 30, "actual_high_f": 34, "absolute_error_f": 4},
        ]
    )


def test_build_model_report_returns_all_tables() -> None:
    report = build_model_report(_rows(), alpha=0.2)

    assert len(report.raw_summary) == 2
    assert len(report.bias_table) == 2
    assert len(report.corrected_evaluation) == 2
    assert len(report.intervals) == 2


def test_write_model_report_writes_expected_files(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "report"
    _rows().to_csv(input_path, index=False)

    report = write_model_report(input_path=input_path, output_dir=output_dir, alpha=0.2)

    assert len(report.raw_summary) == 2
    assert (output_dir / "raw_summary.csv").exists()
    assert (output_dir / "bias_table.csv").exists()
    assert (output_dir / "corrected_evaluation.csv").exists()
    assert (output_dir / "intervals.csv").exists()

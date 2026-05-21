import pandas as pd

from src.models.threshold_calibration import (
    evaluate_threshold_calibration,
    write_threshold_calibration_outputs,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("2025-01-01", 70, 68),
            _row("2025-01-02", 70, 70),
            _row("2025-01-03", 70, 72),
            _row("2025-01-04", 70, 74),
            _row("2025-01-05", 70, 72),
            _row("2025-01-06", 70, 74),
            _row("2025-01-07", 70, 72),
            _row("2025-01-08", 70, 74),
        ]
    )


def _row(target_date: str, point_f: float, actual_high_f: float) -> dict:
    return {
        "city": "denver",
        "source": "gfs_ens",
        "target_date": target_date,
        "point_f": point_f,
        "actual_high_f": actual_high_f,
        "absolute_error_f": abs(point_f - actual_high_f),
    }


def _recommended_sources() -> pd.DataFrame:
    return pd.DataFrame([{"city": "denver", "selected_source": "gfs_ens"}])


def _bias_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "n": 4,
                "mean_error_f": 0.0,
                "bias_correction_f": 0.0,
            }
        ]
    )


def test_evaluate_threshold_calibration_writes_events_and_summary() -> None:
    result = evaluate_threshold_calibration(
        rows=_rows(),
        recommended_sources=_recommended_sources(),
        bias_table=_bias_table(),
        validation_start="2025-01-05",
        test_start="2025-01-07",
        offsets=(0, 2),
        n_buckets=5,
    )

    assert len(result.validation_events) == 4
    assert len(result.test_events) == 4
    assert len(result.threshold_residuals) == 6
    assert set(result.validation_events["threshold_f"]) == {70, 72}
    assert result.test_events["predicted_probability"].between(0, 1).all()
    assert result.summary["split"].tolist() == ["validation", "test"]
    assert "expected_calibration_error" in result.summary.columns
    assert not result.test_calibration.empty
    assert result.validation_group_summary["city"].tolist() == ["denver"]
    assert result.test_group_summary["source"].tolist() == ["gfs_ens"]
    assert result.test_group_summary.iloc[0]["n_events"] == 4
    assert result.validation_group_calibration["city"].unique().tolist() == ["denver"]
    assert result.test_group_calibration["source"].unique().tolist() == ["gfs_ens"]
    assert "calibration_gap" in result.test_group_calibration.columns


def test_write_threshold_calibration_outputs(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    source_path = tmp_path / "recommended_sources.csv"
    bias_path = tmp_path / "bias_table.csv"
    output_dir = tmp_path / "probability_calibration"
    _rows().to_csv(input_path, index=False)
    _recommended_sources().to_csv(source_path, index=False)
    _bias_table().to_csv(bias_path, index=False)

    result = write_threshold_calibration_outputs(
        input_path=input_path,
        recommended_sources_path=source_path,
        bias_table_path=bias_path,
        output_dir=output_dir,
        validation_start="2025-01-05",
        test_start="2025-01-07",
        offsets=(0, 2),
        n_buckets=5,
    )

    assert len(result.test_events) == 4
    assert (output_dir / "threshold_validation_events.csv").exists()
    assert (output_dir / "threshold_test_events.csv").exists()
    assert (output_dir / "threshold_residuals.csv").exists()
    assert (output_dir / "threshold_validation_calibration.csv").exists()
    assert (output_dir / "threshold_test_calibration.csv").exists()
    assert (output_dir / "threshold_calibration_summary.csv").exists()
    assert (output_dir / "threshold_validation_group_summary.csv").exists()
    assert (output_dir / "threshold_test_group_summary.csv").exists()
    assert (output_dir / "threshold_validation_group_calibration.csv").exists()
    assert (output_dir / "threshold_test_group_calibration.csv").exists()

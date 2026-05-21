import pandas as pd

from src import model_policy_report_cli


def _write_artifacts(run_dir):
    (run_dir / "source_selection").mkdir(parents=True)
    (run_dir / "model_policy").mkdir(parents=True)
    (run_dir / "probability_calibration").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "city": "denver",
                "target_date": "2025-01-01",
                "source": "gfs_ens",
                "point_f": 70,
                "actual_high_f": 71,
            },
            {
                "city": "boston",
                "target_date": "2025-01-02",
                "source": "gfs_ens",
                "point_f": 40,
                "actual_high_f": 39,
            },
        ]
    ).to_csv(run_dir / "rows.csv", index=False)
    pd.DataFrame(
        [
            {
                "city": "denver",
                "selected_source": "gfs_ens",
                "recommended_policy": "best_global_validation_source",
            },
            {
                "city": "boston",
                "selected_source": "gfs_ens",
                "recommended_policy": "best_global_validation_source",
            },
        ]
    ).to_csv(run_dir / "source_selection" / "recommended_sources.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy": "global_recent_90d",
                "bias_strategy": "recent",
                "bias_recent_days": 90,
                "alpha": 0.13,
                "selected_by": "validation",
            }
        ]
    ).to_csv(run_dir / "model_policy" / "model_policy.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy": "global_recent_90d",
                "validation_mae_corrected": 0.99,
                "test_mae_corrected": 1.01,
                "test_interval_coverage_raw": 0.84,
                "test_interval_width_raw": 3.7,
                "recommended": True,
            }
        ]
    ).to_csv(run_dir / "model_policy" / "bias_policy_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy": "per_city_alpha",
                "alpha": "selected",
                "split": "test",
                "interval_coverage_raw": 0.827,
                "interval_width_raw": 3.57,
                "target_coverage": 0.8,
                "recommended": True,
            }
        ]
    ).to_csv(run_dir / "model_policy" / "interval_policy_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "test",
                "n_events": 6230,
                "brier_score": 0.0609,
                "expected_calibration_error": 0.0241,
                "mean_predicted_probability": 0.5089,
                "observed_frequency": 0.5140,
            }
        ]
    ).to_csv(
        run_dir / "probability_calibration" / "threshold_calibration_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "split": "test",
                "city": "denver",
                "source": "gfs_ens",
                "n_events": 700,
                "brier_score": 0.07,
                "expected_calibration_error": 0.03,
                "mean_predicted_probability": 0.5,
                "observed_frequency": 0.52,
            },
            {
                "split": "test",
                "city": "boston",
                "source": "gfs_ens",
                "n_events": 650,
                "brier_score": 0.08,
                "expected_calibration_error": 0.05,
                "mean_predicted_probability": 0.49,
                "observed_frequency": 0.55,
            },
        ]
    ).to_csv(
        run_dir / "probability_calibration" / "threshold_test_group_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "split": "test",
                "city": "denver",
                "source": "gfs_ens",
                "bucket_start": 0.4,
                "bucket_end": 0.5,
                "n": 80,
                "mean_predicted_probability": 0.45,
                "observed_frequency": 0.50,
                "calibration_gap": -0.05,
            },
            {
                "split": "test",
                "city": "boston",
                "source": "gfs_ens",
                "bucket_start": 0.5,
                "bucket_end": 0.6,
                "n": 75,
                "mean_predicted_probability": 0.55,
                "observed_frequency": 0.70,
                "calibration_gap": -0.15,
            },
        ]
    ).to_csv(
        run_dir / "probability_calibration" / "threshold_test_group_calibration.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "policy": "raw_empirical_residual",
                "split": "test",
                "n_events": 6230,
                "brier_score": 0.0609,
                "expected_calibration_error": 0.0241,
                "mean_predicted_probability": 0.5089,
                "observed_frequency": 0.5140,
            },
            {
                "policy": "validation_bucket_recalibrated",
                "split": "test",
                "n_events": 6230,
                "brier_score": 0.0569,
                "expected_calibration_error": 0.0096,
                "mean_predicted_probability": 0.5085,
                "observed_frequency": 0.5140,
            },
        ]
    ).to_csv(
        run_dir / "probability_calibration" / "threshold_recalibration_comparison.csv",
        index=False,
    )


def test_build_model_policy_report_summarizes_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_artifacts(run_dir)

    report = model_policy_report_cli.build_model_policy_report(run_dir)

    assert "Data: 2 rows, 2 cities, 1 sources" in report
    assert "Source policy: gfs_ens=2 cities" in report
    assert "Model policy: global_recent_90d" in report
    assert "Bias policy metrics: validation MAE=0.990F, test MAE=1.010F" in report
    assert "Interval policy metrics: per_city_alpha alpha=selected" in report
    assert "test: events=6,230, brier=0.061, ece=0.024" in report
    assert "worst test group: boston/gfs_ens brier=0.080, ece=0.050" in report
    assert "worst test bucket: boston/gfs_ens 50.0%-60.0% gap=-0.150" in report
    assert "recalibrated test: brier=0.057 (raw 0.061), ece=0.010 (raw 0.024)" in report


def test_build_model_policy_report_handles_missing_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    report = model_policy_report_cli.build_model_policy_report(run_dir)

    assert "Data: missing or empty rows.csv" in report
    assert "Model policy: missing model_policy/model_policy.csv" in report
    assert "Threshold probabilities: missing" in report


def test_model_policy_report_cli_prints_and_writes(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    out_path = tmp_path / "report.txt"
    run_dir.mkdir()
    _write_artifacts(run_dir)

    code = model_policy_report_cli.main(
        ["--run-dir", str(run_dir), "--out", str(out_path)]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "Run:" in output
    assert "Threshold probabilities:" in output
    assert out_path.exists()

import pandas as pd

from src import model_gate_cli


def _write_gate_artifacts(run_dir, *, mae=0.99, brier=0.056, ece=0.009) -> None:
    (run_dir / "source_selection").mkdir(parents=True)
    (run_dir / "model_policy").mkdir(parents=True)
    (run_dir / "probability_calibration").mkdir(parents=True)
    pd.DataFrame(
        [
            {"city": "denver", "selected_source": "gfs_ens"},
            {"city": "boston", "selected_source": "gfs_ens"},
        ]
    ).to_csv(run_dir / "source_selection" / "recommended_sources.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy": "global_recent_90d",
                "test_mae_corrected": mae,
                "recommended": True,
            }
        ]
    ).to_csv(run_dir / "model_policy" / "bias_policy_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy": "per_city_alpha",
                "split": "test",
                "interval_coverage_raw": 0.83,
                "interval_width_raw": 3.5,
                "recommended": True,
            }
        ]
    ).to_csv(run_dir / "model_policy" / "interval_policy_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy": "raw_empirical_residual",
                "brier_score": 0.061,
                "expected_calibration_error": 0.024,
            },
            {
                "policy": "validation_bucket_recalibrated",
                "brier_score": brier,
                "expected_calibration_error": ece,
            },
        ]
    ).to_csv(
        run_dir / "probability_calibration" / "threshold_recalibration_comparison.csv",
        index=False,
    )


def test_model_gate_cli_passes_ready_artifacts(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    _write_gate_artifacts(run_dir)

    code = model_gate_cli.main(["--run-dir", str(run_dir)])

    output = capsys.readouterr().out
    assert code == 0
    assert "Outcome: PASS" in output
    assert "PASS test_mae_corrected" in output


def test_model_gate_cli_fails_metric_threshold(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    _write_gate_artifacts(run_dir, mae=1.4)

    code = model_gate_cli.main(["--run-dir", str(run_dir)])

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL test_mae_corrected" in output
    assert "Outcome: FAIL" in output


def test_model_gate_cli_fails_missing_artifacts(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    code = model_gate_cli.main(["--run-dir", str(run_dir)])

    output = capsys.readouterr().out
    assert code == 1
    assert "artifact_error" in output
    assert "Outcome: FAIL" in output

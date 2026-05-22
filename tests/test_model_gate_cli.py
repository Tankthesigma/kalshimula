import json

import pandas as pd

from src import model_gate_cli


def _write_gate_artifacts(run_dir, *, mae=0.99, brier=0.056, ece=0.009) -> None:
    (run_dir / "source_selection").mkdir(parents=True)
    (run_dir / "model_policy").mkdir(parents=True)
    (run_dir / "probability_calibration").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "target_date": "2025-01-01",
                "point_f": 70,
                "actual_high_f": 71,
            },
            {
                "city": "boston",
                "source": "ecmwf_ens",
                "target_date": "2025-01-02",
                "point_f": 40,
                "actual_high_f": 39,
            },
        ]
    ).to_csv(run_dir / "rows.csv", index=False)
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


def _gate_args(run_dir) -> list[str]:
    return [
        "--run-dir",
        str(run_dir),
        "--min-rows",
        "2",
        "--min-cities",
        "2",
        "--min-sources",
        "2",
        "--min-target-dates",
        "2",
    ]


def test_model_gate_cli_passes_ready_artifacts(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    _write_gate_artifacts(run_dir)

    code = model_gate_cli.main(_gate_args(run_dir))

    output = capsys.readouterr().out
    assert code == 0
    assert "Outcome: PASS" in output
    assert "PASS row_count" in output
    assert "PASS city_count" in output
    assert "PASS source_count" in output
    assert "PASS target_date_count" in output
    assert "PASS test_mae_corrected" in output


def test_model_gate_cli_fails_metric_threshold(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    _write_gate_artifacts(run_dir, mae=1.4)

    code = model_gate_cli.main(_gate_args(run_dir))

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL test_mae_corrected" in output
    assert "Outcome: FAIL" in output


def test_model_gate_cli_fails_data_coverage_threshold(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    _write_gate_artifacts(run_dir)

    code = model_gate_cli.main([*_gate_args(run_dir), "--min-rows", "3"])

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL row_count" in output
    assert "Outcome: FAIL" in output


def test_model_gate_cli_fails_missing_artifacts(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    code = model_gate_cli.main(["--run-dir", str(run_dir)])

    output = capsys.readouterr().out
    assert code == 1
    assert "artifact_error" in output
    assert "Outcome: FAIL" in output


def test_model_gate_cli_emits_json_summary(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    _write_gate_artifacts(run_dir)

    code = model_gate_cli.main([*_gate_args(run_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["schema_version"] == "1.0"
    assert payload["run_dir"] == str(run_dir)
    assert payload["passed"] is True
    assert payload["summary"]["total_checks"] == len(payload["checks"])
    assert payload["summary"]["failed_checks"] == 0
    assert payload["summary"]["failed_check_names"] == []
    assert {check["name"] for check in payload["checks"]} >= {
        "row_count",
        "test_mae_corrected",
        "recalibrated_brier",
    }


def test_model_gate_cli_writes_json_failure_summary(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    _write_gate_artifacts(run_dir, mae=1.4)
    out_path = tmp_path / "gate" / "model_gate.json"

    code = model_gate_cli.main([*_gate_args(run_dir), "--out", str(out_path)])

    assert code == 1
    assert capsys.readouterr().out == ""
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["summary"]["failed_checks"] == 1
    assert payload["summary"]["failed_check_names"] == ["test_mae_corrected"]


def test_model_gate_cli_writes_json_artifact_error(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_path = tmp_path / "gate.json"

    code = model_gate_cli.main(["--run-dir", str(run_dir), "--json", "--out", str(out_path)])

    assert code == 1
    assert capsys.readouterr().out == ""
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["summary"]["failed_check_names"] == ["artifact_error"]
    assert payload["checks"][0]["name"] == "artifact_error"

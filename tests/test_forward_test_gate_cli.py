import json

from src import forward_test_gate_cli


def _report(summary=None):
    return {
        "schema_version": "1.0",
        "summary": summary
        or {
            "n_target_dates": 2,
            "n_predictions": 20,
            "n_threshold_events": 60,
            "mae_corrected_f": 1.0,
            "bias_corrected_f": -0.2,
            "interval_coverage": 0.82,
            "threshold_brier_score": 0.08,
            "threshold_ece": 0.12,
        },
        "daily": [],
    }


def _write_report(path, summary=None):
    path.write_text(json.dumps(_report(summary)), encoding="utf-8")


def _gate_args(path):
    return [
        "--report",
        str(path),
        "--min-target-dates",
        "2",
        "--min-predictions",
        "20",
        "--min-threshold-events",
        "60",
        "--max-mae",
        "1.5",
        "--max-abs-bias",
        "0.75",
        "--min-interval-coverage",
        "0.75",
        "--max-threshold-brier",
        "0.12",
        "--max-threshold-ece",
        "0.2",
    ]


def test_forward_test_gate_cli_passes_ready_report(tmp_path, capsys):
    report_path = tmp_path / "report.json"
    _write_report(report_path)

    code = forward_test_gate_cli.main(_gate_args(report_path))

    assert code == 0
    output = capsys.readouterr().out
    assert "PASS mae_corrected_f" in output
    assert "Outcome: PASS" in output


def test_forward_test_gate_cli_fails_bad_metrics(tmp_path, capsys):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        {
            "n_target_dates": 1,
            "n_predictions": 10,
            "n_threshold_events": 30,
            "mae_corrected_f": 2.0,
            "bias_corrected_f": -1.0,
            "interval_coverage": 0.5,
            "threshold_brier_score": 0.2,
            "threshold_ece": 0.3,
        },
    )

    code = forward_test_gate_cli.main(_gate_args(report_path))

    assert code == 1
    output = capsys.readouterr().out
    assert "FAIL target_date_count" in output
    assert "FAIL mae_corrected_f" in output
    assert "FAIL abs_bias_corrected_f" in output
    assert "FAIL interval_coverage" in output
    assert "FAIL threshold_brier_score" in output
    assert "FAIL threshold_ece" in output


def test_forward_test_gate_cli_writes_json(tmp_path):
    report_path = tmp_path / "report.json"
    out_path = tmp_path / "gate.json"
    _write_report(report_path)

    code = forward_test_gate_cli.main([*_gate_args(report_path), "--out", str(out_path)])

    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["passed"] is True
    assert payload["summary"]["failed_check_names"] == []


def test_forward_test_gate_cli_handles_missing_report(tmp_path):
    out_path = tmp_path / "gate.json"

    code = forward_test_gate_cli.main(
        ["--report", str(tmp_path / "missing.json"), "--out", str(out_path)]
    )

    assert code == 1
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["summary"]["failed_check_names"] == ["artifact_error"]

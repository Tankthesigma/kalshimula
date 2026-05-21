import json

from src import daily_packet_check_cli


def _prediction_payload(*, gate_passed=True, n_errors=0, n_predictions=1):
    errors = [{"city": "boston", "error": "failed"}] if n_errors else []
    predictions = (
        [
            {
                "artifact_paths": {"model_run_dir": "run"},
                "city": "denver",
                "forecast": {"point_f": 70.0},
                "calibration": {"corrected_point_f": 71.0},
                "selected_source": "gfs_ens",
                "selected_source_applied": True,
                "station": {"name": "Denver", "nws_station": "KDEN"},
                "threshold_probabilities": [],
            }
        ]
        if n_predictions
        else []
    )
    return {
        "schema_version": "1.0",
        "model_gate": {"required": True, "passed": gate_passed},
        "n_predictions": n_predictions,
        "n_errors": n_errors,
        "predictions": predictions,
        "errors": errors,
    }


def _write_packet(
    tmp_path,
    *,
    exit_code=0,
    missing_artifact=False,
    prediction_payload=None,
):
    review_artifact = tmp_path / "latest_predictions.txt"
    prediction_artifact = tmp_path / "latest_predictions.json"
    if not missing_artifact:
        review_artifact.write_text("Prediction review\n", encoding="utf-8")
        prediction_artifact.write_text(
            json.dumps(prediction_payload or _prediction_payload()),
            encoding="utf-8",
        )
    manifest = tmp_path / "latest_predictions_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generated_at": "2026-05-21T12:00:00+00:00",
                "model_run_dir": str(tmp_path),
                "cities": "denver,boston",
                "target_date": "tomorrow",
                "threshold_offsets": "-2,0,2",
                "require_gate": True,
                "exit_code": exit_code,
                "steps": {
                    "batch_predictions": {"exit_code": exit_code},
                    "prediction_review": {"exit_code": 0},
                    "model_gate_report": {"exit_code": 0},
                    "model_policy_report": {"exit_code": 0},
                },
                "artifacts": {
                    "prediction_json": str(prediction_artifact),
                    "prediction_review": str(review_artifact),
                    "manifest": str(manifest),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_build_packet_checks_passes_complete_packet(tmp_path) -> None:
    manifest = _write_packet(tmp_path)

    payload, checks = daily_packet_check_cli.build_packet_checks(manifest)

    assert payload["exit_code"] == 0
    assert all(check["passed"] for check in checks)
    assert {check["name"] for check in checks} >= {
        "manifest:schema_version",
        "manifest:exit_code",
        "step:batch_predictions",
        "prediction_json:schema_version",
        "prediction_json:prediction_fields",
        "artifact:prediction_review",
    }


def test_daily_packet_check_cli_writes_report(tmp_path, capsys) -> None:
    manifest = _write_packet(tmp_path)
    out_path = tmp_path / "packet_check.txt"

    code = daily_packet_check_cli.main(
        ["--manifest", str(manifest), "--out", str(out_path)]
    )

    assert code == 0
    assert capsys.readouterr().out == ""
    assert "Outcome: PASS" in out_path.read_text(encoding="utf-8")


def test_daily_packet_check_cli_fails_nonzero_step(tmp_path, capsys) -> None:
    manifest = _write_packet(tmp_path, exit_code=1)

    code = daily_packet_check_cli.main(["--manifest", str(manifest)])

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL manifest:exit_code" in output
    assert "FAIL step:batch_predictions" in output
    assert "Outcome: FAIL" in output


def test_daily_packet_check_cli_fails_missing_artifact(tmp_path, capsys) -> None:
    manifest = _write_packet(tmp_path, missing_artifact=True)

    code = daily_packet_check_cli.main(["--manifest", str(manifest)])

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL artifact:prediction_review" in output
    assert "missing" in output


def test_daily_packet_check_cli_fails_prediction_json_gate(tmp_path, capsys) -> None:
    manifest = _write_packet(
        tmp_path,
        prediction_payload=_prediction_payload(gate_passed=False),
    )

    code = daily_packet_check_cli.main(["--manifest", str(manifest)])

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL prediction_json:gate" in output
    assert "required=True passed=False" in output


def test_daily_packet_check_cli_fails_prediction_json_errors(tmp_path, capsys) -> None:
    manifest = _write_packet(
        tmp_path,
        prediction_payload=_prediction_payload(n_errors=1),
    )

    code = daily_packet_check_cli.main(["--manifest", str(manifest)])

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL prediction_json:error_count" in output


def test_daily_packet_check_cli_fails_prediction_json_missing_fields(
    tmp_path, capsys
) -> None:
    payload = _prediction_payload()
    payload["predictions"][0].pop("calibration")
    manifest = _write_packet(tmp_path, prediction_payload=payload)

    code = daily_packet_check_cli.main(["--manifest", str(manifest)])

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL prediction_json:prediction_fields" in output
    assert "calibration" in output


def test_daily_packet_check_cli_requires_dashboard_contract_fields(
    tmp_path, capsys
) -> None:
    payload = _prediction_payload()
    payload["predictions"][0].pop("station")
    payload["predictions"][0].pop("selected_source")
    payload["predictions"][0].pop("selected_source_applied")
    manifest = _write_packet(tmp_path, prediction_payload=payload)

    code = daily_packet_check_cli.main(["--manifest", str(manifest)])

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL prediction_json:prediction_fields" in output
    assert "selected_source" in output
    assert "selected_source_applied" in output
    assert "station" in output

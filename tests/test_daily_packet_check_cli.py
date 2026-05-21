import json
from datetime import UTC, datetime, timedelta

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
        "generated_at": "2026-05-21T12:00:00+00:00",
        "target_date": "2026-05-22",
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
    require_selected_source_applied=False,
    max_packet_age_hours=None,
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
    payload = {
        "schema_version": "1.0",
        "generated_at": "2026-05-21T12:00:00+00:00",
        "model_run_dir": str(tmp_path),
        "cities": "denver",
        "target_date": "tomorrow",
        "threshold_offsets": "-2,0,2",
        "require_gate": True,
        "require_selected_source_applied": require_selected_source_applied,
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
    if max_packet_age_hours is not None:
        payload["max_packet_age_hours"] = max_packet_age_hours
    manifest.write_text(json.dumps(payload), encoding="utf-8")
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


def test_daily_packet_check_cli_emits_json(tmp_path, capsys) -> None:
    manifest = _write_packet(tmp_path)

    code = daily_packet_check_cli.main(["--manifest", str(manifest), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["schema_version"] == "1.0"
    assert payload["manifest"] == str(manifest)
    assert payload["passed"] is True
    assert payload["require_selected_source_applied"] is False
    assert payload["max_packet_age_hours"] is None
    assert any(check["name"] == "prediction_json:prediction_fields" for check in payload["checks"])
    assert any(check["name"] == "prediction_json:cities" for check in payload["checks"])


def test_daily_packet_check_cli_writes_json_report(tmp_path, capsys) -> None:
    manifest = _write_packet(tmp_path)
    out_path = tmp_path / "packet_check.json"

    code = daily_packet_check_cli.main(
        ["--manifest", str(manifest), "--json", "--out", str(out_path)]
    )

    assert code == 0
    assert capsys.readouterr().out == ""
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["target_date"] == "tomorrow"


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


def test_daily_packet_check_cli_fails_city_mismatch(tmp_path, capsys) -> None:
    manifest = _write_packet(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cities"] = "denver,boston"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    code = daily_packet_check_cli.main(["--manifest", str(manifest)])

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL prediction_json:cities" in output
    assert "missing=['boston']" in output


def test_daily_packet_check_cli_fails_absolute_target_date_mismatch(
    tmp_path, capsys
) -> None:
    manifest = _write_packet(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["target_date"] = "2026-05-23"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    code = daily_packet_check_cli.main(["--manifest", str(manifest)])

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL prediction_json:target_date" in output
    assert "manifest_target=2026-05-23" in output


def test_daily_packet_check_cli_fails_invalid_prediction_timestamp(
    tmp_path, capsys
) -> None:
    payload = _prediction_payload()
    payload["generated_at"] = "not-a-date"
    manifest = _write_packet(tmp_path, prediction_payload=payload)

    code = daily_packet_check_cli.main(["--manifest", str(manifest)])

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL prediction_json:generated_at" in output


def test_build_packet_checks_enforces_manifest_freshness(tmp_path) -> None:
    generated_at = datetime(2026, 5, 21, 12, tzinfo=UTC)
    payload = _prediction_payload()
    payload["generated_at"] = generated_at.isoformat()
    manifest = _write_packet(
        tmp_path,
        prediction_payload=payload,
        max_packet_age_hours=2,
    )

    _, checks = daily_packet_check_cli.build_packet_checks(
        manifest,
        now=generated_at + timedelta(hours=3),
    )

    freshness = [check for check in checks if check["name"] == "prediction_json:freshness"]
    assert freshness == [
        {
            "name": "prediction_json:freshness",
            "passed": False,
            "detail": "age_hours=3.000 max_age_hours=2",
        }
    ]


def test_build_packet_checks_accepts_fresh_packet(tmp_path) -> None:
    generated_at = datetime(2026, 5, 21, 12, tzinfo=UTC)
    payload = _prediction_payload()
    payload["generated_at"] = generated_at.isoformat()
    manifest = _write_packet(
        tmp_path,
        prediction_payload=payload,
        max_packet_age_hours=2,
    )

    _, checks = daily_packet_check_cli.build_packet_checks(
        manifest,
        now=generated_at + timedelta(minutes=30),
    )

    assert any(
        check["name"] == "prediction_json:freshness" and check["passed"] is True
        for check in checks
    )


def test_daily_packet_check_cli_max_age_flag_overrides_manifest(tmp_path, capsys) -> None:
    payload = _prediction_payload()
    payload["generated_at"] = "2000-01-01T00:00:00+00:00"
    manifest = _write_packet(tmp_path, prediction_payload=payload)

    code = daily_packet_check_cli.main(
        ["--manifest", str(manifest), "--max-age-hours", "24"]
    )

    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL prediction_json:freshness" in output


def test_daily_packet_check_cli_requires_selected_source_applied_from_manifest(
    tmp_path, capsys
) -> None:
    payload = _prediction_payload()
    payload["predictions"][0]["selected_source_applied"] = False
    manifest = _write_packet(
        tmp_path,
        prediction_payload=payload,
        require_selected_source_applied=True,
    )

    code = daily_packet_check_cli.main(["--manifest", str(manifest)])

    output = capsys.readouterr().out
    assert code == 1
    assert "require_selected_source_applied: true" in output
    assert "FAIL prediction_json:selected_source_applied" in output
    assert "denver" in output


def test_daily_packet_check_cli_can_require_selected_source_applied_by_flag(
    tmp_path, capsys
) -> None:
    payload = _prediction_payload()
    payload["predictions"][0]["selected_source_applied"] = False
    manifest = _write_packet(tmp_path, prediction_payload=payload)

    code = daily_packet_check_cli.main(
        ["--manifest", str(manifest), "--json", "--require-selected-source-applied"]
    )

    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["require_selected_source_applied"] is True
    assert any(
        check["name"] == "prediction_json:selected_source_applied"
        and check["passed"] is False
        for check in result["checks"]
    )

import json

from src import daily_packet_check_cli


def _write_packet(tmp_path, *, exit_code=0, missing_artifact=False):
    artifact = tmp_path / "latest_predictions.txt"
    if not missing_artifact:
        artifact.write_text("Prediction review\n", encoding="utf-8")
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
                    "prediction_review": str(artifact),
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

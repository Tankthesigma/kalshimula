import json
from pathlib import Path

from src import daily_model_refresh_cli


def test_build_refresh_paths_defaults_to_model_run_dir() -> None:
    paths = daily_model_refresh_cli.build_refresh_paths(
        model_run_dir=Path("run"),
        out_dir=None,
        prefix="latest",
    )

    assert paths.json_out == Path("run/latest.json")
    assert paths.review_out == Path("run/latest.txt")
    assert paths.gate_out == Path("run/latest_gate.txt")
    assert paths.gate_json_out == Path("run/latest_gate.json")
    assert paths.policy_out == Path("run/latest_model_policy.txt")
    assert paths.manifest_out == Path("run/latest_manifest.json")
    assert paths.check_out == Path("run/latest_check.json")


def test_daily_model_refresh_cli_runs_batch_then_review(monkeypatch, tmp_path, capsys) -> None:
    calls = []

    def fake_batch_main(argv):
        calls.append(("batch", argv))
        return 0

    def fake_review_main(argv):
        calls.append(("review", argv))
        return 0

    def fake_write_gate_report(*, run_dir, out_path, json_out_path=None):
        calls.append(("gate", [str(run_dir), str(out_path), str(json_out_path)]))
        return 0

    def fake_write_policy_report(*, run_dir, out_path):
        calls.append(("policy", [str(run_dir), str(out_path)]))

    def fake_check_main(argv):
        calls.append(("check", argv))
        return 0

    monkeypatch.setattr("src.predict_batch_cli.main", fake_batch_main)
    monkeypatch.setattr("src.prediction_review_cli.main", fake_review_main)
    monkeypatch.setattr("src.daily_packet_check_cli.main", fake_check_main)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report", fake_write_gate_report
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report", fake_write_policy_report
    )

    code = daily_model_refresh_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--cities",
            "denver,boston",
            "--date",
            "2026-04-30",
            "--threshold-offsets",
            "-2,0,2",
            "--out-dir",
            str(tmp_path / "out"),
            "--prefix",
            "morning",
        ]
    )

    assert code == 0
    assert calls == [
        (
            "batch",
            [
                "--cities",
                "denver,boston",
                "--date",
                "2026-04-30",
                "--model-run-dir",
                str(tmp_path / "run"),
                "--threshold-offsets=-2,0,2",
                "--out",
                str(tmp_path / "out" / "morning.json"),
                "--require-gate",
            ],
        ),
        (
            "review",
            [
                "--input",
                str(tmp_path / "out" / "morning.json"),
                "--out",
                str(tmp_path / "out" / "morning.txt"),
            ],
        ),
        (
            "gate",
            [
                str(tmp_path / "run"),
                str(tmp_path / "out" / "morning_gate.txt"),
                str(tmp_path / "out" / "morning_gate.json"),
            ],
        ),
        (
            "policy",
            [
                str(tmp_path / "run"),
                str(tmp_path / "out" / "morning_model_policy.txt"),
            ],
        ),
        (
            "check",
            [
                "--manifest",
                str(tmp_path / "out" / "morning_manifest.json"),
                "--json",
                "--out",
                str(tmp_path / "out" / "morning_check.json"),
            ],
        ),
        (
            "check",
            [
                "--manifest",
                str(tmp_path / "out" / "morning_manifest.json"),
                "--json",
                "--out",
                str(tmp_path / "out" / "morning_check.json"),
            ],
        ),
    ]
    output = capsys.readouterr().out
    assert "Wrote prediction JSON" in output
    assert "Wrote prediction review" in output
    assert "Wrote model gate report" in output
    assert "Wrote model gate JSON" in output
    assert "Wrote model policy report" in output
    assert "Wrote packet manifest" in output
    assert "Wrote packet check" in output
    assert "Wrote final packet check" in output
    manifest = json.loads((tmp_path / "out" / "morning_manifest.json").read_text())
    assert manifest["exit_code"] == 0
    assert manifest["cities"] == "denver,boston"
    assert manifest["target_date"] == "2026-04-30"
    assert manifest["threshold_offsets"] == "-2,0,2"
    assert manifest["require_gate"] is True
    assert manifest["require_selected_source_applied"] is True
    assert manifest["max_packet_age_hours"] == 24.0
    assert manifest["steps"]["batch_predictions"]["exit_code"] == 0
    assert manifest["steps"]["model_gate_json"]["exit_code"] == 0
    assert manifest["artifacts"]["model_gate_json"] == str(
        tmp_path / "out" / "morning_gate.json"
    )
    assert manifest["artifacts"]["model_policy_report"] == str(
        tmp_path / "out" / "morning_model_policy.txt"
    )


def test_daily_model_refresh_cli_returns_batch_failure_but_still_reviews(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_batch_main(argv):
        calls.append("batch")
        return 1

    def fake_review_main(argv):
        calls.append("review")
        return 1

    def fake_write_gate_report(*, run_dir, out_path, json_out_path=None):
        calls.append("gate")
        return 0

    def fake_write_policy_report(*, run_dir, out_path):
        calls.append("policy")

    def fake_check_main(argv):
        calls.append("check")
        return 1

    monkeypatch.setattr("src.predict_batch_cli.main", fake_batch_main)
    monkeypatch.setattr("src.prediction_review_cli.main", fake_review_main)
    monkeypatch.setattr("src.daily_packet_check_cli.main", fake_check_main)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report", fake_write_gate_report
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report", fake_write_policy_report
    )

    code = daily_model_refresh_cli.main(
        ["--model-run-dir", str(tmp_path / "run"), "--prefix", "bad"]
    )

    assert code == 1
    assert calls == ["batch", "review", "gate", "policy", "check", "check"]
    manifest = json.loads((tmp_path / "run" / "bad_manifest.json").read_text())
    assert manifest["exit_code"] == 1
    assert manifest["steps"]["batch_predictions"]["exit_code"] == 1
    assert manifest["steps"]["prediction_review"]["exit_code"] == 1


def test_daily_model_refresh_cli_can_skip_required_gate(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_batch_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr("src.predict_batch_cli.main", fake_batch_main)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.daily_packet_check_cli.main", lambda argv: 0)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path, json_out_path=None: 0,
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report",
        lambda *, run_dir, out_path: None,
    )

    code = daily_model_refresh_cli.main(
        ["--model-run-dir", str(tmp_path / "run"), "--no-require-gate"]
    )

    assert code == 0
    assert "--require-gate" not in captured["argv"]
    manifest = json.loads(
        (tmp_path / "run" / "latest_predictions_manifest.json").read_text()
    )
    assert manifest["require_gate"] is False


def test_daily_model_refresh_cli_can_allow_source_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.predict_batch_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.daily_packet_check_cli.main", lambda argv: 0)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path, json_out_path=None: 0,
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report",
        lambda *, run_dir, out_path: None,
    )

    code = daily_model_refresh_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--allow-source-fallback",
        ]
    )

    assert code == 0
    manifest = json.loads(
        (tmp_path / "run" / "latest_predictions_manifest.json").read_text()
    )
    assert manifest["require_selected_source_applied"] is False


def test_daily_model_refresh_cli_can_omit_packet_age_limit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.predict_batch_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.daily_packet_check_cli.main", lambda argv: 0)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path, json_out_path=None: 0,
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report",
        lambda *, run_dir, out_path: None,
    )

    code = daily_model_refresh_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--no-max-packet-age",
        ]
    )

    assert code == 0
    manifest = json.loads(
        (tmp_path / "run" / "latest_predictions_manifest.json").read_text()
    )
    assert manifest["max_packet_age_hours"] is None


def test_daily_model_refresh_cli_returns_check_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.predict_batch_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path, json_out_path=None: 0,
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report",
        lambda *, run_dir, out_path: None,
    )
    monkeypatch.setattr("src.daily_packet_check_cli.main", lambda argv: 1)

    code = daily_model_refresh_cli.main(
        ["--model-run-dir", str(tmp_path / "run"), "--prefix", "bad-check"]
    )

    assert code == 1


def test_daily_model_refresh_cli_can_run_forward_settlement(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_batch_main(argv):
        calls.append(("batch", argv))
        return 0

    def fake_settle_main(argv):
        calls.append(("settle", argv))
        return 0

    monkeypatch.setattr("src.predict_batch_cli.main", fake_batch_main)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.daily_packet_check_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.forward_test_settle_cli.main", fake_settle_main)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path, json_out_path=None: 0,
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report",
        lambda *, run_dir, out_path: None,
    )

    code = daily_model_refresh_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--out-dir",
            str(tmp_path / "out"),
            "--prefix",
            "packet",
            "--settle",
            "--settle-target-date",
            "2026-05-22",
            "--settle-actuals-csv",
            str(tmp_path / "actuals.csv"),
        ]
    )

    assert code == 0
    settle_call = next(argv for name, argv in calls if name == "settle")
    assert settle_call == [
        "--packet",
        str(tmp_path / "out" / "packet.json"),
        "--target-date",
        "2026-05-22",
        "--out-dir",
        str(tmp_path / "out" / "forward_test"),
        "--actuals-csv",
        str(tmp_path / "actuals.csv"),
    ]
    manifest = json.loads((tmp_path / "out" / "packet_manifest.json").read_text())
    assert manifest["exit_code"] == 0
    assert manifest["steps"]["packet_check"]["exit_code"] == 0
    assert manifest["steps"]["forward_test_settlement"]["exit_code"] == 0
    assert manifest["artifacts"]["settlement_json"] == str(
        tmp_path / "out" / "forward_test" / "2026-05-22_settlement.json"
    )
    assert manifest["artifacts"]["settlement_history"] == str(
        tmp_path / "out" / "forward_test" / "history.csv"
    )
    assert manifest["artifacts"]["settlement_report"] == str(
        tmp_path / "out" / "forward_test" / "report.json"
    )


def test_daily_model_refresh_cli_can_infer_settlement_target_from_packet(
    monkeypatch, tmp_path
) -> None:
    captured = {}
    out_dir = tmp_path / "out"

    def fake_batch_main(argv):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "packet.json").write_text(
            json.dumps({"target_date": "2026-05-22"}),
            encoding="utf-8",
        )
        return 0

    def fake_settle_main(argv):
        captured["settle"] = argv
        return 0

    monkeypatch.setattr("src.predict_batch_cli.main", fake_batch_main)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.daily_packet_check_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.forward_test_settle_cli.main", fake_settle_main)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path, json_out_path=None: 0,
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report",
        lambda *, run_dir, out_path: None,
    )

    code = daily_model_refresh_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "packet",
            "--settle",
            "--settlement-no-report",
        ]
    )

    assert code == 0
    assert "--target-date" in captured["settle"]
    assert captured["settle"][captured["settle"].index("--target-date") + 1] == "2026-05-22"
    assert "--no-report" in captured["settle"]
    manifest = json.loads((out_dir / "packet_manifest.json").read_text())
    assert "settlement_report" not in manifest["artifacts"]


def test_daily_model_refresh_cli_returns_settlement_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.predict_batch_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.daily_packet_check_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.forward_test_settle_cli.main", lambda argv: 1)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path, json_out_path=None: 0,
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report",
        lambda *, run_dir, out_path: None,
    )

    code = daily_model_refresh_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--settle",
            "--settle-target-date",
            "2026-05-22",
        ]
    )

    assert code == 1
    manifest = json.loads(
        (tmp_path / "run" / "latest_predictions_manifest.json").read_text()
    )
    assert manifest["steps"]["forward_test_settlement"]["exit_code"] == 1


def test_daily_model_refresh_cli_can_run_forward_test_gate_after_settlement(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_settle_main(argv):
        calls.append(("settle", argv))
        return 0

    def fake_forward_gate_main(argv):
        calls.append(("forward_gate", argv))
        return 0

    monkeypatch.setattr("src.predict_batch_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.daily_packet_check_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.forward_test_settle_cli.main", fake_settle_main)
    monkeypatch.setattr("src.forward_test_gate_cli.main", fake_forward_gate_main)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path, json_out_path=None: 0,
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report",
        lambda *, run_dir, out_path: None,
    )

    code = daily_model_refresh_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--out-dir",
            str(tmp_path / "out"),
            "--prefix",
            "packet",
            "--settle",
            "--settle-target-date",
            "2026-05-22",
            "--settle-actuals-csv",
            str(tmp_path / "actuals.csv"),
            "--forward-test-gate",
            "--forward-test-min-target-dates",
            "1",
            "--forward-test-min-predictions",
            "10",
            "--forward-test-min-threshold-events",
            "30",
        ]
    )

    assert code == 0
    gate_call = next(argv for name, argv in calls if name == "forward_gate")
    assert gate_call == [
        "--report",
        str(tmp_path / "out" / "forward_test" / "report.json"),
        "--out",
        str(tmp_path / "out" / "forward_test" / "forward_test_gate.json"),
        "--min-target-dates",
        "1",
        "--min-predictions",
        "10",
        "--min-threshold-events",
        "30",
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
    manifest = json.loads((tmp_path / "out" / "packet_manifest.json").read_text())
    assert manifest["steps"]["forward_test_gate"]["exit_code"] == 0
    assert manifest["artifacts"]["forward_test_gate_json"] == str(
        tmp_path / "out" / "forward_test" / "forward_test_gate.json"
    )


def test_daily_model_refresh_cli_final_check_sees_final_artifacts(
    monkeypatch, tmp_path
) -> None:
    checks_saw_forward_gate = []

    def fake_check_main(argv):
        manifest_path = Path(argv[argv.index("--manifest") + 1])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks_saw_forward_gate.append("forward_test_gate_json" in manifest["artifacts"])
        return 0

    monkeypatch.setattr("src.predict_batch_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.daily_packet_check_cli.main", fake_check_main)
    monkeypatch.setattr("src.forward_test_settle_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.forward_test_gate_cli.main", lambda argv: 0)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path, json_out_path=None: 0,
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report",
        lambda *, run_dir, out_path: None,
    )

    code = daily_model_refresh_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--out-dir",
            str(tmp_path / "out"),
            "--prefix",
            "packet",
            "--settle",
            "--settle-target-date",
            "2026-05-22",
            "--forward-test-gate",
        ]
    )

    assert code == 0
    assert checks_saw_forward_gate == [False, True]


def test_daily_model_refresh_cli_final_check_failure_rewrites_manifest(
    monkeypatch, tmp_path
) -> None:
    check_codes = iter([0, 1, 1])
    check_count = 0

    def fake_check_main(argv):
        nonlocal check_count
        check_count += 1
        return next(check_codes)

    monkeypatch.setattr("src.predict_batch_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.daily_packet_check_cli.main", fake_check_main)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path, json_out_path=None: 0,
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report",
        lambda *, run_dir, out_path: None,
    )

    code = daily_model_refresh_cli.main(
        ["--model-run-dir", str(tmp_path / "run"), "--prefix", "bad-final-check"]
    )

    assert code == 1
    assert check_count == 3
    manifest = json.loads(
        (tmp_path / "run" / "bad-final-check_manifest.json").read_text()
    )
    assert manifest["exit_code"] == 1
    assert manifest["steps"]["packet_check"]["exit_code"] == 1


def test_daily_model_refresh_cli_can_gate_existing_forward_test_report(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    def fake_forward_gate_main(argv):
        captured["gate"] = argv
        return 1

    monkeypatch.setattr("src.predict_batch_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.daily_packet_check_cli.main", lambda argv: 0)
    monkeypatch.setattr("src.forward_test_gate_cli.main", fake_forward_gate_main)
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path, json_out_path=None: 0,
    )
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_policy_report",
        lambda *, run_dir, out_path: None,
    )

    code = daily_model_refresh_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--forward-test-gate",
            "--forward-test-gate-report",
            str(tmp_path / "forward" / "report.json"),
            "--forward-test-gate-out",
            str(tmp_path / "forward" / "gate.json"),
        ]
    )

    assert code == 1
    assert captured["gate"][:4] == [
        "--report",
        str(tmp_path / "forward" / "report.json"),
        "--out",
        str(tmp_path / "forward" / "gate.json"),
    ]
    manifest = json.loads(
        (tmp_path / "run" / "latest_predictions_manifest.json").read_text()
    )
    assert manifest["steps"]["forward_test_gate"]["exit_code"] == 1
    assert manifest["artifacts"]["forward_test_gate_json"] == str(
        tmp_path / "forward" / "gate.json"
    )


def test_write_gate_report_writes_failure_for_missing_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_path = tmp_path / "out" / "gate.txt"
    json_out_path = tmp_path / "out" / "gate.json"

    code = daily_model_refresh_cli._write_gate_report(
        run_dir=run_dir,
        out_path=out_path,
        json_out_path=json_out_path,
    )

    assert code == 1
    text = out_path.read_text(encoding="utf-8")
    assert "Outcome: FAIL" in text
    assert "artifact_error" in text
    payload = json.loads(json_out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["summary"]["failed_check_names"] == ["artifact_error"]


def test_write_policy_report_writes_model_policy_summary(tmp_path) -> None:
    run_dir = tmp_path / "run"
    out_path = tmp_path / "out" / "policy.txt"

    daily_model_refresh_cli._write_policy_report(
        run_dir=run_dir,
        out_path=out_path,
    )

    text = out_path.read_text(encoding="utf-8")
    assert "Data: missing or empty rows.csv" in text
    assert "Model policy: missing model_policy/model_policy.csv" in text


def test_write_manifest_uses_first_nonzero_exit_code(tmp_path) -> None:
    paths = daily_model_refresh_cli.build_refresh_paths(
        model_run_dir=tmp_path / "run",
        out_dir=tmp_path / "out",
        prefix="packet",
    )

    code = daily_model_refresh_cli._write_manifest(
        out_path=paths.manifest_out,
        model_run_dir=tmp_path / "run",
        cities="denver",
        target_date="tomorrow",
        threshold_offsets="-2,0,2",
        require_gate=True,
        require_selected_source_applied=True,
        max_packet_age_hours=24.0,
        paths=paths,
        batch_code=0,
        review_code=1,
        gate_code=0,
    )

    payload = json.loads(paths.manifest_out.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["schema_version"] == "1.0"
    assert payload["exit_code"] == 1
    assert payload["steps"]["prediction_review"]["exit_code"] == 1
    assert payload["artifacts"]["manifest"] == str(paths.manifest_out)

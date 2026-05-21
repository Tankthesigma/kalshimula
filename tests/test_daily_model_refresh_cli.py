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
    assert paths.policy_out == Path("run/latest_model_policy.txt")
    assert paths.manifest_out == Path("run/latest_manifest.json")


def test_daily_model_refresh_cli_runs_batch_then_review(monkeypatch, tmp_path, capsys) -> None:
    calls = []

    def fake_batch_main(argv):
        calls.append(("batch", argv))
        return 0

    def fake_review_main(argv):
        calls.append(("review", argv))
        return 0

    def fake_write_gate_report(*, run_dir, out_path):
        calls.append(("gate", [str(run_dir), str(out_path)]))
        return 0

    def fake_write_policy_report(*, run_dir, out_path):
        calls.append(("policy", [str(run_dir), str(out_path)]))

    monkeypatch.setattr("src.predict_batch_cli.main", fake_batch_main)
    monkeypatch.setattr("src.prediction_review_cli.main", fake_review_main)
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
            ],
        ),
        (
            "policy",
            [
                str(tmp_path / "run"),
                str(tmp_path / "out" / "morning_model_policy.txt"),
            ],
        ),
    ]
    output = capsys.readouterr().out
    assert "Wrote prediction JSON" in output
    assert "Wrote prediction review" in output
    assert "Wrote model gate report" in output
    assert "Wrote model policy report" in output
    assert "Wrote packet manifest" in output
    manifest = json.loads((tmp_path / "out" / "morning_manifest.json").read_text())
    assert manifest["exit_code"] == 0
    assert manifest["cities"] == "denver,boston"
    assert manifest["target_date"] == "2026-04-30"
    assert manifest["threshold_offsets"] == "-2,0,2"
    assert manifest["require_gate"] is True
    assert manifest["steps"]["batch_predictions"]["exit_code"] == 0
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

    def fake_write_gate_report(*, run_dir, out_path):
        calls.append("gate")
        return 0

    def fake_write_policy_report(*, run_dir, out_path):
        calls.append("policy")

    monkeypatch.setattr("src.predict_batch_cli.main", fake_batch_main)
    monkeypatch.setattr("src.prediction_review_cli.main", fake_review_main)
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
    assert calls == ["batch", "review", "gate", "policy"]
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
    monkeypatch.setattr(
        "src.daily_model_refresh_cli._write_gate_report",
        lambda *, run_dir, out_path: 0,
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


def test_write_gate_report_writes_failure_for_missing_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_path = tmp_path / "out" / "gate.txt"

    code = daily_model_refresh_cli._write_gate_report(
        run_dir=run_dir,
        out_path=out_path,
    )

    assert code == 1
    text = out_path.read_text(encoding="utf-8")
    assert "Outcome: FAIL" in text
    assert "artifact_error" in text


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

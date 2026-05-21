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


def test_daily_model_refresh_cli_runs_batch_then_review(monkeypatch, tmp_path, capsys) -> None:
    calls = []

    def fake_batch_main(argv):
        calls.append(("batch", argv))
        return 0

    def fake_review_main(argv):
        calls.append(("review", argv))
        return 0

    monkeypatch.setattr("src.predict_batch_cli.main", fake_batch_main)
    monkeypatch.setattr("src.prediction_review_cli.main", fake_review_main)

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
    ]
    output = capsys.readouterr().out
    assert "Wrote prediction JSON" in output
    assert "Wrote prediction review" in output


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

    monkeypatch.setattr("src.predict_batch_cli.main", fake_batch_main)
    monkeypatch.setattr("src.prediction_review_cli.main", fake_review_main)

    code = daily_model_refresh_cli.main(
        ["--model-run-dir", str(tmp_path / "run"), "--prefix", "bad"]
    )

    assert code == 1
    assert calls == ["batch", "review"]


def test_daily_model_refresh_cli_can_skip_required_gate(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_batch_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr("src.predict_batch_cli.main", fake_batch_main)
    monkeypatch.setattr("src.prediction_review_cli.main", lambda argv: 0)

    code = daily_model_refresh_cli.main(
        ["--model-run-dir", str(tmp_path / "run"), "--no-require-gate"]
    )

    assert code == 0
    assert "--require-gate" not in captured["argv"]

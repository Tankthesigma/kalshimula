import json
from pathlib import Path

from src import weather_desk_refresh_cli
from src.models.station_rules import DEFAULT_STATION_RULES_PATH


def test_weather_desk_refresh_cli_runs_predictions_then_desk(monkeypatch, tmp_path: Path, capsys) -> None:
    calls = []

    def fake_predict_batch_main(argv):
        calls.append(("predict", argv))
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"predictions": []}\n', encoding="utf-8")
        return 0

    def fake_weather_desk_main(argv):
        calls.append(("desk", argv))
        return 0

    monkeypatch.setattr(
        "src.weather_desk_refresh_cli.predict_batch_cli.main",
        fake_predict_batch_main,
    )
    monkeypatch.setattr(
        "src.weather_desk_refresh_cli.weather_desk_cli.main",
        fake_weather_desk_main,
    )
    monkeypatch.setattr(
        "src.weather_desk_refresh_cli._git_commit",
        lambda: "abc123",
    )

    code = weather_desk_refresh_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--cities",
            "nyc,boston",
            "--date",
            "2026-05-24",
            "--as-of",
            "2026-05-24T15:00:00Z",
            "--threshold-offsets",
            "-2,0,2",
            "--multi-source-mode",
            "blend_equal",
            "--decision-time-label",
            "morning",
            "--include-nws-guidance",
            "--include-nbm-guidance",
            "--fetch-live",
            "--out-dir",
            str(tmp_path / "out"),
            "--prefix",
            "morning",
        ]
    )

    assert code == 0
    prediction_path = tmp_path / "out" / "morning_predictions.json"
    desk_dir = tmp_path / "out" / "morning"
    assert calls == [
        (
            "predict",
            [
                "--cities",
                "nyc,boston",
                "--date",
                "2026-05-24",
                "--model-run-dir",
                str(tmp_path / "run"),
                "--threshold-offsets=-2,0,2",
                "--multi-source-mode",
                "blend_equal",
                "--out",
                str(prediction_path),
                "--require-gate",
            ],
        ),
        (
            "desk",
            [
                "--predictions-json",
                str(prediction_path),
                "--target-date",
                "2026-05-24",
                "--as-of",
                "2026-05-24T15:00:00+00:00",
                "--decision-time-label",
                "morning",
                "--station-rules",
                str(DEFAULT_STATION_RULES_PATH),
                "--cities",
                "nyc,boston",
                "--market-type",
                "high",
                "--model-version",
                "mainline-nowcast-v1",
                "--out-dir",
                str(desk_dir),
                "--fetch-live",
                "--include-nws-guidance",
                "--include-nbm-guidance",
            ],
        ),
    ]
    manifest = json.loads((tmp_path / "out" / "morning_refresh_manifest.json").read_text())
    assert manifest["exit_code"] == 0
    assert manifest["steps"]["predict_batch"]["exit_code"] == 0
    assert manifest["steps"]["weather_desk"]["exit_code"] == 0
    assert manifest["include_nbm_guidance"] is True
    assert manifest["artifacts"]["weather_desk_dir"] == str(desk_dir)
    assert "Wrote weather desk refresh manifest" in capsys.readouterr().out


def test_weather_desk_refresh_cli_skips_desk_if_prediction_file_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "src.weather_desk_refresh_cli.predict_batch_cli.main",
        lambda argv: 1,
    )

    called = False

    def fake_weather_desk_main(argv):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(
        "src.weather_desk_refresh_cli.weather_desk_cli.main",
        fake_weather_desk_main,
    )

    code = weather_desk_refresh_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--date",
            "2026-05-24",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 1
    assert called is False

import json
from pathlib import Path

from src import weather_desk_backfill_cli


def test_weather_desk_backfill_cli_runs_inclusive_date_range(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_schedule_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(
        "src.weather_desk_backfill_cli.weather_desk_schedule_cli.main",
        fake_schedule_main,
    )
    monkeypatch.setattr("src.weather_desk_backfill_cli._git_commit", lambda: "abc123")

    out_dir = tmp_path / "backfill"
    code = weather_desk_backfill_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--start-date",
            "2026-05-22",
            "--end-date",
            "2026-05-24",
            "--cities",
            "nyc,la",
            "--decision-hours",
            "07,13",
            "--threshold-offsets",
            "-2,0,2",
            "--multi-source-mode",
            "blend_equal",
            "--observation-store",
            str(tmp_path / "obs.csv"),
            "--update-observation-store",
            "--fetch-live",
            "--include-nws-guidance",
            "--no-require-gate",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    assert len(calls) == 3
    assert calls[0][calls[0].index("--date") + 1] == "2026-05-22"
    assert calls[0][calls[0].index("--out-dir") + 1] == str(out_dir / "2026-05-22")
    assert calls[0][calls[0].index("--decision-minute") + 1] == "20"
    assert calls[2][calls[2].index("--date") + 1] == "2026-05-24"
    assert "--threshold-offsets=-2,0,2" in calls[0]
    assert "--update-observation-store" in calls[0]
    assert "--fetch-live" in calls[0]
    assert "--include-nws-guidance" in calls[0]
    manifest = json.loads((out_dir / "weather_desk_backfill_manifest.json").read_text())
    assert manifest["git_commit"] == "abc123"
    assert manifest["date_range"] == {"start": "2026-05-22", "end": "2026-05-24"}
    assert manifest["decision_minute"] == 20
    assert manifest["exit_code"] == 0


def test_weather_desk_backfill_cli_stops_on_failure_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_schedule_main(argv):
        return 1 if argv[argv.index("--date") + 1] == "2026-05-23" else 0

    monkeypatch.setattr(
        "src.weather_desk_backfill_cli.weather_desk_schedule_cli.main",
        fake_schedule_main,
    )

    out_dir = tmp_path / "backfill"
    code = weather_desk_backfill_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--start-date",
            "2026-05-22",
            "--end-date",
            "2026-05-24",
            "--out-dir",
            str(out_dir),
        ]
    )

    manifest = json.loads((out_dir / "weather_desk_backfill_manifest.json").read_text())
    assert code == 1
    assert [run["target_date"] for run in manifest["runs"]] == [
        "2026-05-22",
        "2026-05-23",
    ]
    assert manifest["exit_code"] == 1


def test_weather_desk_backfill_cli_can_continue_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_schedule_main(argv):
        return 1 if argv[argv.index("--date") + 1] == "2026-05-23" else 0

    monkeypatch.setattr(
        "src.weather_desk_backfill_cli.weather_desk_schedule_cli.main",
        fake_schedule_main,
    )

    out_dir = tmp_path / "backfill"
    code = weather_desk_backfill_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--start-date",
            "2026-05-22",
            "--end-date",
            "2026-05-24",
            "--continue-on-error",
            "--out-dir",
            str(out_dir),
        ]
    )

    manifest = json.loads((out_dir / "weather_desk_backfill_manifest.json").read_text())
    assert code == 1
    assert [run["target_date"] for run in manifest["runs"]] == [
        "2026-05-22",
        "2026-05-23",
        "2026-05-24",
    ]

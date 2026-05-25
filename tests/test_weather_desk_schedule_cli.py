import json
from pathlib import Path

from src import weather_desk_schedule_cli
from src.models.nbm_guidance import NOMADS_BLEND_BASE_URL
from src.models.station_rules import DEFAULT_STATION_RULES_PATH


def test_weather_desk_schedule_cli_runs_city_local_time_slices(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    calls = []

    def fake_refresh_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(
        "src.weather_desk_schedule_cli.weather_desk_refresh_cli.main",
        fake_refresh_main,
    )
    monkeypatch.setattr("src.weather_desk_schedule_cli._git_commit", lambda: "abc123")

    out_dir = tmp_path / "schedule"
    code = weather_desk_schedule_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--cities",
            "nyc,la",
            "--date",
            "2026-05-24",
            "--decision-hours",
            "07,10",
            "--threshold-offsets",
            "-2,0,2",
            "--multi-source-mode",
            "blend_equal",
            "--observation-store",
            str(tmp_path / "obs.csv"),
            "--include-nws-guidance",
            "--include-nbm-guidance",
            "--no-require-gate",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    assert len(calls) == 4
    nyc_07 = calls[0]
    la_07 = calls[1]
    assert nyc_07[nyc_07.index("--cities") + 1] == "nyc"
    assert nyc_07[nyc_07.index("--as-of") + 1] == "2026-05-24T11:20:00+00:00"
    assert nyc_07[nyc_07.index("--decision-time-label") + 1] == "07"
    assert nyc_07[nyc_07.index("--out-dir") + 1] == str(out_dir / "07_local" / "nyc")
    assert "--threshold-offsets=-2,0,2" in nyc_07
    assert "--include-nws-guidance" in nyc_07
    assert "--include-nbm-guidance" in nyc_07
    assert nyc_07[nyc_07.index("--nbm-base-url") + 1] == NOMADS_BLEND_BASE_URL
    assert "--no-require-gate" in nyc_07
    assert la_07[la_07.index("--cities") + 1] == "la"
    assert la_07[la_07.index("--as-of") + 1] == "2026-05-24T14:20:00+00:00"
    manifest = json.loads((out_dir / "weather_desk_schedule_manifest.json").read_text())
    assert manifest["git_commit"] == "abc123"
    assert manifest["decision_time_labels"] == ["07", "10"]
    assert manifest["decision_minute"] == 20
    assert manifest["nbm_base_url"] == NOMADS_BLEND_BASE_URL
    assert manifest["packet_layout"] == "one directory per decision_time_label/city"
    assert manifest["runs"][0]["local_minute"] == 20
    assert manifest["runs"][0]["out_dir"] == str(out_dir / "07_local" / "nyc")
    assert "Wrote weather desk schedule manifest" in capsys.readouterr().out


def test_weather_desk_schedule_cli_accepts_explicit_decision_minute(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_refresh_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(
        "src.weather_desk_schedule_cli.weather_desk_refresh_cli.main",
        fake_refresh_main,
    )

    out_dir = tmp_path / "schedule"
    code = weather_desk_schedule_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--cities",
            "nyc",
            "--date",
            "2026-05-24",
            "--decision-hours",
            "13",
            "--decision-minute",
            "5",
            "--station-rules",
            str(DEFAULT_STATION_RULES_PATH),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    assert calls[0][calls[0].index("--as-of") + 1] == "2026-05-24T17:05:00+00:00"
    manifest = json.loads((out_dir / "weather_desk_schedule_manifest.json").read_text())
    assert manifest["decision_minute"] == 5
    assert manifest["runs"][0]["local_minute"] == 5


def test_weather_desk_schedule_cli_returns_failure_if_any_slice_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_refresh_main(argv):
        return 1 if argv[argv.index("--cities") + 1] == "boston" else 0

    monkeypatch.setattr(
        "src.weather_desk_schedule_cli.weather_desk_refresh_cli.main",
        fake_refresh_main,
    )

    out_dir = tmp_path / "schedule"
    code = weather_desk_schedule_cli.main(
        [
            "--model-run-dir",
            str(tmp_path / "run"),
            "--cities",
            "nyc,boston",
            "--date",
            "2026-05-24",
            "--decision-hours",
            "07",
            "--station-rules",
            str(DEFAULT_STATION_RULES_PATH),
            "--out-dir",
            str(out_dir),
        ]
    )

    manifest = json.loads((out_dir / "weather_desk_schedule_manifest.json").read_text())
    assert code == 1
    assert manifest["exit_code"] == 1
    assert [run["exit_code"] for run in manifest["runs"]] == [0, 1]

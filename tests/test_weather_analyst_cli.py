from pathlib import Path

from src.weather_analyst_cli import main
from tests.test_weather_analyst import _guidance, _summary


def test_weather_analyst_cli_writes_packet(tmp_path: Path, capsys) -> None:
    summary_path = tmp_path / "nowcast_report_summary.csv"
    guidance_path = tmp_path / "model_vs_nws_guidance.csv"
    out_dir = tmp_path / "analyst"
    _summary().to_csv(summary_path, index=False)
    _guidance(3.5).to_csv(guidance_path, index=False)

    exit_code = main(
        [
            "--nowcast-summary",
            str(summary_path),
            "--guidance-comparison",
            str(guidance_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "weather_analyst_packet.csv").exists()
    assert (out_dir / "weather_analyst_packet.md").exists()
    assert (out_dir / "weather_analyst_manifest.json").exists()
    assert "Wrote 1 weather analyst rows" in capsys.readouterr().out

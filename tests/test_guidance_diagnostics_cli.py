from pathlib import Path

import pandas as pd

from src.guidance_diagnostics_cli import main


def test_guidance_diagnostics_cli_writes_outputs(tmp_path: Path, capsys) -> None:
    source = tmp_path / "guidance.csv"
    out_dir = tmp_path / "guidance_out"
    pd.DataFrame(
        [
            {
                "city": "boston",
                "source": "lamp",
                "station_id": "KBOS",
                "market_type": "high",
                "target_date": "2026-05-24",
                "issue_ts_utc": "2026-05-24T06:00:00Z",
                "valid_ts_utc": "2026-05-25T00:00:00Z",
                "available_ts_utc": "2026-05-24T06:30:00Z",
                "guidance_point_f": 56,
                "actual_high_f": 57,
            }
        ]
    ).to_csv(source, index=False)

    exit_code = main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(out_dir),
            "--as-of",
            "2026-05-24T07:00:00Z",
            "--target-date",
            "2026-05-24",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "guidance_latest.csv").exists()
    assert (out_dir / "guidance_score_summary.csv").exists()
    assert (out_dir / "guidance_report.md").exists()
    assert (out_dir / "guidance_manifest.json").exists()
    assert "1 latest rows, 1 summary rows" in capsys.readouterr().out

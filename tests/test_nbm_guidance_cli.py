from pathlib import Path

import pandas as pd

from src.nbm_guidance_cli import main


def test_nbm_guidance_cli_writes_rows(tmp_path: Path, monkeypatch, capsys) -> None:
    out = tmp_path / "nbm.csv"

    def fake_write_nbm_guidance_rows(
        *,
        output_path,
        target,
        as_of_ts,
        station_rules_path,
        cities=None,
    ):
        assert target.isoformat() == "2026-05-25"
        assert as_of_ts == "2026-05-25T18:00:00Z"
        assert cities == ["nyc"]
        rows = pd.DataFrame(
            [
                {
                    "city": "nyc",
                    "source": "nbm_text",
                    "station_id": "KNYC",
                    "market_type": "high",
                    "target_date": "2026-05-25",
                    "issue_ts_utc": "2026-05-25T13:00:00+00:00",
                    "valid_ts_utc": "2026-05-26T01:00:00+00:00",
                    "available_ts_utc": "2026-05-25T13:00:00+00:00",
                    "guidance_point_f": 74,
                    "guidance_q10_f": 70,
                    "guidance_q50_f": 74,
                    "guidance_q90_f": 78,
                    "actual_high_f": None,
                    "raw_payload_hash": "abc",
                }
            ]
        )
        rows.to_csv(output_path, index=False)
        return rows

    monkeypatch.setattr(
        "src.nbm_guidance_cli.write_nbm_guidance_rows",
        fake_write_nbm_guidance_rows,
    )

    exit_code = main(
        [
            "--date",
            "2026-05-25",
            "--as-of",
            "2026-05-25T18:00:00Z",
            "--cities",
            "nyc",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    assert pd.read_csv(out)["source"].tolist() == ["nbm_text"]
    assert "Wrote 1 NBM guidance rows" in capsys.readouterr().out

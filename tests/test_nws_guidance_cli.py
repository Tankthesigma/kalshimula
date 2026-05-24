from pathlib import Path

import pandas as pd

from src.nws_guidance_cli import main


def test_nws_guidance_cli_writes_rows(tmp_path: Path, monkeypatch, capsys) -> None:
    out = tmp_path / "nws_guidance.csv"

    def fake_write_nws_guidance_rows(*, output_path, target, cities, market_types, fetched_at):
        assert target.isoformat() == "2026-05-24"
        assert cities == ["nyc", "boston"]
        assert market_types == ["high", "low"]
        pd.DataFrame(
            [
                {
                    "city": "nyc",
                    "source": "nws_forecast",
                    "station_id": "KNYC",
                    "market_type": "high",
                    "target_date": "2026-05-24",
                    "issue_ts_utc": "2026-05-24T06:30:00+00:00",
                    "valid_ts_utc": "2026-05-25T00:00:00+00:00",
                    "available_ts_utc": "2026-05-24T06:30:00+00:00",
                    "guidance_point_f": 73,
                    "guidance_q10_f": None,
                    "guidance_q50_f": 73,
                    "guidance_q90_f": None,
                    "actual_high_f": None,
                    "raw_payload_hash": "abc",
                }
            ]
        ).to_csv(output_path, index=False)
        return pd.read_csv(output_path)

    monkeypatch.setattr(
        "src.nws_guidance_cli.write_nws_guidance_rows",
        fake_write_nws_guidance_rows,
    )

    exit_code = main(
        [
            "--date",
            "2026-05-24",
            "--cities",
            "nyc,boston",
            "--market-type",
            "both",
            "--fetched-at",
            "2026-05-24T07:00:00Z",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    assert out.exists()
    assert "Wrote 1 NWS guidance rows" in capsys.readouterr().out

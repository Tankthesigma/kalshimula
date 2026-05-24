import json

import pandas as pd

from src.tomorrow_decision_packet_cli import main


def test_tomorrow_decision_packet_cli_writes_model_only_packet(tmp_path, capsys) -> None:
    predictions = tmp_path / "predictions.json"
    contrarian = tmp_path / "contrarian.csv"
    walkforward = tmp_path / "walkforward.csv"
    policy = tmp_path / "policy.csv"
    out_md = tmp_path / "packet.md"
    out_csv = tmp_path / "packet.csv"

    predictions.write_text(
        json.dumps(
            {
                "target_date": "2026-05-25",
                "predictions": [
                    {
                        "city": "denver",
                        "target_date": "2026-05-25",
                        "selected_source": "gfs_ens",
                        "forecast": {"point_f": 80},
                        "calibration": {
                            "corrected_point_f": 81,
                            "interval_lower_f": 78,
                            "interval_upper_f": 84,
                        },
                        "multi_source": {"calibration": {"corrected_point_f": 79}},
                        "threshold_probabilities": [{"threshold_f": 80, "probability": 0.6}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "promoted": True,
                "contrarian_correct_rate": 0.6,
                "contrarian_correct_ci_lower_95": 0.51,
                "mean_abs_delta_f": 1.2,
            }
        ]
    ).to_csv(contrarian, index=False)
    pd.DataFrame(
        [{"city": "denver", "source": "gfs_ens", "mae": 1.2, "brier_raw": 0.2}]
    ).to_csv(walkforward, index=False)
    pd.DataFrame([{"source": "gfs_ens", "mae": 1.2}]).to_csv(policy, index=False)

    exit_code = main(
        [
            "--predictions",
            str(predictions),
            "--source-contrarian-summary",
            str(contrarian),
            "--walkforward-summary",
            str(walkforward),
            "--policy-leaderboard",
            str(policy),
            "--out-md",
            str(out_md),
            "--out-csv",
            str(out_csv),
        ]
    )

    assert exit_code == 0
    assert "No Kalshi prices" in out_md.read_text()
    output = pd.read_csv(out_csv)
    assert output.iloc[0]["manual_priority"] == "high"
    assert "Wrote tomorrow model packet" in capsys.readouterr().out

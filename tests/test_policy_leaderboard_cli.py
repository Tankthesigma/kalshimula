import pandas as pd

from src.policy_leaderboard_cli import main


def test_policy_leaderboard_cli_writes_outputs(tmp_path, capsys) -> None:
    walkforward = tmp_path / "walkforward.csv"
    output_dir = tmp_path / "policy"
    pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "n_predictions": 10,
                "n_events": 70,
                "mae": 1.0,
                "bias": 0.1,
                "brier_raw": 0.2,
                "ece_raw": 0.05,
                "logloss_raw": 0.6,
                "stability_score": 0.2,
            }
        ]
    ).to_csv(walkforward, index=False)

    exit_code = main(
        [
            "--walkforward-summary",
            str(walkforward),
            "--out-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "policy_leaderboard.csv").exists()
    assert (output_dir / "policy_leaderboard.md").exists()
    assert "Wrote policy leaderboard" in capsys.readouterr().out

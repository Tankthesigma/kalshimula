import pandas as pd

from src.models.policy_leaderboard import build_policy_leaderboard


def test_build_policy_leaderboard_ranks_by_mae_and_counts_promoted() -> None:
    walkforward = pd.DataFrame(
        [
            _row("denver", "gfs_ens", 1.0),
            _row("nyc", "gfs_ens", 2.0),
            _row("denver", "openmeteo_naive", 1.5),
            _row("nyc", "openmeteo_naive", 2.5),
        ]
    )
    contrarian = pd.DataFrame(
        [
            {"city": "denver", "source": "gfs_ens", "promoted": True},
            {"city": "nyc", "source": "gfs_ens", "promoted": False},
        ]
    )

    leaderboard = build_policy_leaderboard(walkforward, source_contrarian_summary=contrarian)

    assert leaderboard.iloc[0]["source"] == "gfs_ens"
    assert leaderboard.iloc[0]["promoted_city_sources"] == 1
    assert leaderboard.iloc[0]["worst_city"] == "nyc"


def _row(city: str, source: str, mae: float) -> dict:
    return {
        "city": city,
        "source": source,
        "n_predictions": 10,
        "n_events": 70,
        "mae": mae,
        "bias": 0.1,
        "brier_raw": 0.2 + mae / 100,
        "ece_raw": 0.05,
        "logloss_raw": 0.6,
        "stability_score": 0.2,
    }

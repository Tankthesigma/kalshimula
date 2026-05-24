import pandas as pd

from src.models.climate_features import build_climate_feature_diagnostics


def test_rolling_features_use_only_prior_dates() -> None:
    rows = _rows(35)

    diagnostics = build_climate_feature_diagnostics(rows)
    day_31 = diagnostics.features[diagnostics.features["target_date"] == "2025-01-31"].iloc[0]

    assert day_31["rolling_30d_actual_f"] == 14.5
    assert day_31["rolling_30d_anomaly_f"] == 15.5


def test_insufficient_history_returns_missing_features() -> None:
    diagnostics = build_climate_feature_diagnostics(_rows(10))

    first = diagnostics.features.iloc[0]

    assert pd.isna(first["historical_normal_f"])
    assert pd.isna(first["rolling_30d_actual_f"])
    assert first["feature_missing"]


def test_season_encoding_is_stable() -> None:
    diagnostics = build_climate_feature_diagnostics(_rows(2))

    row = diagnostics.features.iloc[0]

    assert -1 <= row["day_of_year_sin"] <= 1
    assert -1 <= row["day_of_year_cos"] <= 1


def test_no_target_date_leakage_into_historical_normal() -> None:
    rows = pd.DataFrame(
        [
            {"city": "denver", "target_date": "2024-01-01", "actual_high_f": 10},
            {"city": "denver", "target_date": "2025-01-01", "actual_high_f": 100},
        ]
    )

    diagnostics = build_climate_feature_diagnostics(rows)
    second = diagnostics.features.iloc[1]

    assert second["historical_normal_f"] == 10
    assert second["climatology_error_baseline_f"] == 90


def _rows(days: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": "denver",
                "target_date": target_date.date().isoformat(),
                "actual_high_f": float(index),
            }
            for index, target_date in enumerate(pd.date_range("2025-01-01", periods=days))
        ]
    )

import pandas as pd
import pytest

from src.models.source_contrarian_diagnostics import (
    build_daily_source_deltas,
    build_monthly_source_metrics,
    build_source_contrarian_summary,
    build_source_threshold_grid,
    wilson_interval,
)


def test_openmeteo_naive_is_preferred_as_blend() -> None:
    daily = build_daily_source_deltas(
        pd.DataFrame(
            [
                _row("denver", "2025-01-01", "openmeteo_naive", 70, 73),
                _row("denver", "2025-01-01", "gfs_ens", 72, 73),
                _row("denver", "2025-01-01", "ecmwf_ens", 60, 73),
            ]
        )
    )

    gfs = daily[daily["source"] == "gfs_ens"].iloc[0]

    assert gfs["blend_source"] == "openmeteo_naive"
    assert gfs["blend_point_f"] == pytest.approx(70)
    assert gfs["signed_delta_f"] == pytest.approx(2)


def test_fallback_consensus_excludes_candidate_source() -> None:
    daily = build_daily_source_deltas(
        pd.DataFrame(
            [
                _row("denver", "2025-01-01", "gfs_ens", 72, 73),
                _row("denver", "2025-01-01", "ecmwf_ens", 70, 73),
                _row("denver", "2025-01-01", "icon_ens", 68, 73),
            ]
        )
    )

    gfs = daily[daily["source"] == "gfs_ens"].iloc[0]

    assert gfs["blend_source"] == "computed_equal_primary"
    assert gfs["blend_point_f"] == pytest.approx(69)


def test_contrarian_correct_true_when_source_delta_matches_actual_direction() -> None:
    daily = build_daily_source_deltas(
        pd.DataFrame(
            [
                _row("denver", "2025-01-01", "openmeteo_naive", 70, 73),
                _row("denver", "2025-01-01", "gfs_ens", 72, 73),
            ]
        )
    )

    row = daily.iloc[0]

    assert row["source_residual_f"] == pytest.approx(1)
    assert row["blend_residual_f"] == pytest.approx(3)
    assert row["contrarian_correct"] == True  # noqa: E712


def test_contrarian_correct_false_when_source_delta_opposes_actual_direction() -> None:
    daily = build_daily_source_deltas(
        pd.DataFrame(
            [
                _row("denver", "2025-01-01", "openmeteo_naive", 70, 68),
                _row("denver", "2025-01-01", "gfs_ens", 72, 68),
            ]
        )
    )

    assert daily.iloc[0]["contrarian_correct"] == False  # noqa: E712


def test_equal_source_and_blend_point_has_missing_contrarian_correct() -> None:
    daily = build_daily_source_deltas(
        pd.DataFrame(
            [
                _row("denver", "2025-01-01", "openmeteo_naive", 70, 72),
                _row("denver", "2025-01-01", "gfs_ens", 70, 72),
            ]
        )
    )

    assert pd.isna(daily.iloc[0]["contrarian_correct"])


def test_monthly_aggregation_excludes_missing_contrarian_denominator() -> None:
    daily = build_daily_source_deltas(
        pd.DataFrame(
            [
                _row("denver", "2025-01-01", "openmeteo_naive", 70, 72),
                _row("denver", "2025-01-01", "gfs_ens", 70, 72),
                _row("denver", "2025-01-02", "openmeteo_naive", 70, 73),
                _row("denver", "2025-01-02", "gfs_ens", 72, 73),
            ]
        )
    )

    monthly = build_monthly_source_metrics(daily)

    assert monthly.iloc[0]["n_days"] == 2
    assert monthly.iloc[0]["contrarian_n"] == 1
    assert monthly.iloc[0]["contrarian_correct_n"] == 1


def test_wilson_interval_sanity() -> None:
    lower, upper = wilson_interval(60, 100)

    assert 0.50 < lower < 0.60
    assert 0.60 < upper < 0.70


def test_promotion_rule_true_when_all_criteria_pass() -> None:
    rows = []
    for day, target_date in enumerate(pd.date_range("2025-01-01", periods=220)):
        actual = 73 if day < 132 else 68
        rows.extend(
            [
                _row("denver", target_date.date().isoformat(), "openmeteo_naive", 70, actual),
                _row("denver", target_date.date().isoformat(), "gfs_ens", 72, actual),
            ]
        )
    daily = build_daily_source_deltas(pd.DataFrame(rows))

    summary = build_source_contrarian_summary(daily)

    assert summary.iloc[0]["promoted"] == True  # noqa: E712
    assert summary.iloc[0]["promote_reason"] == "promoted"


def test_promotion_rule_false_for_low_sample() -> None:
    daily = build_daily_source_deltas(
        pd.DataFrame(
            [
                _row("denver", "2025-01-01", "openmeteo_naive", 70, 73),
                _row("denver", "2025-01-01", "gfs_ens", 72, 73),
            ]
        )
    )

    summary = build_source_contrarian_summary(daily)

    assert summary.iloc[0]["promoted"] == False  # noqa: E712
    assert "low_n" in summary.iloc[0]["promote_reason"]


def test_threshold_grid_reports_probability_and_brier_columns() -> None:
    daily = build_daily_source_deltas(
        pd.DataFrame(
            [
                _row("denver", "2025-01-01", "openmeteo_naive", 70, 73),
                _row("denver", "2025-01-01", "gfs_ens", 72, 73),
                _row("denver", "2025-01-02", "openmeteo_naive", 70, 68),
                _row("denver", "2025-01-02", "gfs_ens", 72, 68),
            ]
        )
    )

    grid = build_source_threshold_grid(daily, offsets=(0,))

    assert len(grid) == 1
    assert grid.iloc[0]["offset_f"] == pytest.approx(0)
    assert 0 <= grid.iloc[0]["mean_source_prob_above"] <= 1
    assert 0 <= grid.iloc[0]["source_brier"] <= 1


def _row(city: str, target_date: str, source: str, point: float, actual: float) -> dict:
    return {
        "city": city,
        "target_date": target_date,
        "source": source,
        "point_f": point,
        "actual_high_f": actual,
    }

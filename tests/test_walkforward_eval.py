import pandas as pd
import pytest

from src.models.walkforward_eval import evaluate_walkforward


def test_walkforward_split_never_uses_future_rows() -> None:
    rows = _rows(days=60, sources=("gfs_ens",))

    result = evaluate_walkforward(
        rows,
        cities=("denver",),
        sources=("gfs_ens",),
        train_window_days=30,
        test_window_days=7,
        step_days=7,
        threshold_offsets=(0,),
    )

    assert not result.predictions.empty
    parsed = result.predictions.copy()
    parsed["target_date"] = pd.to_datetime(parsed["target_date"])
    parsed["train_end"] = pd.to_datetime(parsed["train_end"])
    assert (parsed["train_end"] < parsed["target_date"]).all()


def test_blocked_holdout_and_purge_exclude_target_rows() -> None:
    rows = _rows(days=80, sources=("gfs_ens",))

    result = evaluate_walkforward(
        rows,
        cities=("denver",),
        sources=("gfs_ens",),
        train_window_days=30,
        test_window_days=30,
        step_days=30,
        threshold_offsets=(0,),
        holdout_start="2025-02-10",
        holdout_end="2025-02-12",
        purge_days_before=1,
        purge_days_after=1,
    )

    dates = set(result.predictions["target_date"])

    assert "2025-02-09" not in dates
    assert "2025-02-10" not in dates
    assert "2025-02-12" not in dates
    assert "2025-02-13" not in dates


def test_threshold_outcome_and_brier_are_computed() -> None:
    rows = _rows(days=50, sources=("gfs_ens",))

    result = evaluate_walkforward(
        rows,
        cities=("denver",),
        sources=("gfs_ens",),
        train_window_days=30,
        test_window_days=5,
        step_days=5,
        threshold_offsets=(0,),
    )

    event = result.events.iloc[0]

    assert event["outcome"] in {True, False}
    assert 0 <= event["predicted_probability"] <= 1
    assert 0 <= result.city_source_summary.iloc[0]["brier_raw"] <= 1
    assert result.city_source_summary.iloc[0]["logloss_raw"] >= 0


def test_blend_equal_policy_uses_multiple_sources() -> None:
    rows = _rows(days=50, sources=("gfs_ens", "ecmwf_ens"))
    rows.loc[rows["source"] == "ecmwf_ens", "point_f"] += 2

    result = evaluate_walkforward(
        rows,
        cities=("denver",),
        sources=("blend_equal",),
        train_window_days=30,
        test_window_days=5,
        step_days=5,
        threshold_offsets=(0,),
    )

    assert set(result.predictions["source"]) == {"blend_equal"}
    assert result.predictions.iloc[0]["point_f"] == pytest.approx(
        rows[rows["target_date"] == result.predictions.iloc[0]["target_date"]]["point_f"].mean()
    )


def _rows(days: int, sources: tuple[str, ...]) -> pd.DataFrame:
    out = []
    for day, target_date in enumerate(pd.date_range("2025-01-01", periods=days)):
        actual = 70 + (day % 5)
        for source in sources:
            point = actual - 1 if source == "gfs_ens" else actual + 1
            out.append(
                {
                    "city": "denver",
                    "target_date": target_date.date().isoformat(),
                    "source": source,
                    "point_f": point,
                    "actual_high_f": actual,
                }
            )
    return pd.DataFrame(out)

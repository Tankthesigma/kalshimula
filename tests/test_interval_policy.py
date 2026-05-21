import pandas as pd

from src.models.interval_policy import (
    compare_interval_policies,
    filter_rows_to_recommended_sources,
    write_interval_policy_outputs,
)


def _rows() -> pd.DataFrame:
    records = [
        _row("denver", "gfs_ens", "2025-01-01", 70, 70),
        _row("denver", "gfs_ens", "2025-01-02", 70, 72),
        _row("denver", "gfs_ens", "2025-01-03", 70, 74),
        _row("denver", "gfs_ens", "2025-01-04", 70, 78),
        _row("denver", "gfs_ens", "2025-01-05", 70, 74),
        _row("denver", "gfs_ens", "2025-01-06", 70, 76),
        _row("denver", "gfs_ens", "2025-01-07", 70, 74),
        _row("denver", "gfs_ens", "2025-01-08", 70, 76),
        _row("denver", "openmeteo_naive", "2025-01-07", 10, 20),
    ]
    return pd.DataFrame(records)


def _row(
    city: str, source: str, target_date: str, point_f: float, actual_high_f: float
) -> dict:
    return {
        "city": city,
        "source": source,
        "target_date": target_date,
        "point_f": point_f,
        "actual_high_f": actual_high_f,
        "absolute_error_f": abs(point_f - actual_high_f),
    }


def _recommended_sources() -> pd.DataFrame:
    return pd.DataFrame([{"city": "denver", "selected_source": "gfs_ens"}])


def test_filter_rows_to_recommended_sources_is_reexported() -> None:
    filtered = filter_rows_to_recommended_sources(_rows(), _recommended_sources())

    assert set(filtered["source"]) == {"gfs_ens"}


def test_compare_interval_policies_selects_per_city_alpha() -> None:
    result = compare_interval_policies(
        rows=_rows(),
        recommended_sources=_recommended_sources(),
        validation_start="2025-01-05",
        test_start="2025-01-07",
        alphas=(0.5, 0.1),
        target_coverage=0.8,
    )

    selected = result.selected_policy.iloc[0]
    assert selected["selected_alpha"] == 0.1
    assert selected["selection_reason"] == "narrowest_meeting_target"
    assert not result.interval_table.empty
    assert "per_city_alpha" in set(result.comparison["policy"])


def test_write_interval_policy_outputs(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    source_path = tmp_path / "recommended_sources.csv"
    output_dir = tmp_path / "model_policy"
    _rows().to_csv(input_path, index=False)
    _recommended_sources().to_csv(source_path, index=False)

    result = write_interval_policy_outputs(
        input_path=input_path,
        recommended_sources_path=source_path,
        output_dir=output_dir,
        validation_start="2025-01-05",
        test_start="2025-01-07",
        alphas=(0.5, 0.1),
    )

    assert not result.selected_policy.empty
    assert (output_dir / "interval_policy.csv").exists()
    assert (output_dir / "interval_policy_comparison.csv").exists()
    assert (output_dir / "interval_table.csv").exists()

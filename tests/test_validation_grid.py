import pandas as pd
import pytest

from src.models.validation_grid import evaluate_recency_alpha_grid, write_recency_alpha_grid_outputs


def _rows() -> pd.DataFrame:
    records = [
        _row("2025-01-01", 100, 50),
        _row("2025-01-02", 100, 50),
        _row("2025-01-03", 70, 72),
        _row("2025-01-04", 70, 72),
        _row("2025-01-05", 70, 72),
        _row("2025-01-06", 70, 72),
        _row("2025-01-07", 70, 72),
        _row("2025-01-08", 70, 72),
    ]
    return pd.DataFrame(records)


def _row(target_date: str, point_f: float, actual_high_f: float) -> dict:
    return {
        "city": "denver",
        "target_date": target_date,
        "source": "gfs_ens",
        "point_f": point_f,
        "actual_high_f": actual_high_f,
        "absolute_error_f": abs(point_f - actual_high_f),
    }


def test_evaluate_recency_alpha_grid_selects_best_validation_config() -> None:
    result = evaluate_recency_alpha_grid(
        _rows(),
        validation_start="2025-01-05",
        test_start="2025-01-07",
        recent_days=(2, 4),
        alphas=(0.2,),
        target_coverage=0.8,
    )

    assert len(result.validation_grid) == 2
    assert len(result.test_grid) == 2
    selected = result.selected_config.iloc[0]
    assert selected["bias_recent_days"] == 2
    assert selected["validation_mae_corrected"] == pytest.approx(0.0)

    old_window = result.validation_grid[
        result.validation_grid["bias_recent_days"] == 4
    ].iloc[0]
    assert old_window["mae_corrected"] > selected["validation_mae_corrected"]


def test_evaluate_recency_alpha_grid_rejects_empty_validation_slice() -> None:
    with pytest.raises(ValueError, match="validation slice is empty"):
        evaluate_recency_alpha_grid(
            _rows(),
            validation_start="2025-01-07",
            test_start="2025-01-07",
            recent_days=(2,),
            alphas=(0.2,),
        )


def test_write_recency_alpha_grid_outputs_filters_source(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "grid"
    rows = pd.concat(
        [
            _rows(),
            _rows().assign(source="openmeteo_naive", point_f=10, actual_high_f=20),
        ],
        ignore_index=True,
    )
    rows.to_csv(input_path, index=False)

    result = write_recency_alpha_grid_outputs(
        input_path=input_path,
        output_dir=output_dir,
        validation_start="2025-01-05",
        test_start="2025-01-07",
        recent_days=(2,),
        alphas=(0.2, 0.13),
        source="gfs_ens",
    )

    assert len(result.validation_grid) == 2
    assert (output_dir / "validation_grid.csv").exists()
    assert (output_dir / "test_grid.csv").exists()
    assert (output_dir / "selected_config.csv").exists()
    assert result.validation_grid["n_rows"].tolist() == [2, 2]


def test_write_recency_alpha_grid_outputs_can_write_policy_artifacts(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "grid"
    policy_dir = tmp_path / "model_policy"
    _rows().to_csv(input_path, index=False)

    write_recency_alpha_grid_outputs(
        input_path=input_path,
        output_dir=output_dir,
        policy_out_dir=policy_dir,
        validation_start="2025-01-05",
        test_start="2025-01-07",
        recent_days=(2,),
        alphas=(0.2,),
        source="gfs_ens",
    )

    policy = pd.read_csv(policy_dir / "model_policy.csv")
    bias_table = pd.read_csv(policy_dir / "bias_table.csv")
    interval_table = pd.read_csv(policy_dir / "interval_table.csv")
    assert policy.iloc[0]["source"] == "gfs_ens"
    assert policy.iloc[0]["bias_recent_days"] == 2
    assert bias_table.iloc[0]["bias_correction_f"] == pytest.approx(2.0)
    assert interval_table.iloc[0]["alpha"] == pytest.approx(0.2)

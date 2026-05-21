import pandas as pd

from src.models.bias_policy import (
    compare_bias_policies,
    filter_rows_to_recommended_sources,
    write_bias_policy_outputs,
)
from src.models.train_eval import train_eval_split


def _rows() -> pd.DataFrame:
    records = [
        _row("denver", "gfs_ens", "2025-01-01", 100, 50),
        _row("denver", "gfs_ens", "2025-01-02", 100, 50),
        _row("denver", "gfs_ens", "2025-01-03", 70, 72),
        _row("denver", "gfs_ens", "2025-01-04", 70, 72),
        _row("denver", "gfs_ens", "2025-01-05", 70, 72),
        _row("denver", "gfs_ens", "2025-01-06", 70, 72),
        _row("denver", "gfs_ens", "2025-01-07", 70, 72),
        _row("denver", "gfs_ens", "2025-01-08", 70, 72),
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
    return pd.DataFrame(
        [
            {
                "city": "denver",
                "selected_source": "gfs_ens",
                "recommended_policy": "best_global_validation_source",
            }
        ]
    )


def test_filter_rows_to_recommended_sources() -> None:
    filtered = filter_rows_to_recommended_sources(_rows(), _recommended_sources())

    assert set(filtered["source"]) == {"gfs_ens"}
    assert len(filtered) == 8


def test_compare_bias_policies_writes_recommended_global_policy() -> None:
    rows = _rows()
    train_eval = train_eval_split(
        rows[rows["source"] == "gfs_ens"],
        validation_start="2025-01-05",
        test_start="2025-01-07",
    )

    result = compare_bias_policies(
        rows=rows,
        recommended_sources=_recommended_sources(),
        evaluation=train_eval.evaluation,
        selected_methods=train_eval.selected_methods,
        validation_start="2025-01-05",
        test_start="2025-01-07",
        recent_days=(2, 4),
        alphas=(0.2,),
    )

    assert result.comparison["policy"].tolist() == [
        "per_city_bias_selection",
        "global_recent_2d",
        "global_recent_4d",
    ]
    assert result.recommended_policy.iloc[0]["policy"] == "global_recent_2d"
    assert not result.recommended_bias_table.empty
    assert not result.recommended_interval_table.empty


def test_write_bias_policy_outputs(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    train_eval_dir = tmp_path / "train_eval"
    source_dir = tmp_path / "source_selection"
    out_dir = tmp_path / "model_policy"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    train_eval_dir.mkdir()
    source_dir.mkdir()

    rows = _rows()
    rows.to_csv(input_path, index=False)
    train_eval = train_eval_split(
        rows[rows["source"] == "gfs_ens"],
        validation_start="2025-01-05",
        test_start="2025-01-07",
    )
    train_eval.evaluation.to_csv(train_eval_dir / "evaluation.csv", index=False)
    train_eval.selected_methods.to_csv(train_eval_dir / "selected_methods.csv", index=False)
    _recommended_sources().to_csv(source_dir / "recommended_sources.csv", index=False)

    result = write_bias_policy_outputs(
        input_path=input_path,
        train_eval_dir=train_eval_dir,
        recommended_sources_path=source_dir / "recommended_sources.csv",
        output_dir=out_dir,
        validation_start="2025-01-05",
        test_start="2025-01-07",
        recent_days=(2,),
        alphas=(0.2,),
    )

    assert not result.comparison.empty
    assert (out_dir / "bias_policy_comparison.csv").exists()
    assert (out_dir / "model_policy.csv").exists()
    assert (out_dir / "bias_table.csv").exists()
    assert (out_dir / "interval_table.csv").exists()

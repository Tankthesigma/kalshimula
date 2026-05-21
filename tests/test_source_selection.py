import pandas as pd
import pytest

from src.models.source_selection import (
    compare_source_policies,
    evaluate_selected_sources,
    recommend_sources,
    select_sources_by_validation,
    summarize_selected_sources,
    write_source_selection_outputs,
)


def test_select_sources_by_validation_picks_lowest_mae_per_city() -> None:
    validation_scores = pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "openmeteo_naive",
                "method": "recent_180d",
                "validation_mae": 1.2,
            },
            {
                "city": "denver",
                "source": "gfs_ens",
                "method": "recent_180d",
                "validation_mae": 0.8,
            },
            {
                "city": "denver",
                "source": "hrrr",
                "method": "recent_180d",
                "validation_mae": 0.8,
            },
            {
                "city": "nyc",
                "source": "openmeteo_naive",
                "method": "recent_180d",
                "validation_mae": 1.1,
            },
        ]
    )

    selected = select_sources_by_validation(validation_scores)

    denver = selected[selected["city"] == "denver"].iloc[0]
    nyc = selected[selected["city"] == "nyc"].iloc[0]
    assert denver["selected_source"] == "gfs_ens"
    assert denver["source_selection_bias_method"] == "recent_180d"
    assert not denver["source_selection_fallback"]
    assert nyc["selected_source"] == "openmeteo_naive"


def test_select_sources_by_validation_uses_model_method_tie_breaker() -> None:
    validation_scores = pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "method": "all_global",
                "validation_mae": 0.8,
            },
            {
                "city": "denver",
                "source": "gfs_ens",
                "method": "recent_180d",
                "validation_mae": 0.8,
            },
        ]
    )

    selected = select_sources_by_validation(validation_scores)

    assert selected.iloc[0]["selected_source"] == "gfs_ens"
    assert selected.iloc[0]["source_selection_bias_method"] == "recent_180d"


def test_select_sources_by_validation_falls_back_when_scores_are_missing() -> None:
    validation_scores = pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "method": "recent_180d",
                "validation_mae": pd.NA,
            },
            {
                "city": "denver",
                "source": "openmeteo_naive",
                "method": "recent_180d",
                "validation_mae": pd.NA,
            },
        ]
    )

    selected = select_sources_by_validation(validation_scores)

    assert selected.iloc[0]["selected_source"] == "openmeteo_naive"
    assert selected.iloc[0]["source_selection_fallback"]


def test_evaluate_selected_sources_joins_test_metrics() -> None:
    selected = pd.DataFrame(
        [
            {
                "city": "denver",
                "selected_source": "gfs_ens",
                "source_selection_bias_method": "recent_180d",
                "source_validation_mae": 0.8,
                "source_selection_fallback": False,
            }
        ]
    )
    evaluation = pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "mae_raw": 1.0,
                "mae_corrected": 0.7,
                "interval_coverage_raw": 0.8,
                "interval_width_raw": 3.0,
            },
            {
                "city": "denver",
                "source": "openmeteo_naive",
                "mae_raw": 1.5,
                "mae_corrected": 1.2,
                "interval_coverage_raw": 0.9,
                "interval_width_raw": 4.0,
            },
        ]
    )

    selected_eval = evaluate_selected_sources(evaluation, selected)

    assert selected_eval.iloc[0]["selected_source"] == "gfs_ens"
    assert selected_eval.iloc[0]["mae_corrected"] == pytest.approx(0.7)
    assert "source" not in selected_eval.columns


def test_summarize_selected_sources_averages_metrics() -> None:
    selected_eval = pd.DataFrame(
        [
            {
                "city": "denver",
                "mae_raw": 1.0,
                "mae_corrected": 0.5,
                "interval_coverage_raw": 0.8,
                "interval_width_raw": 3.0,
            },
            {
                "city": "nyc",
                "mae_raw": 2.0,
                "mae_corrected": 1.5,
                "interval_coverage_raw": 0.7,
                "interval_width_raw": 5.0,
            },
        ]
    )

    summary = summarize_selected_sources(selected_eval)

    assert summary.iloc[0]["n_cities"] == 2
    assert summary.iloc[0]["mae_corrected"] == pytest.approx(1.0)
    assert summary.iloc[0]["interval_coverage_raw"] == pytest.approx(0.75)


def test_compare_source_policies_adds_best_global_source() -> None:
    validation_scores = pd.DataFrame(
        [
            {"city": "denver", "source": "gfs_ens", "method": "recent_180d", "validation_mae": 0.7},
            {"city": "nyc", "source": "gfs_ens", "method": "recent_180d", "validation_mae": 0.9},
            {"city": "denver", "source": "openmeteo_naive", "method": "recent_180d", "validation_mae": 1.0},
            {"city": "nyc", "source": "openmeteo_naive", "method": "recent_180d", "validation_mae": 1.1},
        ]
    )
    evaluation = pd.DataFrame(
        [
            {"city": "denver", "source": "gfs_ens", "mae_raw": 1.0, "mae_corrected": 0.6, "interval_coverage_raw": 0.8, "interval_width_raw": 3.0},
            {"city": "nyc", "source": "gfs_ens", "mae_raw": 1.2, "mae_corrected": 0.8, "interval_coverage_raw": 0.9, "interval_width_raw": 4.0},
            {"city": "denver", "source": "openmeteo_naive", "mae_raw": 1.5, "mae_corrected": 1.1, "interval_coverage_raw": 0.7, "interval_width_raw": 5.0},
            {"city": "nyc", "source": "openmeteo_naive", "mae_raw": 1.7, "mae_corrected": 1.3, "interval_coverage_raw": 0.8, "interval_width_raw": 6.0},
        ]
    )
    selected_sources = pd.DataFrame(
        [
            {"city": "denver", "selected_source": "gfs_ens", "source_selection_bias_method": "recent_180d", "source_validation_mae": 0.7, "source_selection_fallback": False},
            {"city": "nyc", "selected_source": "openmeteo_naive", "source_selection_bias_method": "recent_180d", "source_validation_mae": 0.6, "source_selection_fallback": False},
        ]
    )
    selected_summary = pd.DataFrame(
        [{"n_cities": 2, "mae_raw": 1.3, "mae_corrected": 0.9, "interval_coverage_raw": 0.75, "interval_width_raw": 4.5}]
    )

    comparison = compare_source_policies(
        validation_scores=validation_scores,
        evaluation=evaluation,
        selected_sources=selected_sources,
        selected_summary=selected_summary,
    )

    global_policy = comparison[
        comparison["policy"] == "best_global_validation_source"
    ].iloc[0]
    assert global_policy["selected_source"] == "gfs_ens"
    assert global_policy["validation_mae"] == pytest.approx(0.8)
    assert global_policy["mae_corrected"] == pytest.approx(0.7)


def test_recommend_sources_maps_all_cities_to_best_global_policy() -> None:
    selected_sources = pd.DataFrame(
        [
            {"city": "denver", "selected_source": "gfs_ens"},
            {"city": "nyc", "selected_source": "openmeteo_naive"},
        ]
    )
    policy_comparison = pd.DataFrame(
        [
            {
                "policy": "per_city_validation",
                "selected_source": "per_city",
            },
            {
                "policy": "best_global_validation_source",
                "selected_source": "gfs_ens",
            },
        ]
    )

    recommended = recommend_sources(selected_sources, policy_comparison)

    assert recommended["city"].tolist() == ["denver", "nyc"]
    assert recommended["selected_source"].tolist() == ["gfs_ens", "gfs_ens"]
    assert recommended["recommended_policy"].tolist() == [
        "best_global_validation_source",
        "best_global_validation_source",
    ]


def test_write_source_selection_outputs(tmp_path) -> None:
    validation_scores_path = tmp_path / "validation_scores.csv"
    evaluation_path = tmp_path / "evaluation.csv"
    output_dir = tmp_path / "source_selection"
    pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "method": "recent_180d",
                "validation_mae": 0.8,
            }
        ]
    ).to_csv(validation_scores_path, index=False)
    pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "mae_raw": 1.0,
                "mae_corrected": 0.7,
                "interval_coverage_raw": 0.8,
                "interval_width_raw": 3.0,
            }
        ]
    ).to_csv(evaluation_path, index=False)

    result = write_source_selection_outputs(
        validation_scores_path=validation_scores_path,
        evaluation_path=evaluation_path,
        output_dir=output_dir,
    )

    assert len(result.selected_sources) == 1
    assert len(result.recommended_sources) == 1
    assert (output_dir / "selected_sources.csv").exists()
    assert (output_dir / "recommended_sources.csv").exists()
    assert (output_dir / "selected_source_evaluation.csv").exists()
    assert (output_dir / "selected_source_summary.csv").exists()
    assert (output_dir / "source_policy_comparison.csv").exists()
    assert not result.policy_comparison.empty

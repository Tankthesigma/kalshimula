"""Validation-driven forecast source selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_FALLBACK_SOURCE = "openmeteo_naive"
DEFAULT_METHOD_ORDER = (
    "recent_180d",
    "prior_same_month",
    "recent_global",
    "all_global",
)
SOURCE_SELECTION_COLUMNS = [
    "city",
    "selected_source",
    "source_selection_bias_method",
    "source_validation_mae",
    "source_selection_fallback",
]
RECOMMENDED_SOURCE_COLUMNS = [
    "city",
    "selected_source",
    "recommended_policy",
]
SUMMARY_COLUMNS = [
    "n_cities",
    "mae_raw",
    "mae_corrected",
    "interval_coverage_raw",
    "interval_width_raw",
]
POLICY_COMPARISON_COLUMNS = [
    "policy",
    "selected_source",
    "validation_mae",
    *SUMMARY_COLUMNS,
]


@dataclass(frozen=True)
class SourceSelectionResult:
    """Artifacts from validation-based source selection."""

    selected_sources: pd.DataFrame
    recommended_sources: pd.DataFrame
    selected_evaluation: pd.DataFrame
    summary: pd.DataFrame
    policy_comparison: pd.DataFrame


def select_sources_by_validation(
    validation_scores: pd.DataFrame,
    *,
    fallback_source: str = DEFAULT_FALLBACK_SOURCE,
    method_order: tuple[str, ...] = DEFAULT_METHOD_ORDER,
) -> pd.DataFrame:
    """Pick one forecast source per city using lowest validation MAE.

    ``validation_scores`` is the long table written by train/eval when a
    validation split is enabled. It has one row per city/source/bias-method.
    Selection is based only on validation MAE; test metrics are joined later.
    """
    required = {"city", "source", "method", "validation_mae"}
    missing = required - set(validation_scores.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if validation_scores.empty:
        return pd.DataFrame(columns=SOURCE_SELECTION_COLUMNS)

    method_rank = {method: index for index, method in enumerate(method_order)}
    selections = []
    for city, group in validation_scores.groupby("city", sort=True):
        valid = group[group["validation_mae"].notna()].copy()
        if valid.empty:
            fallback = _fallback_source_row(group, fallback_source=fallback_source)
            selections.append(
                {
                    "city": city,
                    "selected_source": fallback["source"],
                    "source_selection_bias_method": fallback["method"],
                    "source_validation_mae": pd.NA,
                    "source_selection_fallback": True,
                }
            )
            continue

        valid["_method_rank"] = valid["method"].map(method_rank).fillna(
            len(method_rank)
        )
        winner = valid.sort_values(
            ["validation_mae", "_method_rank", "source", "method"]
        ).iloc[0]
        selections.append(
            {
                "city": city,
                "selected_source": winner["source"],
                "source_selection_bias_method": winner["method"],
                "source_validation_mae": winner["validation_mae"],
                "source_selection_fallback": False,
            }
        )
    return pd.DataFrame(selections, columns=SOURCE_SELECTION_COLUMNS)


def evaluate_selected_sources(
    evaluation: pd.DataFrame, selected_sources: pd.DataFrame
) -> pd.DataFrame:
    """Join selected city/source pairs to their held-out test metrics."""
    required_evaluation = {"city", "source"}
    missing_evaluation = required_evaluation - set(evaluation.columns)
    if missing_evaluation:
        raise ValueError(f"missing evaluation columns: {sorted(missing_evaluation)}")
    missing_selection = set(SOURCE_SELECTION_COLUMNS) - set(selected_sources.columns)
    if missing_selection:
        raise ValueError(f"missing selection columns: {sorted(missing_selection)}")
    if selected_sources.empty:
        return pd.DataFrame()

    merged = selected_sources.merge(
        evaluation,
        left_on=["city", "selected_source"],
        right_on=["city", "source"],
        how="left",
        validate="one_to_one",
    )
    return merged.drop(columns=["source"])


def summarize_selected_sources(selected_evaluation: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-source held-out metrics across cities."""
    if selected_evaluation.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    row = {"n_cities": int(selected_evaluation["city"].nunique())}
    for column in SUMMARY_COLUMNS[1:]:
        row[column] = (
            float(selected_evaluation[column].mean())
            if column in selected_evaluation.columns
            else pd.NA
        )
    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)


def compare_source_policies(
    *,
    validation_scores: pd.DataFrame,
    evaluation: pd.DataFrame,
    selected_sources: pd.DataFrame,
    selected_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare per-city selected sources against the best global source.

    The global policy is selected from validation only: for each source/city,
    use the best validation MAE across bias methods, then average across cities.
    Held-out test metrics are joined from ``evaluation`` afterward.
    """
    if selected_summary.empty:
        return pd.DataFrame(columns=POLICY_COMPARISON_COLUMNS)

    rows = [
        {
            "policy": "per_city_validation",
            "selected_source": "per_city",
            "validation_mae": _mean_or_na(selected_sources["source_validation_mae"]),
            **selected_summary.iloc[0].to_dict(),
        }
    ]

    global_row = _best_global_source_policy(validation_scores, evaluation)
    if global_row is not None:
        rows.append(global_row)
    return pd.DataFrame(rows, columns=POLICY_COMPARISON_COLUMNS)


def recommend_sources(
    selected_sources: pd.DataFrame, policy_comparison: pd.DataFrame
) -> pd.DataFrame:
    """Create the production source map from the comparison artifact.

    ``selected_sources.csv`` remains the diagnostic per-city selector. This
    artifact maps every city to the validation-selected global source so
    ``--model-run-dir`` can default to the simpler regularized policy.
    """
    if selected_sources.empty or policy_comparison.empty:
        return pd.DataFrame(columns=RECOMMENDED_SOURCE_COLUMNS)

    policy_rows = policy_comparison[
        policy_comparison["policy"] == "best_global_validation_source"
    ]
    if policy_rows.empty:
        return pd.DataFrame(columns=RECOMMENDED_SOURCE_COLUMNS)

    source = str(policy_rows.iloc[0]["selected_source"]).strip()
    if not source:
        return pd.DataFrame(columns=RECOMMENDED_SOURCE_COLUMNS)

    rows = [
        {
            "city": city,
            "selected_source": source,
            "recommended_policy": "best_global_validation_source",
        }
        for city in sorted(selected_sources["city"].astype(str).unique())
    ]
    return pd.DataFrame(rows, columns=RECOMMENDED_SOURCE_COLUMNS)


def write_source_selection_outputs(
    *, validation_scores_path: Path, evaluation_path: Path, output_dir: Path
) -> SourceSelectionResult:
    """Write selected source, selected evaluation, and summary CSV artifacts."""
    validation_scores = pd.read_csv(validation_scores_path)
    evaluation = pd.read_csv(evaluation_path)
    selected_sources = select_sources_by_validation(validation_scores)
    selected_evaluation = evaluate_selected_sources(evaluation, selected_sources)
    summary = summarize_selected_sources(selected_evaluation)
    policy_comparison = compare_source_policies(
        validation_scores=validation_scores,
        evaluation=evaluation,
        selected_sources=selected_sources,
        selected_summary=summary,
    )
    recommended_sources = recommend_sources(selected_sources, policy_comparison)

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_sources.to_csv(output_dir / "selected_sources.csv", index=False)
    recommended_sources.to_csv(output_dir / "recommended_sources.csv", index=False)
    selected_evaluation.to_csv(
        output_dir / "selected_source_evaluation.csv", index=False
    )
    summary.to_csv(output_dir / "selected_source_summary.csv", index=False)
    policy_comparison.to_csv(output_dir / "source_policy_comparison.csv", index=False)
    return SourceSelectionResult(
        selected_sources=selected_sources,
        recommended_sources=recommended_sources,
        selected_evaluation=selected_evaluation,
        summary=summary,
        policy_comparison=policy_comparison,
    )


def _fallback_source_row(group: pd.DataFrame, *, fallback_source: str) -> pd.Series:
    fallback_rows = group[group["source"] == fallback_source]
    if not fallback_rows.empty:
        return fallback_rows.sort_values(["source", "method"]).iloc[0]
    return group.sort_values(["source", "method"]).iloc[0]


def _best_global_source_policy(
    validation_scores: pd.DataFrame, evaluation: pd.DataFrame
) -> dict | None:
    required_validation = {"city", "source", "validation_mae"}
    required_evaluation = {"city", "source"}
    if required_validation - set(validation_scores.columns):
        return None
    if required_evaluation - set(evaluation.columns):
        return None

    valid = validation_scores[validation_scores["validation_mae"].notna()].copy()
    if valid.empty:
        return None

    city_source = (
        valid.groupby(["city", "source"], sort=True)["validation_mae"]
        .min()
        .reset_index()
    )
    source_scores = (
        city_source.groupby("source", sort=True)["validation_mae"]
        .mean()
        .reset_index()
        .sort_values(["validation_mae", "source"])
    )
    winner = source_scores.iloc[0]
    source = str(winner["source"])
    source_evaluation = evaluation[evaluation["source"].astype(str) == source]
    summary = _summarize_evaluation(source_evaluation)
    return {
        "policy": "best_global_validation_source",
        "selected_source": source,
        "validation_mae": float(winner["validation_mae"]),
        **summary,
    }


def _summarize_evaluation(evaluation: pd.DataFrame) -> dict:
    row = {"n_cities": int(evaluation["city"].nunique()) if "city" in evaluation else 0}
    for column in SUMMARY_COLUMNS[1:]:
        row[column] = (
            float(evaluation[column].mean())
            if column in evaluation.columns and not evaluation.empty
            else pd.NA
        )
    return row


def _mean_or_na(values: pd.Series) -> object:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return pd.NA
    return float(numeric.mean())

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
SUMMARY_COLUMNS = [
    "n_cities",
    "mae_raw",
    "mae_corrected",
    "interval_coverage_raw",
    "interval_width_raw",
]


@dataclass(frozen=True)
class SourceSelectionResult:
    """Artifacts from validation-based source selection."""

    selected_sources: pd.DataFrame
    selected_evaluation: pd.DataFrame
    summary: pd.DataFrame


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


def write_source_selection_outputs(
    *, validation_scores_path: Path, evaluation_path: Path, output_dir: Path
) -> SourceSelectionResult:
    """Write selected source, selected evaluation, and summary CSV artifacts."""
    validation_scores = pd.read_csv(validation_scores_path)
    evaluation = pd.read_csv(evaluation_path)
    selected_sources = select_sources_by_validation(validation_scores)
    selected_evaluation = evaluate_selected_sources(evaluation, selected_sources)
    summary = summarize_selected_sources(selected_evaluation)

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_sources.to_csv(output_dir / "selected_sources.csv", index=False)
    selected_evaluation.to_csv(
        output_dir / "selected_source_evaluation.csv", index=False
    )
    summary.to_csv(output_dir / "selected_source_summary.csv", index=False)
    return SourceSelectionResult(
        selected_sources=selected_sources,
        selected_evaluation=selected_evaluation,
        summary=summary,
    )


def _fallback_source_row(group: pd.DataFrame, *, fallback_source: str) -> pd.Series:
    fallback_rows = group[group["source"] == fallback_source]
    if not fallback_rows.empty:
        return fallback_rows.sort_values(["source", "method"]).iloc[0]
    return group.sort_values(["source", "method"]).iloc[0]

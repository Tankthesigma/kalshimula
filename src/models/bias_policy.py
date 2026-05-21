"""Bias-policy comparison helpers for a locked source policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.train_eval import train_eval_split
from src.models.validation_grid import ValidationGridResult, evaluate_recency_alpha_grid

POLICY_COMPARISON_COLUMNS = [
    "policy",
    "bias_strategy",
    "bias_recent_days",
    "alpha",
    "validation_mae_corrected",
    "validation_interval_coverage_raw",
    "validation_interval_width_raw",
    "test_mae_corrected",
    "test_interval_coverage_raw",
    "test_interval_width_raw",
    "n_groups",
    "n_rows",
    "recommended",
    "notes",
]
RECOMMENDED_POLICY_COLUMNS = [
    "policy",
    "bias_strategy",
    "bias_recent_days",
    "alpha",
    "selected_by",
]


@dataclass(frozen=True)
class BiasPolicyResult:
    """Artifacts from comparing bias policies for a recommended source map."""

    comparison: pd.DataFrame
    recommended_policy: pd.DataFrame
    recommended_bias_table: pd.DataFrame
    recommended_interval_table: pd.DataFrame


def filter_rows_to_recommended_sources(
    rows: pd.DataFrame, recommended_sources: pd.DataFrame
) -> pd.DataFrame:
    """Keep rows whose source matches the recommended source for the row city."""
    required_rows = {"city", "source"}
    missing_rows = required_rows - set(rows.columns)
    if missing_rows:
        raise ValueError(f"missing row columns: {sorted(missing_rows)}")
    required_sources = {"city", "selected_source"}
    missing_sources = required_sources - set(recommended_sources.columns)
    if missing_sources:
        raise ValueError(f"missing recommended source columns: {sorted(missing_sources)}")
    if rows.empty or recommended_sources.empty:
        return rows.iloc[0:0].copy()

    mapping = recommended_sources[["city", "selected_source"]].drop_duplicates()
    merged = rows.merge(mapping, on="city", how="inner")
    filtered = merged[merged["source"].astype(str) == merged["selected_source"].astype(str)]
    return filtered.drop(columns=["selected_source"]).copy()


def compare_bias_policies(
    *,
    rows: pd.DataFrame,
    recommended_sources: pd.DataFrame,
    evaluation: pd.DataFrame,
    selected_methods: pd.DataFrame,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    recent_days: tuple[int, ...] = (90, 180, 365),
    alphas: tuple[float, ...] = (0.2, 0.13),
    target_coverage: float = 0.8,
) -> BiasPolicyResult:
    """Compare current per-city bias selection against global recent windows."""
    source_rows = filter_rows_to_recommended_sources(rows, recommended_sources)
    if source_rows.empty:
        raise ValueError("no rows matched recommended sources")

    grid = evaluate_recency_alpha_grid(
        source_rows,
        validation_start=validation_start,
        test_start=test_start,
        recent_days=recent_days,
        alphas=alphas,
        target_coverage=target_coverage,
    )
    comparison = _build_comparison(
        recommended_sources=recommended_sources,
        evaluation=evaluation,
        selected_methods=selected_methods,
        grid=grid,
    )
    recommended_policy = _recommended_policy_from_grid(grid)
    selected = recommended_policy.iloc[0]
    recommended_result = train_eval_split(
        source_rows,
        test_start=test_start,
        alpha=float(selected["alpha"]),
        bias_strategy="recent",
        bias_recent_days=int(selected["bias_recent_days"]),
    )
    return BiasPolicyResult(
        comparison=comparison,
        recommended_policy=recommended_policy,
        recommended_bias_table=recommended_result.bias_table,
        recommended_interval_table=recommended_result.interval_table,
    )


def write_bias_policy_outputs(
    *,
    input_path: Path,
    train_eval_dir: Path,
    recommended_sources_path: Path,
    output_dir: Path,
    validation_start: str,
    test_start: str,
    recent_days: tuple[int, ...] = (90, 180, 365),
    alphas: tuple[float, ...] = (0.2, 0.13),
    target_coverage: float = 0.8,
) -> BiasPolicyResult:
    """Run bias-policy comparison from CSV artifacts and write outputs."""
    rows = pd.read_csv(input_path, parse_dates=["target_date"])
    recommended_sources = pd.read_csv(recommended_sources_path)
    evaluation = pd.read_csv(train_eval_dir / "evaluation.csv")
    selected_methods = pd.read_csv(train_eval_dir / "selected_methods.csv")
    result = compare_bias_policies(
        rows=rows,
        recommended_sources=recommended_sources,
        evaluation=evaluation,
        selected_methods=selected_methods,
        validation_start=validation_start,
        test_start=test_start,
        recent_days=recent_days,
        alphas=alphas,
        target_coverage=target_coverage,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(output_dir / "bias_policy_comparison.csv", index=False)
    result.recommended_policy.to_csv(output_dir / "model_policy.csv", index=False)
    result.recommended_bias_table.to_csv(output_dir / "bias_table.csv", index=False)
    result.recommended_interval_table.to_csv(output_dir / "interval_table.csv", index=False)
    return result


def _build_comparison(
    *,
    recommended_sources: pd.DataFrame,
    evaluation: pd.DataFrame,
    selected_methods: pd.DataFrame,
    grid: ValidationGridResult,
) -> pd.DataFrame:
    rows = [_per_city_policy_row(recommended_sources, evaluation, selected_methods)]
    selected = grid.selected_config.iloc[0]
    for test_row in grid.test_grid.itertuples(index=False):
        validation_row = grid.validation_grid[
            (grid.validation_grid["bias_recent_days"] == test_row.bias_recent_days)
            & (grid.validation_grid["alpha"] == test_row.alpha)
        ].iloc[0]
        recommended = (
            int(test_row.bias_recent_days) == int(selected["bias_recent_days"])
            and float(test_row.alpha) == float(selected["alpha"])
        )
        rows.append(
            {
                "policy": f"global_recent_{int(test_row.bias_recent_days)}d",
                "bias_strategy": "recent",
                "bias_recent_days": int(test_row.bias_recent_days),
                "alpha": float(test_row.alpha),
                "validation_mae_corrected": validation_row["mae_corrected"],
                "validation_interval_coverage_raw": validation_row["interval_coverage_raw"],
                "validation_interval_width_raw": validation_row["interval_width_raw"],
                "test_mae_corrected": test_row.mae_corrected,
                "test_interval_coverage_raw": test_row.interval_coverage_raw,
                "test_interval_width_raw": test_row.interval_width_raw,
                "n_groups": int(test_row.n_groups),
                "n_rows": int(test_row.n_rows),
                "recommended": recommended,
                "notes": "single global recent-window bias policy",
            }
        )
    return pd.DataFrame(rows, columns=POLICY_COMPARISON_COLUMNS)


def _per_city_policy_row(
    recommended_sources: pd.DataFrame,
    evaluation: pd.DataFrame,
    selected_methods: pd.DataFrame,
) -> dict:
    selected_eval = _filter_artifact_to_recommended_sources(evaluation, recommended_sources)
    selected_method_rows = _filter_artifact_to_recommended_sources(
        selected_methods, recommended_sources
    )
    return {
        "policy": "per_city_bias_selection",
        "bias_strategy": "per_city",
        "bias_recent_days": pd.NA,
        "alpha": pd.NA,
        "validation_mae_corrected": _mean_or_na(
            selected_method_rows["selected_validation_mae"]
        )
        if "selected_validation_mae" in selected_method_rows
        else pd.NA,
        "validation_interval_coverage_raw": pd.NA,
        "validation_interval_width_raw": pd.NA,
        "test_mae_corrected": _weighted_mean(selected_eval, "mae_corrected"),
        "test_interval_coverage_raw": _weighted_mean(selected_eval, "interval_coverage_raw"),
        "test_interval_width_raw": _weighted_mean(selected_eval, "interval_width_raw"),
        "n_groups": int(len(selected_eval)),
        "n_rows": int(selected_eval["n"].sum()) if "n" in selected_eval else 0,
        "recommended": False,
        "notes": "current per-city validation-selected bias methods",
    }


def _recommended_policy_from_grid(grid: ValidationGridResult) -> pd.DataFrame:
    selected = grid.selected_config.iloc[0]
    return pd.DataFrame(
        [
            {
                "policy": f"global_recent_{int(selected['bias_recent_days'])}d",
                "bias_strategy": selected["bias_strategy"],
                "bias_recent_days": int(selected["bias_recent_days"]),
                "alpha": float(selected["alpha"]),
                "selected_by": selected["selected_by"],
            }
        ],
        columns=RECOMMENDED_POLICY_COLUMNS,
    )


def _filter_artifact_to_recommended_sources(
    artifact: pd.DataFrame, recommended_sources: pd.DataFrame
) -> pd.DataFrame:
    mapping = recommended_sources[["city", "selected_source"]].drop_duplicates()
    merged = artifact.merge(mapping, on="city", how="inner")
    return merged[
        merged["source"].astype(str) == merged["selected_source"].astype(str)
    ].drop(columns=["selected_source"])


def _weighted_mean(frame: pd.DataFrame, column: str) -> object:
    if column not in frame.columns or frame.empty:
        return pd.NA
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame["n"], errors="coerce") if "n" in frame else None
    valid = values.notna()
    if weights is not None:
        valid = valid & weights.notna() & (weights > 0)
    if not valid.any():
        return pd.NA
    if weights is None:
        return float(values[valid].mean())
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def _mean_or_na(values: pd.Series) -> object:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return pd.NA
    return float(numeric.mean())

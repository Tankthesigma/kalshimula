"""Validation-grid helpers for bias recency and interval alpha choices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.bias import fit_bias_table
from src.models.intervals import fit_empirical_intervals
from src.models.train_eval import train_eval_split

GRID_COLUMNS = [
    "bias_strategy",
    "bias_recent_days",
    "alpha",
    "split",
    "n_groups",
    "n_rows",
    "mae_raw",
    "mae_corrected",
    "rmse_corrected",
    "bias_corrected",
    "interval_coverage_raw",
    "interval_width_raw",
    "target_coverage",
    "coverage_shortfall",
    "meets_coverage_target",
]
SELECTED_CONFIG_COLUMNS = [
    "bias_strategy",
    "bias_recent_days",
    "alpha",
    "selected_by",
    "validation_mae_corrected",
    "validation_interval_coverage_raw",
    "validation_interval_width_raw",
    "target_coverage",
]
MODEL_POLICY_COLUMNS = [
    "source",
    "bias_strategy",
    "bias_recent_days",
    "alpha",
    "fit_start",
    "fit_end",
    "n_train_rows",
]


@dataclass(frozen=True)
class ValidationGridResult:
    """Artifacts from evaluating bias/interval configs on validation and test."""

    validation_grid: pd.DataFrame
    test_grid: pd.DataFrame
    selected_config: pd.DataFrame


def evaluate_recency_alpha_grid(
    rows: pd.DataFrame,
    *,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    recent_days: tuple[int, ...] = (90, 180, 365),
    alphas: tuple[float, ...] = (0.2, 0.13),
    target_coverage: float = 0.8,
) -> ValidationGridResult:
    """Compare recent-window bias and empirical interval alpha settings.

    Validation rows are the date range ``validation_start <= target_date <
    test_start``. Test rows are ``target_date >= test_start``. Each grid config
    is selected from validation only; test metrics are reported afterward for
    leakage-safe comparison.
    """
    if not recent_days:
        raise ValueError("recent_days must contain at least one value")
    if not alphas:
        raise ValueError("alphas must contain at least one value")
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be between 0 and 1")
    if any(days < 1 for days in recent_days):
        raise ValueError("recent_days values must be at least 1")
    if any(not 0 < alpha < 1 for alpha in alphas):
        raise ValueError("alpha values must be between 0 and 1")

    validation_records = []
    test_records = []
    for days in recent_days:
        for alpha in alphas:
            validation_result = train_eval_split(
                rows,
                test_start=validation_start,
                alpha=alpha,
                bias_strategy="recent",
                bias_recent_days=days,
            )
            validation_window = validation_result.corrected_test_rows[
                pd.to_datetime(validation_result.corrected_test_rows["target_date"])
                < pd.Timestamp(test_start)
            ]
            if validation_window.empty:
                raise ValueError("validation slice is empty; check validation_start/test_start")
            validation_eval = _evaluation_for_rows(validation_result.evaluation, validation_window)
            validation_records.append(
                _summarize_evaluation(
                    validation_eval,
                    split="validation",
                    bias_recent_days=days,
                    alpha=alpha,
                    target_coverage=target_coverage,
                )
            )

            test_result = train_eval_split(
                rows,
                test_start=test_start,
                alpha=alpha,
                bias_strategy="recent",
                bias_recent_days=days,
            )
            test_records.append(
                _summarize_evaluation(
                    test_result.evaluation,
                    split="test",
                    bias_recent_days=days,
                    alpha=alpha,
                    target_coverage=target_coverage,
                )
            )

    validation_grid = pd.DataFrame(validation_records, columns=GRID_COLUMNS)
    test_grid = pd.DataFrame(test_records, columns=GRID_COLUMNS)
    selected_config = _select_config(validation_grid, target_coverage=target_coverage)
    return ValidationGridResult(
        validation_grid=validation_grid,
        test_grid=test_grid,
        selected_config=selected_config,
    )


def write_recency_alpha_grid_outputs(
    *,
    input_path: Path,
    output_dir: Path,
    validation_start: str,
    test_start: str,
    recent_days: tuple[int, ...] = (90, 180, 365),
    alphas: tuple[float, ...] = (0.2, 0.13),
    target_coverage: float = 0.8,
    source: str | None = None,
    policy_out_dir: Path | None = None,
) -> ValidationGridResult:
    """Run the recency/alpha grid from CSV and write output artifacts."""
    rows = pd.read_csv(input_path, parse_dates=["target_date"])
    if source is not None:
        rows = rows[rows["source"].astype(str) == source].copy()
        if rows.empty:
            raise ValueError(f"no rows found for source: {source}")

    result = evaluate_recency_alpha_grid(
        rows,
        validation_start=validation_start,
        test_start=test_start,
        recent_days=recent_days,
        alphas=alphas,
        target_coverage=target_coverage,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.validation_grid.to_csv(output_dir / "validation_grid.csv", index=False)
    result.test_grid.to_csv(output_dir / "test_grid.csv", index=False)
    result.selected_config.to_csv(output_dir / "selected_config.csv", index=False)
    if policy_out_dir is not None:
        write_selected_policy_outputs(
            rows=rows,
            selected_config=result.selected_config,
            output_dir=policy_out_dir,
            test_start=test_start,
            source=source,
        )
    return result


def write_selected_policy_outputs(
    *,
    rows: pd.DataFrame,
    selected_config: pd.DataFrame,
    output_dir: Path,
    test_start: str | pd.Timestamp,
    source: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit live prediction artifacts from the validation-selected config."""
    if selected_config.empty:
        raise ValueError("selected_config is empty")
    config = selected_config.iloc[0]
    strategy = str(config["bias_strategy"])
    if strategy != "recent":
        raise ValueError(f"unsupported model policy bias_strategy: {strategy}")

    days = int(config["bias_recent_days"])
    alpha = float(config["alpha"])
    train = _pre_test_rows(rows, test_start=test_start)
    if train.empty:
        raise ValueError("model policy train split is empty")

    recent = _recent_rows(train, days=days)
    bias_table = fit_bias_table(recent)
    interval_table = fit_empirical_intervals(train, alpha=alpha)
    policy = pd.DataFrame(
        [
            {
                "source": source or "",
                "bias_strategy": strategy,
                "bias_recent_days": days,
                "alpha": alpha,
                "fit_start": train["target_date"].min(),
                "fit_end": train["target_date"].max(),
                "n_train_rows": len(train),
            }
        ],
        columns=MODEL_POLICY_COLUMNS,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    policy.to_csv(output_dir / "model_policy.csv", index=False)
    bias_table.to_csv(output_dir / "bias_table.csv", index=False)
    interval_table.to_csv(output_dir / "interval_table.csv", index=False)
    return policy, bias_table, interval_table


def _pre_test_rows(rows: pd.DataFrame, *, test_start: str | pd.Timestamp) -> pd.DataFrame:
    if "target_date" not in rows.columns:
        raise ValueError("rows must include target_date")
    df = rows.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df[df["target_date"] < pd.Timestamp(test_start)].copy()


def _recent_rows(train: pd.DataFrame, *, days: int) -> pd.DataFrame:
    if days < 1:
        raise ValueError("bias_recent_days must be at least 1")
    cutoff = pd.to_datetime(train["target_date"]).max() - pd.Timedelta(days=days - 1)
    recent = train[pd.to_datetime(train["target_date"]) >= cutoff].copy()
    if recent.empty:
        raise ValueError("recent model policy train split is empty")
    return recent


def _evaluation_for_rows(evaluation: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Recompute grouped evaluation for an already-corrected row subset."""
    from src.models.baseline_training import evaluate_corrected_predictions

    if rows.empty:
        return evaluation.iloc[0:0].copy()
    return evaluate_corrected_predictions(rows)


def _summarize_evaluation(
    evaluation: pd.DataFrame,
    *,
    split: str,
    bias_recent_days: int,
    alpha: float,
    target_coverage: float,
) -> dict:
    row = {
        "bias_strategy": "recent",
        "bias_recent_days": bias_recent_days,
        "alpha": alpha,
        "split": split,
        "n_groups": int(len(evaluation)),
        "n_rows": int(evaluation["n"].sum()) if "n" in evaluation else 0,
    }
    for column in GRID_COLUMNS[6:12]:
        row[column] = _weighted_mean(evaluation, column)

    coverage = row["interval_coverage_raw"]
    if pd.isna(coverage):
        shortfall = pd.NA
        meets = False
    else:
        shortfall = max(0.0, target_coverage - float(coverage))
        meets = shortfall == 0.0
    row["target_coverage"] = target_coverage
    row["coverage_shortfall"] = shortfall
    row["meets_coverage_target"] = meets
    return row


def _weighted_mean(evaluation: pd.DataFrame, column: str) -> object:
    if column not in evaluation.columns or evaluation.empty:
        return pd.NA
    values = pd.to_numeric(evaluation[column], errors="coerce")
    weights = pd.to_numeric(evaluation["n"], errors="coerce") if "n" in evaluation else None
    valid = values.notna()
    if weights is not None:
        valid = valid & weights.notna() & (weights > 0)
    if not valid.any():
        return pd.NA
    if weights is None:
        return float(values[valid].mean())
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def _select_config(
    validation_grid: pd.DataFrame, *, target_coverage: float
) -> pd.DataFrame:
    if validation_grid.empty:
        return pd.DataFrame(columns=SELECTED_CONFIG_COLUMNS)

    ranked = validation_grid.copy()
    ranked["_coverage_miss_rank"] = (~ranked["meets_coverage_target"]).astype(int)
    ranked = ranked.sort_values(
        [
            "_coverage_miss_rank",
            "mae_corrected",
            "coverage_shortfall",
            "interval_width_raw",
            "bias_recent_days",
            "alpha",
        ],
        ascending=[True, True, True, True, True, True],
    )
    winner = ranked.iloc[0]
    return pd.DataFrame(
        [
            {
                "bias_strategy": winner["bias_strategy"],
                "bias_recent_days": winner["bias_recent_days"],
                "alpha": winner["alpha"],
                "selected_by": "validation_meets_coverage_then_mae",
                "validation_mae_corrected": winner["mae_corrected"],
                "validation_interval_coverage_raw": winner["interval_coverage_raw"],
                "validation_interval_width_raw": winner["interval_width_raw"],
                "target_coverage": target_coverage,
            }
        ],
        columns=SELECTED_CONFIG_COLUMNS,
    )

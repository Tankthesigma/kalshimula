"""Leakage-safe train/test evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import pandas as pd

from src.models.baseline_training import evaluate_corrected_predictions
from src.models.baselines import mean_absolute_error
from src.models.bias import apply_bias_correction, fit_bias_table
from src.models.diagnostics import build_residual_diagnostics
from src.models.intervals import apply_empirical_intervals, fit_empirical_intervals

PER_CITY_BIAS_METHODS = (
    "recent_180d",
    "prior_same_month",
    "recent_global",
    "all_global",
)
DEFAULT_SELECTION_FALLBACK = "seasonal"
VALIDATION_SCORE_COLUMNS = [
    "city",
    "source",
    "method",
    "validation_n",
    "validation_mae",
]
SELECTED_METHOD_COLUMNS = [
    "city",
    "source",
    "selected_bias_method",
    "selected_validation_mae",
    "selection_fallback",
]


@dataclass(frozen=True)
class TrainEvalResult:
    """Artifacts from fitting on train rows and evaluating on test rows."""

    train_rows: pd.DataFrame
    validation_rows: pd.DataFrame
    test_rows: pd.DataFrame
    bias_table: pd.DataFrame
    interval_table: pd.DataFrame
    validation_scores: pd.DataFrame
    selected_methods: pd.DataFrame
    corrected_test_rows: pd.DataFrame
    evaluation: pd.DataFrame
    source_residuals: pd.DataFrame
    monthly_residuals: pd.DataFrame


def split_rows_by_date(
    rows: pd.DataFrame, *, test_start: str | pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train/test by target_date, with test_start inclusive."""
    if "target_date" not in rows.columns:
        raise ValueError("rows must include target_date")
    df = rows.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    cutoff = pd.Timestamp(test_start)
    train = df[df["target_date"] < cutoff].copy()
    test = df[df["target_date"] >= cutoff].copy()
    return train, test


def split_rows_by_month_stratified(
    rows: pd.DataFrame, *, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each city/source/month into train/test rows by date order.

    This is a diagnostic split for measuring month-aware calibration on short
    windows. It is not a substitute for the default chronological split.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    required = {"city", "source", "target_date"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = rows.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["_month"] = df["target_date"].dt.month.astype("Int64")
    df["_row_order"] = range(len(df))

    train_parts = []
    test_parts = []
    for _, group in df.sort_values(["target_date", "_row_order"]).groupby(
        ["city", "source", "_month"], sort=True
    ):
        if len(group) < 2:
            train_parts.append(group)
            continue
        test_count = min(max(1, ceil(len(group) * test_fraction)), len(group) - 1)
        train_parts.append(group.iloc[:-test_count])
        test_parts.append(group.iloc[-test_count:])

    train = pd.concat(train_parts, ignore_index=False) if train_parts else df.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=False) if test_parts else df.iloc[0:0]
    drop_cols = ["_month", "_row_order"]
    return (
        train.drop(columns=drop_cols).sort_values("target_date").copy(),
        test.drop(columns=drop_cols).sort_values("target_date").copy(),
    )


def train_eval_split(
    rows: pd.DataFrame,
    *,
    test_start: str | pd.Timestamp | None = None,
    alpha: float = 0.2,
    split_strategy: str = "date",
    test_fraction: float = 0.2,
    bias_strategy: str = "seasonal",
    bias_recent_days: int | None = None,
    validation_start: str | pd.Timestamp | None = None,
    candidate_methods: tuple[str, ...] = PER_CITY_BIAS_METHODS,
    fallback_method: str = DEFAULT_SELECTION_FALLBACK,
) -> TrainEvalResult:
    """Fit bias/intervals on train rows and evaluate corrected forecasts on test rows."""
    normalized_strategy = split_strategy.replace("-", "_")
    if normalized_strategy == "date":
        if test_start is None:
            raise ValueError("test_start is required for date split")
        train, test = split_rows_by_date(rows, test_start=test_start)
    elif normalized_strategy == "month_stratified":
        train, test = split_rows_by_month_stratified(rows, test_fraction=test_fraction)
    else:
        raise ValueError(f"unknown split_strategy: {split_strategy}")
    if train.empty:
        raise ValueError("train split is empty")
    if test.empty:
        raise ValueError("test split is empty")

    interval_table = fit_empirical_intervals(train, alpha=alpha)

    if validation_start is None:
        validation = pd.DataFrame(columns=train.columns)
        bias_table = _fit_bias_for_strategy(
            train,
            bias_strategy=bias_strategy,
            bias_recent_days=bias_recent_days,
        )
        corrected = apply_bias_correction(test, bias_table)
        selected_methods = _default_selected_methods(test, bias_strategy, bias_recent_days)
        validation_scores = pd.DataFrame(columns=VALIDATION_SCORE_COLUMNS)
    else:
        train, validation = split_rows_by_date(train, test_start=validation_start)
        if train.empty:
            raise ValueError("train-validation fit split is empty")
        validation_scores = _score_candidate_methods(
            train_rows=train,
            validation_rows=validation,
            candidate_methods=candidate_methods,
        )
        selected_methods = _select_methods(
            rows=pd.concat([train, validation, test], ignore_index=True),
            validation_scores=validation_scores,
            candidate_methods=candidate_methods,
            fallback_method=fallback_method,
        )
        full_train = pd.concat([train, validation], ignore_index=True)
        bias_tables = _fit_method_tables(full_train, (*candidate_methods, fallback_method))
        corrected = _apply_selected_methods(test, selected_methods, bias_tables)
        bias_table = _selected_bias_table(bias_tables, selected_methods)

    corrected = apply_empirical_intervals(corrected, interval_table)
    evaluation = _attach_selection_metadata(
        evaluate_corrected_predictions(corrected), selected_methods
    )
    residuals = build_residual_diagnostics(corrected)
    return TrainEvalResult(
        train_rows=train,
        validation_rows=validation,
        test_rows=test,
        bias_table=bias_table,
        interval_table=interval_table,
        validation_scores=validation_scores,
        selected_methods=selected_methods,
        corrected_test_rows=corrected,
        evaluation=evaluation,
        source_residuals=residuals.source_summary,
        monthly_residuals=residuals.monthly_summary,
    )


def write_train_eval_outputs(
    *,
    input_path: Path,
    output_dir: Path,
    test_start: str | None = None,
    alpha: float = 0.2,
    split_strategy: str = "date",
    test_fraction: float = 0.2,
    bias_strategy: str = "seasonal",
    bias_recent_days: int | None = None,
    validation_start: str | None = None,
) -> TrainEvalResult:
    """Run leakage-safe train/test evaluation and write CSV artifacts."""
    rows = pd.read_csv(input_path, parse_dates=["target_date"])
    result = train_eval_split(
        rows,
        test_start=test_start,
        alpha=alpha,
        split_strategy=split_strategy,
        test_fraction=test_fraction,
        bias_strategy=bias_strategy,
        bias_recent_days=bias_recent_days,
        validation_start=validation_start,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.train_rows.to_csv(output_dir / "train_rows.csv", index=False)
    result.validation_rows.to_csv(output_dir / "validation_rows.csv", index=False)
    result.test_rows.to_csv(output_dir / "test_rows.csv", index=False)
    result.bias_table.to_csv(output_dir / "bias_table.csv", index=False)
    result.interval_table.to_csv(output_dir / "interval_table.csv", index=False)
    result.validation_scores.to_csv(output_dir / "validation_scores.csv", index=False)
    result.selected_methods.to_csv(output_dir / "selected_methods.csv", index=False)
    result.corrected_test_rows.to_csv(output_dir / "corrected_test_rows.csv", index=False)
    result.evaluation.to_csv(output_dir / "evaluation.csv", index=False)
    result.source_residuals.to_csv(output_dir / "source_residuals.csv", index=False)
    result.monthly_residuals.to_csv(output_dir / "monthly_residuals.csv", index=False)
    return result


def _fit_bias_for_strategy(
    train: pd.DataFrame,
    *,
    bias_strategy: str,
    bias_recent_days: int | None,
) -> pd.DataFrame:
    normalized_strategy = bias_strategy.replace("-", "_")
    if normalized_strategy == "seasonal":
        return fit_bias_table(train, group_month=True)
    if normalized_strategy == "global":
        return fit_bias_table(train)
    if normalized_strategy == "recent":
        recent = _recent_train_rows(train, bias_recent_days=bias_recent_days)
        return fit_bias_table(recent)
    if normalized_strategy in {"recent_180d", "recent_180"}:
        recent = _recent_train_rows(train, bias_recent_days=180)
        return fit_bias_table(recent)
    if normalized_strategy in {"recent_global", "recent_365d", "recent_365"}:
        recent = _recent_train_rows(train, bias_recent_days=365)
        return fit_bias_table(recent)
    if normalized_strategy in {"prior_same_month", "monthly"}:
        return fit_bias_table(train, group_month=True)
    if normalized_strategy == "all_global":
        return fit_bias_table(train)
    raise ValueError(f"unknown bias_strategy: {bias_strategy}")


def _recent_train_rows(train: pd.DataFrame, *, bias_recent_days: int | None) -> pd.DataFrame:
    if bias_recent_days is None:
        raise ValueError("bias_recent_days is required when bias_strategy='recent'")
    if bias_recent_days < 1:
        raise ValueError("bias_recent_days must be at least 1")
    if "target_date" not in train.columns:
        raise ValueError("train rows must include target_date for recent bias correction")

    df = train.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    cutoff = df["target_date"].max() - pd.Timedelta(days=bias_recent_days - 1)
    recent = df[df["target_date"] >= cutoff].copy()
    if recent.empty:
        raise ValueError("recent bias training split is empty")
    return recent


def _score_candidate_methods(
    *,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    candidate_methods: tuple[str, ...],
) -> pd.DataFrame:
    if validation_rows.empty:
        return pd.DataFrame(columns=VALIDATION_SCORE_COLUMNS)

    bias_tables = _fit_method_tables(train_rows, candidate_methods)
    records = []
    for method in candidate_methods:
        corrected = apply_bias_correction(validation_rows, bias_tables[method])
        for (city, source), group in corrected.groupby(["city", "source"], sort=True):
            actual = group["actual_high_f"].astype(float).tolist()
            predicted = group["corrected_point_f"].astype(float).tolist()
            records.append(
                {
                    "city": city,
                    "source": source,
                    "method": method,
                    "validation_n": len(group),
                    "validation_mae": mean_absolute_error(actual, predicted),
                }
            )
    return pd.DataFrame(records, columns=VALIDATION_SCORE_COLUMNS)


def _select_methods(
    *,
    rows: pd.DataFrame,
    validation_scores: pd.DataFrame,
    candidate_methods: tuple[str, ...],
    fallback_method: str,
) -> pd.DataFrame:
    method_rank = {method: index for index, method in enumerate(candidate_methods)}
    pairs = rows[["city", "source"]].drop_duplicates().sort_values(["city", "source"])
    selections = []
    for pair in pairs.itertuples(index=False):
        scores = validation_scores[
            (validation_scores["city"] == pair.city)
            & (validation_scores["source"] == pair.source)
        ].copy()
        if scores.empty or scores["validation_mae"].isna().all():
            selections.append(
                {
                    "city": pair.city,
                    "source": pair.source,
                    "selected_bias_method": fallback_method,
                    "selected_validation_mae": pd.NA,
                    "selection_fallback": True,
                }
            )
            continue

        scores["method_rank"] = scores["method"].map(method_rank)
        winner = scores.sort_values(["validation_mae", "method_rank"]).iloc[0]
        selections.append(
            {
                "city": pair.city,
                "source": pair.source,
                "selected_bias_method": winner["method"],
                "selected_validation_mae": winner["validation_mae"],
                "selection_fallback": False,
            }
        )
    return pd.DataFrame(selections, columns=SELECTED_METHOD_COLUMNS)


def _fit_method_tables(
    train: pd.DataFrame, methods: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    return {
        method: _fit_bias_for_strategy(
            train,
            bias_strategy=method,
            bias_recent_days=None,
        )
        for method in dict.fromkeys(methods)
    }


def _apply_selected_methods(
    test: pd.DataFrame,
    selected_methods: pd.DataFrame,
    bias_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = test.reset_index(drop=True).reset_index(names="_row_order")
    rows = rows.merge(
        selected_methods,
        on=["city", "source"],
        how="left",
    )
    rows["selected_bias_method"] = rows["selected_bias_method"].fillna(
        DEFAULT_SELECTION_FALLBACK
    )

    corrected_parts = []
    for method, group in rows.groupby("selected_bias_method", sort=False):
        corrected_parts.append(apply_bias_correction(group, bias_tables[str(method)]))
    if not corrected_parts:
        return rows.drop(columns=["_row_order"])
    corrected = pd.concat(corrected_parts, ignore_index=True)
    corrected = corrected.sort_values("_row_order").drop(columns=["_row_order"])
    return corrected.reset_index(drop=True)


def _default_selected_methods(
    test: pd.DataFrame, bias_strategy: str, bias_recent_days: int | None
) -> pd.DataFrame:
    method = bias_strategy.replace("-", "_")
    if method == "recent" and bias_recent_days is not None:
        method = f"recent_{bias_recent_days}d"
    pairs = test[["city", "source"]].drop_duplicates().sort_values(["city", "source"])
    selected = pairs.copy()
    selected["selected_bias_method"] = method
    selected["selected_validation_mae"] = pd.NA
    selected["selection_fallback"] = False
    return selected[SELECTED_METHOD_COLUMNS]


def _attach_selection_metadata(
    evaluation: pd.DataFrame, selected_methods: pd.DataFrame
) -> pd.DataFrame:
    return evaluation.merge(selected_methods, on=["city", "source"], how="left")


def _selected_bias_table(
    bias_tables: dict[str, pd.DataFrame], selected_methods: pd.DataFrame
) -> pd.DataFrame:
    frames = []
    for method in selected_methods["selected_bias_method"].dropna().unique():
        selected_pairs = selected_methods[
            selected_methods["selected_bias_method"] == method
        ][["city", "source"]]
        table = bias_tables[str(method)].merge(
            selected_pairs, on=["city", "source"], how="inner"
        )
        tagged = table.copy()
        tagged.insert(0, "method", method)
        frames.append(tagged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

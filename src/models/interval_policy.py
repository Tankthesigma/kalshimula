"""Interval policy calibration helpers for a locked source policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.bias_policy import filter_rows_to_recommended_sources
from src.models.intervals import apply_empirical_intervals, fit_empirical_intervals
from src.models.scoring import interval_coverage

INTERVAL_POLICY_COLUMNS = [
    "city",
    "source",
    "selected_alpha",
    "validation_coverage",
    "validation_width",
    "validation_n",
    "selection_reason",
]
INTERVAL_POLICY_COMPARISON_COLUMNS = [
    "policy",
    "alpha",
    "split",
    "n_groups",
    "n_rows",
    "interval_coverage_raw",
    "interval_width_raw",
    "target_coverage",
    "coverage_shortfall",
    "recommended",
]


@dataclass(frozen=True)
class IntervalPolicyResult:
    """Artifacts from selecting interval alpha per recommended city/source."""

    selected_policy: pd.DataFrame
    comparison: pd.DataFrame
    interval_table: pd.DataFrame


def compare_interval_policies(
    *,
    rows: pd.DataFrame,
    recommended_sources: pd.DataFrame,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    alphas: tuple[float, ...] = (0.2, 0.13, 0.1, 0.05),
    target_coverage: float = 0.8,
) -> IntervalPolicyResult:
    """Select per-city/source interval alpha from validation coverage.

    Validation rows are used only to select alpha. Final interval rows are fit
    on all rows before ``test_start`` and then evaluated on held-out test rows.
    """
    if not alphas:
        raise ValueError("alphas must contain at least one value")
    if any(not 0 < alpha < 1 for alpha in alphas):
        raise ValueError("alpha values must be between 0 and 1")
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be between 0 and 1")

    source_rows = filter_rows_to_recommended_sources(rows, recommended_sources)
    if source_rows.empty:
        raise ValueError("no rows matched recommended sources")
    train_fit, validation, full_train, test = _split_rows(
        source_rows, validation_start=validation_start, test_start=test_start
    )
    if train_fit.empty:
        raise ValueError("interval train split is empty")
    if validation.empty:
        raise ValueError("interval validation split is empty")
    if test.empty:
        raise ValueError("interval test split is empty")

    validation_candidates = _score_alpha_candidates(
        train_rows=train_fit,
        eval_rows=validation,
        alphas=alphas,
        split="validation",
        target_coverage=target_coverage,
    )
    selected_policy = _select_per_group_alpha(
        validation_candidates, target_coverage=target_coverage
    )
    interval_table = _fit_selected_interval_table(
        train_rows=full_train, selected_policy=selected_policy
    )
    comparison = _build_comparison(
        train_fit=train_fit,
        validation=validation,
        full_train=full_train,
        test=test,
        alphas=alphas,
        selected_policy=selected_policy,
        interval_table=interval_table,
        target_coverage=target_coverage,
    )
    return IntervalPolicyResult(
        selected_policy=selected_policy,
        comparison=comparison,
        interval_table=interval_table,
    )


def write_interval_policy_outputs(
    *,
    input_path: Path,
    recommended_sources_path: Path,
    output_dir: Path,
    validation_start: str,
    test_start: str,
    alphas: tuple[float, ...] = (0.2, 0.13, 0.1, 0.05),
    target_coverage: float = 0.8,
) -> IntervalPolicyResult:
    """Run interval-policy calibration from CSV artifacts and write outputs."""
    rows = pd.read_csv(input_path, parse_dates=["target_date"])
    recommended_sources = pd.read_csv(recommended_sources_path)
    result = compare_interval_policies(
        rows=rows,
        recommended_sources=recommended_sources,
        validation_start=validation_start,
        test_start=test_start,
        alphas=alphas,
        target_coverage=target_coverage,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.selected_policy.to_csv(output_dir / "interval_policy.csv", index=False)
    result.comparison.to_csv(output_dir / "interval_policy_comparison.csv", index=False)
    result.interval_table.to_csv(output_dir / "interval_table.csv", index=False)
    return result


def _split_rows(
    rows: pd.DataFrame,
    *,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "target_date" not in rows.columns:
        raise ValueError("rows must include target_date")
    df = rows.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    validation_cutoff = pd.Timestamp(validation_start)
    test_cutoff = pd.Timestamp(test_start)
    train_fit = df[df["target_date"] < validation_cutoff].copy()
    validation = df[
        (df["target_date"] >= validation_cutoff) & (df["target_date"] < test_cutoff)
    ].copy()
    full_train = df[df["target_date"] < test_cutoff].copy()
    test = df[df["target_date"] >= test_cutoff].copy()
    return train_fit, validation, full_train, test


def _score_alpha_candidates(
    *,
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    alphas: tuple[float, ...],
    split: str,
    target_coverage: float,
) -> pd.DataFrame:
    records = []
    for alpha in alphas:
        interval_table = fit_empirical_intervals(train_rows, alpha=alpha)
        applied = apply_empirical_intervals(eval_rows, interval_table)
        for (city, source), group in applied.groupby(["city", "source"], sort=True):
            metrics = _interval_metrics(group)
            records.append(
                {
                    "city": city,
                    "source": source,
                    "alpha": alpha,
                    "split": split,
                    "n": metrics["n"],
                    "interval_coverage_raw": metrics["coverage"],
                    "interval_width_raw": metrics["width"],
                    "target_coverage": target_coverage,
                    "coverage_shortfall": max(
                        0.0, target_coverage - float(metrics["coverage"])
                    ),
                }
            )
    return pd.DataFrame(records)


def _select_per_group_alpha(
    validation_candidates: pd.DataFrame, *, target_coverage: float
) -> pd.DataFrame:
    rows = []
    for (city, source), group in validation_candidates.groupby(["city", "source"], sort=True):
        ranked = group.copy()
        ranked["_miss_rank"] = (
            ranked["interval_coverage_raw"].astype(float) < target_coverage
        ).astype(int)
        ranked = ranked.sort_values(
            [
                "_miss_rank",
                "interval_width_raw",
                "coverage_shortfall",
                "alpha",
            ],
            ascending=[True, True, True, False],
        )
        winner = ranked.iloc[0]
        reason = (
            "narrowest_meeting_target"
            if float(winner["interval_coverage_raw"]) >= target_coverage
            else "highest_coverage_available"
        )
        rows.append(
            {
                "city": city,
                "source": source,
                "selected_alpha": float(winner["alpha"]),
                "validation_coverage": float(winner["interval_coverage_raw"]),
                "validation_width": float(winner["interval_width_raw"]),
                "validation_n": int(winner["n"]),
                "selection_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=INTERVAL_POLICY_COLUMNS)


def _fit_selected_interval_table(
    *, train_rows: pd.DataFrame, selected_policy: pd.DataFrame
) -> pd.DataFrame:
    frames = []
    for selected in selected_policy.itertuples(index=False):
        group = train_rows[
            (train_rows["city"] == selected.city) & (train_rows["source"] == selected.source)
        ]
        if group.empty:
            continue
        frames.append(fit_empirical_intervals(group, alpha=float(selected.selected_alpha)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_comparison(
    *,
    train_fit: pd.DataFrame,
    validation: pd.DataFrame,
    full_train: pd.DataFrame,
    test: pd.DataFrame,
    alphas: tuple[float, ...],
    selected_policy: pd.DataFrame,
    interval_table: pd.DataFrame,
    target_coverage: float,
) -> pd.DataFrame:
    records = []
    for split, train_rows, eval_rows in (
        ("validation", train_fit, validation),
        ("test", full_train, test),
    ):
        global_scores = _score_alpha_candidates(
            train_rows=train_rows,
            eval_rows=eval_rows,
            alphas=alphas,
            split=split,
            target_coverage=target_coverage,
        )
        for alpha, group in global_scores.groupby("alpha", sort=True):
            summary = _summarize_scores(group)
            records.append(
                {
                    "policy": "global_alpha",
                    "alpha": float(alpha),
                    "split": split,
                    **summary,
                    "target_coverage": target_coverage,
                    "coverage_shortfall": max(0.0, target_coverage - summary["interval_coverage_raw"]),
                    "recommended": False,
                }
            )

        selected_eval = apply_empirical_intervals(eval_rows, interval_table)
        selected_scores = []
        for _, group in selected_eval.groupby(["city", "source"], sort=True):
            selected_scores.append(_interval_metrics(group))
        selected_summary = _summarize_metric_records(selected_scores)
        records.append(
            {
                "policy": "per_city_alpha",
                "alpha": "selected",
                "split": split,
                **selected_summary,
                "target_coverage": target_coverage,
                "coverage_shortfall": max(
                    0.0, target_coverage - selected_summary["interval_coverage_raw"]
                ),
                "recommended": split == "test",
            }
        )
    return pd.DataFrame(records, columns=INTERVAL_POLICY_COMPARISON_COLUMNS)


def _interval_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    interval_rows = group[
        ["actual_high_f", "interval_lower_raw_f", "interval_upper_raw_f"]
    ].dropna()
    if interval_rows.empty:
        return {"n": 0, "coverage": 0.0, "width": 0.0}
    actual = interval_rows["actual_high_f"].astype(float).tolist()
    lower = interval_rows["interval_lower_raw_f"].astype(float).tolist()
    upper = interval_rows["interval_upper_raw_f"].astype(float).tolist()
    width = (
        interval_rows["interval_upper_raw_f"].astype(float)
        - interval_rows["interval_lower_raw_f"].astype(float)
    ).mean()
    return {
        "n": int(len(interval_rows)),
        "coverage": interval_coverage(actual, lower, upper),
        "width": float(width),
    }


def _summarize_scores(scores: pd.DataFrame) -> dict:
    return _summarize_metric_records(
        [
            {
                "n": int(row.n),
                "coverage": float(row.interval_coverage_raw),
                "width": float(row.interval_width_raw),
            }
            for row in scores.itertuples(index=False)
        ]
    )


def _summarize_metric_records(records: list[dict]) -> dict:
    if not records:
        return {
            "n_groups": 0,
            "n_rows": 0,
            "interval_coverage_raw": 0.0,
            "interval_width_raw": 0.0,
        }
    n_rows = sum(int(record["n"]) for record in records)
    if n_rows == 0:
        coverage = 0.0
        width = 0.0
    else:
        coverage = sum(
            float(record["coverage"]) * int(record["n"]) for record in records
        ) / n_rows
        width = sum(float(record["width"]) * int(record["n"]) for record in records) / n_rows
    return {
        "n_groups": len(records),
        "n_rows": n_rows,
        "interval_coverage_raw": float(coverage),
        "interval_width_raw": float(width),
    }

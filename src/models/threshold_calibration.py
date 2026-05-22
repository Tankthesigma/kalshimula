"""Threshold probability calibration from empirical residual distributions."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from pathlib import Path

import pandas as pd

from src.models.bias import apply_bias_correction
from src.models.bias_policy import filter_rows_to_recommended_sources
from src.models.calibration import calibration_table

EVENT_COLUMNS = [
    "city",
    "source",
    "target_date",
    "threshold_f",
    "offset_f",
    "predicted_probability",
    "outcome",
    "actual_high_f",
    "corrected_point_f",
]
RESIDUAL_COLUMNS = [
    "city",
    "source",
    "target_date",
    "residual_f",
]
SUMMARY_COLUMNS = [
    "split",
    "n_events",
    "brier_score",
    "expected_calibration_error",
    "mean_predicted_probability",
    "observed_frequency",
]
GROUP_SUMMARY_COLUMNS = [
    "split",
    "city",
    "source",
    "n_events",
    "brier_score",
    "expected_calibration_error",
    "mean_predicted_probability",
    "observed_frequency",
]
GROUP_CALIBRATION_COLUMNS = [
    "split",
    "city",
    "source",
    "bucket_index",
    "bucket_start",
    "bucket_end",
    "n",
    "mean_predicted_probability",
    "observed_frequency",
    "calibration_gap",
]
RECALIBRATION_TABLE_COLUMNS = [
    "city",
    "source",
    "bucket_index",
    "bucket_start",
    "bucket_end",
    "n",
    "mean_predicted_probability",
    "observed_frequency",
    "recalibrated_probability",
    "prior_strength",
    "min_events",
    "used",
]
RECALIBRATION_COMPARISON_COLUMNS = ["policy", *SUMMARY_COLUMNS]
PROBABILITY_GAP_REPORT_COLUMNS = [
    "city",
    "source",
    "bucket_index",
    "bucket_start",
    "bucket_end",
    "n",
    "mean_raw_probability",
    "mean_recalibrated_probability",
    "observed_frequency",
    "raw_calibration_gap",
    "recalibrated_calibration_gap",
    "abs_raw_calibration_gap",
    "abs_recalibrated_calibration_gap",
    "abs_gap_improvement",
    "city_source_recalibrated_events",
    "global_recalibrated_events",
    "unrecalibrated_events",
]
GLOBAL_RECALIBRATION_KEY = "__global__"


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    """Artifacts from threshold event probability calibration."""

    validation_events: pd.DataFrame
    test_events: pd.DataFrame
    threshold_residuals: pd.DataFrame
    validation_calibration: pd.DataFrame
    test_calibration: pd.DataFrame
    summary: pd.DataFrame
    validation_group_summary: pd.DataFrame
    test_group_summary: pd.DataFrame
    validation_group_calibration: pd.DataFrame
    test_group_calibration: pd.DataFrame
    recalibration_table: pd.DataFrame
    test_recalibrated_events: pd.DataFrame
    test_recalibrated_calibration: pd.DataFrame
    test_recalibrated_group_summary: pd.DataFrame
    test_recalibrated_group_calibration: pd.DataFrame
    recalibration_comparison: pd.DataFrame
    probability_gap_report: pd.DataFrame


def evaluate_threshold_calibration(
    *,
    rows: pd.DataFrame,
    recommended_sources: pd.DataFrame,
    bias_table: pd.DataFrame,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    offsets: tuple[int, ...] = (-4, -2, 0, 2, 4),
    n_buckets: int = 10,
    recalibration_prior_strength: float = 25.0,
    min_recalibration_events: int = 20,
    probability_gap_min_events: int = 20,
    probability_gap_min: float = 0.2,
    probability_gap_max: float = 0.8,
) -> ThresholdCalibrationResult:
    """Evaluate threshold probabilities from empirical corrected residuals.

    For each row and offset, the event is ``actual_high_f >= threshold_f`` where
    ``threshold_f`` is the rounded corrected point forecast plus the offset.
    Predicted probabilities come from the empirical residual distribution fit on
    prior rows from the same city/source.
    """
    if not offsets:
        raise ValueError("offsets must contain at least one value")
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")
    if recalibration_prior_strength < 0:
        raise ValueError("recalibration_prior_strength must be non-negative")
    if min_recalibration_events <= 0:
        raise ValueError("min_recalibration_events must be positive")
    if probability_gap_min_events <= 0:
        raise ValueError("probability_gap_min_events must be positive")
    if not 0.0 <= probability_gap_min < probability_gap_max <= 1.0:
        raise ValueError("probability gap bounds must satisfy 0 <= min < max <= 1")

    source_rows = filter_rows_to_recommended_sources(rows, recommended_sources)
    if source_rows.empty:
        raise ValueError("no rows matched recommended sources")
    corrected = apply_bias_correction(source_rows, bias_table)
    train_fit, validation, full_train, test = _split_rows(
        corrected, validation_start=validation_start, test_start=test_start
    )
    if train_fit.empty:
        raise ValueError("threshold train split is empty")
    if validation.empty:
        raise ValueError("threshold validation split is empty")
    if test.empty:
        raise ValueError("threshold test split is empty")

    validation_events = _event_rows(train_rows=train_fit, eval_rows=validation, offsets=offsets)
    test_events = _event_rows(train_rows=full_train, eval_rows=test, offsets=offsets)
    threshold_residuals = _residual_rows(full_train)
    validation_calibration = _calibration(validation_events, n_buckets=n_buckets)
    test_calibration = _calibration(test_events, n_buckets=n_buckets)
    summary = pd.DataFrame(
        [
            _summary_row("validation", validation_events, validation_calibration),
            _summary_row("test", test_events, test_calibration),
        ],
        columns=SUMMARY_COLUMNS,
    )
    validation_group_summary = _group_summary(
        "validation", validation_events, n_buckets=n_buckets
    )
    test_group_summary = _group_summary("test", test_events, n_buckets=n_buckets)
    validation_group_calibration = _group_calibration(
        "validation", validation_events, n_buckets=n_buckets
    )
    test_group_calibration = _group_calibration("test", test_events, n_buckets=n_buckets)
    recalibration_table = _recalibration_table(
        validation_events,
        n_buckets=n_buckets,
        prior_strength=recalibration_prior_strength,
        min_events=min_recalibration_events,
    )
    test_recalibrated_events = _apply_recalibration(
        test_events,
        recalibration_table=recalibration_table,
        n_buckets=n_buckets,
    )
    test_recalibrated_calibration = _calibration(
        test_recalibrated_events,
        n_buckets=n_buckets,
        probability_column="recalibrated_probability",
    )
    test_recalibrated_group_summary = _group_summary(
        "test_recalibrated",
        test_recalibrated_events,
        n_buckets=n_buckets,
        probability_column="recalibrated_probability",
    )
    test_recalibrated_group_calibration = _group_calibration(
        "test_recalibrated",
        test_recalibrated_events,
        n_buckets=n_buckets,
        probability_column="recalibrated_probability",
    )
    recalibration_comparison = _recalibration_comparison(
        raw_events=test_events,
        raw_calibration=test_calibration,
        recalibrated_events=test_recalibrated_events,
        recalibrated_calibration=test_recalibrated_calibration,
    )
    probability_gap_report = _probability_gap_report(
        test_recalibrated_events,
        n_buckets=n_buckets,
        min_events=probability_gap_min_events,
        probability_min=probability_gap_min,
        probability_max=probability_gap_max,
    )
    return ThresholdCalibrationResult(
        validation_events=validation_events,
        test_events=test_events,
        threshold_residuals=threshold_residuals,
        validation_calibration=validation_calibration,
        test_calibration=test_calibration,
        summary=summary,
        validation_group_summary=validation_group_summary,
        test_group_summary=test_group_summary,
        validation_group_calibration=validation_group_calibration,
        test_group_calibration=test_group_calibration,
        recalibration_table=recalibration_table,
        test_recalibrated_events=test_recalibrated_events,
        test_recalibrated_calibration=test_recalibrated_calibration,
        test_recalibrated_group_summary=test_recalibrated_group_summary,
        test_recalibrated_group_calibration=test_recalibrated_group_calibration,
        recalibration_comparison=recalibration_comparison,
        probability_gap_report=probability_gap_report,
    )


def write_threshold_calibration_outputs(
    *,
    input_path: Path,
    recommended_sources_path: Path,
    bias_table_path: Path,
    output_dir: Path,
    validation_start: str,
    test_start: str,
    offsets: tuple[int, ...] = (-4, -2, 0, 2, 4),
    n_buckets: int = 10,
    recalibration_prior_strength: float = 25.0,
    min_recalibration_events: int = 20,
    probability_gap_min_events: int = 20,
    probability_gap_min: float = 0.2,
    probability_gap_max: float = 0.8,
) -> ThresholdCalibrationResult:
    """Run threshold probability calibration from CSV artifacts and write outputs."""
    rows = pd.read_csv(input_path, parse_dates=["target_date"])
    recommended_sources = pd.read_csv(recommended_sources_path)
    bias_table = pd.read_csv(bias_table_path)
    result = evaluate_threshold_calibration(
        rows=rows,
        recommended_sources=recommended_sources,
        bias_table=bias_table,
        validation_start=validation_start,
        test_start=test_start,
        offsets=offsets,
        n_buckets=n_buckets,
        recalibration_prior_strength=recalibration_prior_strength,
        min_recalibration_events=min_recalibration_events,
        probability_gap_min_events=probability_gap_min_events,
        probability_gap_min=probability_gap_min,
        probability_gap_max=probability_gap_max,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.validation_events.to_csv(output_dir / "threshold_validation_events.csv", index=False)
    result.test_events.to_csv(output_dir / "threshold_test_events.csv", index=False)
    result.threshold_residuals.to_csv(output_dir / "threshold_residuals.csv", index=False)
    result.validation_calibration.to_csv(
        output_dir / "threshold_validation_calibration.csv", index=False
    )
    result.test_calibration.to_csv(output_dir / "threshold_test_calibration.csv", index=False)
    result.summary.to_csv(output_dir / "threshold_calibration_summary.csv", index=False)
    result.validation_group_summary.to_csv(
        output_dir / "threshold_validation_group_summary.csv", index=False
    )
    result.test_group_summary.to_csv(
        output_dir / "threshold_test_group_summary.csv", index=False
    )
    result.validation_group_calibration.to_csv(
        output_dir / "threshold_validation_group_calibration.csv", index=False
    )
    result.test_group_calibration.to_csv(
        output_dir / "threshold_test_group_calibration.csv", index=False
    )
    result.recalibration_table.to_csv(
        output_dir / "threshold_recalibration_table.csv", index=False
    )
    result.test_recalibrated_events.to_csv(
        output_dir / "threshold_test_recalibrated_events.csv", index=False
    )
    result.test_recalibrated_calibration.to_csv(
        output_dir / "threshold_test_recalibrated_calibration.csv", index=False
    )
    result.test_recalibrated_group_summary.to_csv(
        output_dir / "threshold_test_recalibrated_group_summary.csv", index=False
    )
    result.test_recalibrated_group_calibration.to_csv(
        output_dir / "threshold_test_recalibrated_group_calibration.csv", index=False
    )
    result.recalibration_comparison.to_csv(
        output_dir / "threshold_recalibration_comparison.csv", index=False
    )
    result.probability_gap_report.to_csv(
        output_dir / "threshold_probability_gap_report.csv", index=False
    )
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
    return (
        df[df["target_date"] < validation_cutoff].copy(),
        df[(df["target_date"] >= validation_cutoff) & (df["target_date"] < test_cutoff)].copy(),
        df[df["target_date"] < test_cutoff].copy(),
        df[df["target_date"] >= test_cutoff].copy(),
    )


def _event_rows(
    *, train_rows: pd.DataFrame, eval_rows: pd.DataFrame, offsets: tuple[int, ...]
) -> pd.DataFrame:
    residuals = _residuals_by_group(train_rows)
    records = []
    for row in eval_rows.itertuples(index=False):
        key = (row.city, row.source)
        group_residuals = residuals.get(key)
        if group_residuals is None or group_residuals.empty:
            continue
        center = _round_half_up(float(row.corrected_point_f))
        for offset in offsets:
            threshold = center + int(offset)
            needed_residual = threshold - float(row.corrected_point_f)
            probability = float((group_residuals >= needed_residual).mean())
            outcome = bool(float(row.actual_high_f) >= threshold)
            records.append(
                {
                    "city": row.city,
                    "source": row.source,
                    "target_date": row.target_date,
                    "threshold_f": threshold,
                    "offset_f": int(offset),
                    "predicted_probability": probability,
                    "outcome": outcome,
                    "actual_high_f": float(row.actual_high_f),
                    "corrected_point_f": float(row.corrected_point_f),
                }
            )
    return pd.DataFrame(records, columns=EVENT_COLUMNS)


def _residuals_by_group(rows: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    required = {"city", "source", "actual_high_f", "corrected_point_f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    df = rows.copy()
    df["residual_f"] = df["actual_high_f"].astype(float) - df["corrected_point_f"].astype(float)
    return {
        (str(city), str(source)): group["residual_f"].astype(float)
        for (city, source), group in df.groupby(["city", "source"], sort=True)
    }


def _residual_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"city", "source", "target_date", "actual_high_f", "corrected_point_f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    out = rows.copy()
    out["residual_f"] = out["actual_high_f"].astype(float) - out["corrected_point_f"].astype(float)
    return out[RESIDUAL_COLUMNS].copy()


def _calibration(
    events: pd.DataFrame,
    *,
    n_buckets: int,
    probability_column: str = "predicted_probability",
) -> pd.DataFrame:
    return calibration_table(
        predicted_probabilities=events[probability_column].astype(float).tolist(),
        outcomes=events["outcome"].astype(bool).tolist(),
        n_buckets=n_buckets,
    )


def _summary_row(
    split: str,
    events: pd.DataFrame,
    buckets: pd.DataFrame,
    *,
    probability_column: str = "predicted_probability",
) -> dict:
    if events.empty:
        return {
            "split": split,
            "n_events": 0,
            "brier_score": pd.NA,
            "expected_calibration_error": pd.NA,
            "mean_predicted_probability": pd.NA,
            "observed_frequency": pd.NA,
        }
    probabilities = events[probability_column].astype(float)
    outcomes = events["outcome"].astype(float)
    return {
        "split": split,
        "n_events": len(events),
        "brier_score": float(((probabilities - outcomes) ** 2).mean()),
        "expected_calibration_error": _expected_calibration_error(buckets),
        "mean_predicted_probability": float(probabilities.mean()),
        "observed_frequency": float(outcomes.mean()),
    }


def _group_summary(
    split: str,
    events: pd.DataFrame,
    *,
    n_buckets: int,
    probability_column: str = "predicted_probability",
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=GROUP_SUMMARY_COLUMNS)
    rows = []
    for (city, source), group in events.groupby(["city", "source"], sort=True):
        buckets = _calibration(
            group, n_buckets=n_buckets, probability_column=probability_column
        )
        row = _summary_row(split, group, buckets, probability_column=probability_column)
        rows.append(
            {
                "split": split,
                "city": city,
                "source": source,
                "n_events": row["n_events"],
                "brier_score": row["brier_score"],
                "expected_calibration_error": row["expected_calibration_error"],
                "mean_predicted_probability": row["mean_predicted_probability"],
                "observed_frequency": row["observed_frequency"],
            }
        )
    return pd.DataFrame(rows, columns=GROUP_SUMMARY_COLUMNS)


def _group_calibration(
    split: str,
    events: pd.DataFrame,
    *,
    n_buckets: int,
    probability_column: str = "predicted_probability",
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=GROUP_CALIBRATION_COLUMNS)
    rows = []
    for (city, source), group in events.groupby(["city", "source"], sort=True):
        buckets = _calibration(
            group, n_buckets=n_buckets, probability_column=probability_column
        )
        for bucket in buckets.itertuples(index=False):
            rows.append(
                {
                    "split": split,
                    "city": city,
                    "source": source,
                    "bucket_index": int(bucket.bucket_start * n_buckets),
                    "bucket_start": bucket.bucket_start,
                    "bucket_end": bucket.bucket_end,
                    "n": bucket.n,
                    "mean_predicted_probability": bucket.mean_predicted_probability,
                    "observed_frequency": bucket.observed_frequency,
                    "calibration_gap": (
                        float(bucket.mean_predicted_probability)
                        - float(bucket.observed_frequency)
                    ),
                }
            )
    return pd.DataFrame(rows, columns=GROUP_CALIBRATION_COLUMNS)


def _recalibration_table(
    validation_events: pd.DataFrame,
    *,
    n_buckets: int,
    prior_strength: float,
    min_events: int,
) -> pd.DataFrame:
    group_buckets = _group_calibration("validation", validation_events, n_buckets=n_buckets)
    global_buckets = _global_calibration_rows(validation_events, n_buckets=n_buckets)
    buckets = pd.concat([group_buckets, global_buckets], ignore_index=True)
    if buckets.empty:
        return pd.DataFrame(columns=RECALIBRATION_TABLE_COLUMNS)
    out = buckets.drop(columns=["split", "calibration_gap"]).copy()
    numerator = (
        out["observed_frequency"].astype(float) * out["n"].astype(float)
        + out["mean_predicted_probability"].astype(float) * float(prior_strength)
    )
    denominator = out["n"].astype(float) + float(prior_strength)
    out["recalibrated_probability"] = (numerator / denominator).clip(0.0, 1.0)
    out["prior_strength"] = float(prior_strength)
    out["min_events"] = int(min_events)
    out["used"] = out["n"].astype(int) >= int(min_events)
    return out[RECALIBRATION_TABLE_COLUMNS].copy()


def _global_calibration_rows(
    validation_events: pd.DataFrame, *, n_buckets: int
) -> pd.DataFrame:
    if validation_events.empty:
        return pd.DataFrame(columns=GROUP_CALIBRATION_COLUMNS)
    buckets = _calibration(validation_events, n_buckets=n_buckets)
    rows = []
    for bucket in buckets.itertuples(index=False):
        rows.append(
            {
                "split": "validation",
                "city": GLOBAL_RECALIBRATION_KEY,
                "source": GLOBAL_RECALIBRATION_KEY,
                "bucket_index": int(bucket.bucket_start * n_buckets),
                "bucket_start": bucket.bucket_start,
                "bucket_end": bucket.bucket_end,
                "n": bucket.n,
                "mean_predicted_probability": bucket.mean_predicted_probability,
                "observed_frequency": bucket.observed_frequency,
                "calibration_gap": (
                    float(bucket.mean_predicted_probability)
                    - float(bucket.observed_frequency)
                ),
            }
        )
    return pd.DataFrame(rows, columns=GROUP_CALIBRATION_COLUMNS)


def _apply_recalibration(
    events: pd.DataFrame,
    *,
    recalibration_table: pd.DataFrame,
    n_buckets: int,
) -> pd.DataFrame:
    out = events.copy()
    out["raw_predicted_probability"] = out["predicted_probability"].astype(float)
    out["bucket_index"] = _bucket_indexes(out["predicted_probability"], n_buckets=n_buckets)
    if recalibration_table.empty:
        out["recalibrated_probability"] = out["raw_predicted_probability"]
        out["recalibration_n"] = pd.NA
        out["recalibration_used"] = False
        out["recalibration_scope"] = "none"
        return out

    exact_lookup = recalibration_table[
        ["city", "source", "bucket_index", "n", "recalibrated_probability", "used"]
    ].rename(
        columns={
            "n": "exact_recalibration_n",
            "recalibrated_probability": "exact_recalibrated_probability",
            "used": "exact_recalibration_used",
        }
    )
    out = out.merge(exact_lookup, how="left", on=["city", "source", "bucket_index"])
    out["exact_recalibration_used"] = (
        out["exact_recalibration_used"].fillna(False).astype(bool)
    )

    global_lookup = recalibration_table[
        (recalibration_table["city"].astype(str) == GLOBAL_RECALIBRATION_KEY)
        & (recalibration_table["source"].astype(str) == GLOBAL_RECALIBRATION_KEY)
    ][["bucket_index", "n", "recalibrated_probability", "used"]].rename(
        columns={
            "n": "global_recalibration_n",
            "recalibrated_probability": "global_recalibrated_probability",
            "used": "global_recalibration_used",
        }
    )
    out = out.merge(global_lookup, how="left", on="bucket_index")
    out["global_recalibration_used"] = (
        out["global_recalibration_used"].fillna(False).astype(bool)
    )

    out["recalibration_scope"] = "none"
    out["recalibration_n"] = pd.NA
    exact_mask = out["exact_recalibration_used"]
    global_mask = ~exact_mask & out["global_recalibration_used"]
    out["recalibrated_probability"] = out["raw_predicted_probability"]
    out.loc[exact_mask, "recalibrated_probability"] = out.loc[
        exact_mask, "exact_recalibrated_probability"
    ]
    out.loc[exact_mask, "recalibration_n"] = out.loc[
        exact_mask, "exact_recalibration_n"
    ]
    out.loc[exact_mask, "recalibration_scope"] = "city_source"
    out.loc[global_mask, "recalibrated_probability"] = out.loc[
        global_mask, "global_recalibrated_probability"
    ]
    out.loc[global_mask, "recalibration_n"] = out.loc[
        global_mask, "global_recalibration_n"
    ]
    out.loc[global_mask, "recalibration_scope"] = "global"
    out["recalibration_used"] = exact_mask | global_mask
    out["recalibrated_probability"] = out["recalibrated_probability"].astype(float).clip(0.0, 1.0)
    out = out.drop(
        columns=[
            "exact_recalibration_n",
            "exact_recalibrated_probability",
            "exact_recalibration_used",
            "global_recalibration_n",
            "global_recalibrated_probability",
            "global_recalibration_used",
        ]
    )
    return out


def _bucket_indexes(probabilities: pd.Series, *, n_buckets: int) -> pd.Series:
    return (probabilities.astype(float).clip(lower=0.0, upper=0.999999) * n_buckets).astype(int)


def _recalibration_comparison(
    *,
    raw_events: pd.DataFrame,
    raw_calibration: pd.DataFrame,
    recalibrated_events: pd.DataFrame,
    recalibrated_calibration: pd.DataFrame,
) -> pd.DataFrame:
    raw = _summary_row("test", raw_events, raw_calibration)
    recalibrated = _summary_row(
        "test",
        recalibrated_events,
        recalibrated_calibration,
        probability_column="recalibrated_probability",
    )
    rows = [
        {"policy": "raw_empirical_residual", **raw},
        {"policy": "validation_bucket_recalibrated", **recalibrated},
    ]
    return pd.DataFrame(rows, columns=RECALIBRATION_COMPARISON_COLUMNS)


def _probability_gap_report(
    events: pd.DataFrame,
    *,
    n_buckets: int,
    min_events: int,
    probability_min: float,
    probability_max: float,
) -> pd.DataFrame:
    """Rank city/source raw-probability buckets by residual calibration gap."""
    if events.empty:
        return pd.DataFrame(columns=PROBABILITY_GAP_REPORT_COLUMNS)

    df = events.copy()
    raw_probability_column = (
        "raw_predicted_probability"
        if "raw_predicted_probability" in df.columns
        else "predicted_probability"
    )
    recalibrated_column = (
        "recalibrated_probability"
        if "recalibrated_probability" in df.columns
        else raw_probability_column
    )
    if "bucket_index" not in df.columns:
        df["bucket_index"] = _bucket_indexes(
            df[raw_probability_column], n_buckets=n_buckets
        )

    records = []
    for (city, source, bucket_index), group in df.groupby(
        ["city", "source", "bucket_index"], sort=True
    ):
        bucket_index = int(bucket_index)
        bucket_start = bucket_index / n_buckets
        bucket_end = (bucket_index + 1) / n_buckets
        if bucket_start < probability_min or bucket_end > probability_max:
            continue
        if len(group) < min_events:
            continue

        mean_raw = float(group[raw_probability_column].astype(float).mean())
        mean_recalibrated = float(group[recalibrated_column].astype(float).mean())
        observed = float(group["outcome"].astype(float).mean())
        raw_gap = mean_raw - observed
        recalibrated_gap = mean_recalibrated - observed
        scope_counts = (
            group["recalibration_scope"].value_counts()
            if "recalibration_scope" in group.columns
            else pd.Series(dtype=int)
        )
        records.append(
            {
                "city": city,
                "source": source,
                "bucket_index": bucket_index,
                "bucket_start": bucket_start,
                "bucket_end": bucket_end,
                "n": int(len(group)),
                "mean_raw_probability": mean_raw,
                "mean_recalibrated_probability": mean_recalibrated,
                "observed_frequency": observed,
                "raw_calibration_gap": raw_gap,
                "recalibrated_calibration_gap": recalibrated_gap,
                "abs_raw_calibration_gap": abs(raw_gap),
                "abs_recalibrated_calibration_gap": abs(recalibrated_gap),
                "abs_gap_improvement": abs(raw_gap) - abs(recalibrated_gap),
                "city_source_recalibrated_events": int(
                    scope_counts.get("city_source", 0)
                ),
                "global_recalibrated_events": int(scope_counts.get("global", 0)),
                "unrecalibrated_events": int(scope_counts.get("none", 0)),
            }
        )

    if not records:
        return pd.DataFrame(columns=PROBABILITY_GAP_REPORT_COLUMNS)
    return (
        pd.DataFrame(records, columns=PROBABILITY_GAP_REPORT_COLUMNS)
        .sort_values(
            ["abs_recalibrated_calibration_gap", "abs_raw_calibration_gap", "city"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def _expected_calibration_error(buckets: pd.DataFrame) -> float:
    if buckets.empty:
        return 0.0
    total = float(buckets["n"].sum())
    if total <= 0:
        return 0.0
    error = (
        buckets["n"].astype(float)
        * (
            buckets["mean_predicted_probability"].astype(float)
            - buckets["observed_frequency"].astype(float)
        ).abs()
    ).sum()
    return float(error / total)


def _round_half_up(value: float) -> int:
    return int(floor(value + 0.5))

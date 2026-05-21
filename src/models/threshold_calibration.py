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


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    """Artifacts from threshold event probability calibration."""

    validation_events: pd.DataFrame
    test_events: pd.DataFrame
    threshold_residuals: pd.DataFrame
    validation_calibration: pd.DataFrame
    test_calibration: pd.DataFrame
    summary: pd.DataFrame


def evaluate_threshold_calibration(
    *,
    rows: pd.DataFrame,
    recommended_sources: pd.DataFrame,
    bias_table: pd.DataFrame,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    offsets: tuple[int, ...] = (-4, -2, 0, 2, 4),
    n_buckets: int = 10,
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
    return ThresholdCalibrationResult(
        validation_events=validation_events,
        test_events=test_events,
        threshold_residuals=threshold_residuals,
        validation_calibration=validation_calibration,
        test_calibration=test_calibration,
        summary=summary,
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


def _calibration(events: pd.DataFrame, *, n_buckets: int) -> pd.DataFrame:
    return calibration_table(
        predicted_probabilities=events["predicted_probability"].astype(float).tolist(),
        outcomes=events["outcome"].astype(bool).tolist(),
        n_buckets=n_buckets,
    )


def _summary_row(split: str, events: pd.DataFrame, buckets: pd.DataFrame) -> dict:
    if events.empty:
        return {
            "split": split,
            "n_events": 0,
            "brier_score": pd.NA,
            "expected_calibration_error": pd.NA,
            "mean_predicted_probability": pd.NA,
            "observed_frequency": pd.NA,
        }
    probabilities = events["predicted_probability"].astype(float)
    outcomes = events["outcome"].astype(float)
    return {
        "split": split,
        "n_events": len(events),
        "brier_score": float(((probabilities - outcomes) ** 2).mean()),
        "expected_calibration_error": _expected_calibration_error(buckets),
        "mean_predicted_probability": float(probabilities.mean()),
        "observed_frequency": float(outcomes.mean()),
    }


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

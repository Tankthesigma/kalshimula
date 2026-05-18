"""Calibration table helpers."""

from __future__ import annotations

import pandas as pd

CALIBRATION_COLUMNS = [
    "bucket_start",
    "bucket_end",
    "n",
    "mean_predicted_probability",
    "observed_frequency",
]


def calibration_table(
    predicted_probabilities: list[float], outcomes: list[bool], *, n_buckets: int = 10
) -> pd.DataFrame:
    """Bucket predicted probabilities and compare with observed frequencies."""
    if not predicted_probabilities:
        return pd.DataFrame(columns=CALIBRATION_COLUMNS)
    if len(predicted_probabilities) != len(outcomes):
        raise ValueError("predicted_probabilities and outcomes must have the same length")
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")

    rows = []
    width = 1.0 / n_buckets
    for bucket in range(n_buckets):
        start = bucket * width
        end = 1.0 if bucket == n_buckets - 1 else (bucket + 1) * width
        indexes = []
        for i, probability in enumerate(predicted_probabilities):
            in_bucket = start <= probability <= end if bucket == n_buckets - 1 else start <= probability < end
            if in_bucket:
                indexes.append(i)
        if not indexes:
            continue
        predicted = [predicted_probabilities[i] for i in indexes]
        observed = [outcomes[i] for i in indexes]
        rows.append(
            {
                "bucket_start": start,
                "bucket_end": end,
                "n": len(indexes),
                "mean_predicted_probability": sum(predicted) / len(predicted),
                "observed_frequency": sum(observed) / len(observed),
            }
        )
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)

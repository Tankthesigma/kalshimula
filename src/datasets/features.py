"""Feature matrix helpers for lightweight model experiments."""

from __future__ import annotations

import pandas as pd

from src.datasets.training import TrainingExample, examples_to_dataframe

FEATURE_COLUMNS = ["forecast_high_f"]
TARGET_COLUMN = "actual_high_f"


def training_examples_to_xy(
    examples: list[TrainingExample],
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert examples to a minimal feature matrix and target vector."""
    df = examples_to_dataframe(examples)
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def add_error_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple deterministic features without mutating the input dataframe."""
    out = df.copy()
    if "forecast_high_f" in out.columns:
        out["forecast_high_f_squared"] = out["forecast_high_f"] ** 2
    if "source" in out.columns:
        out["source_code"] = pd.Categorical(out["source"]).codes
    return out

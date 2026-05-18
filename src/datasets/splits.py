"""Temporal split helpers for backtests and model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DateSplit:
    """Inclusive date ranges for train/validation/test."""

    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date


def make_three_way_date_split(
    *, start: date, validation_start: date, test_start: date, end: date
) -> DateSplit:
    """Create a chronological three-way split with no overlap."""
    if not (start < validation_start < test_start <= end):
        raise ValueError("expected start < validation_start < test_start <= end")
    return DateSplit(
        train_start=start,
        train_end=validation_start.fromordinal(validation_start.toordinal() - 1),
        validation_start=validation_start,
        validation_end=test_start.fromordinal(test_start.toordinal() - 1),
        test_start=test_start,
        test_end=end,
    )


def label_split(target: date, split: DateSplit) -> str | None:
    """Return the split label for a target date, or None when outside ranges."""
    if split.train_start <= target <= split.train_end:
        return "train"
    if split.validation_start <= target <= split.validation_end:
        return "validation"
    if split.test_start <= target <= split.test_end:
        return "test"
    return None

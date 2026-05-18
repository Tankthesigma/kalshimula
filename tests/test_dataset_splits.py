from datetime import date

import pytest

from src.datasets.splits import label_split, make_three_way_date_split


def test_make_three_way_date_split_creates_non_overlapping_ranges() -> None:
    split = make_three_way_date_split(
        start=date(2024, 1, 1),
        validation_start=date(2024, 7, 1),
        test_start=date(2024, 10, 1),
        end=date(2024, 12, 31),
    )

    assert split.train_end == date(2024, 6, 30)
    assert split.validation_end == date(2024, 9, 30)
    assert split.test_end == date(2024, 12, 31)


def test_make_three_way_date_split_rejects_bad_order() -> None:
    with pytest.raises(ValueError):
        make_three_way_date_split(
            start=date(2024, 1, 1),
            validation_start=date(2024, 1, 1),
            test_start=date(2024, 2, 1),
            end=date(2024, 3, 1),
        )


def test_label_split_returns_expected_labels() -> None:
    split = make_three_way_date_split(
        start=date(2024, 1, 1),
        validation_start=date(2024, 7, 1),
        test_start=date(2024, 10, 1),
        end=date(2024, 12, 31),
    )

    assert label_split(date(2024, 6, 1), split) == "train"
    assert label_split(date(2024, 8, 1), split) == "validation"
    assert label_split(date(2024, 11, 1), split) == "test"
    assert label_split(date(2025, 1, 1), split) is None

from datetime import date

import pytest

from src.datasets.collection import (
    build_collection_tasks,
    collection_tasks_to_dataframe,
    date_range,
)


def test_date_range_is_inclusive() -> None:
    assert date_range(date(2025, 1, 1), date(2025, 1, 3)) == [
        date(2025, 1, 1),
        date(2025, 1, 2),
        date(2025, 1, 3),
    ]


def test_date_range_rejects_reversed_dates() -> None:
    with pytest.raises(ValueError):
        date_range(date(2025, 1, 2), date(2025, 1, 1))


def test_build_collection_tasks_crosses_cities_dates_and_sources() -> None:
    tasks = build_collection_tasks(
        cities=["denver", "chicago"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        sources=["nws", "ncei"],
    )

    assert len(tasks) == 8
    assert tasks[0].city == "denver"
    assert tasks[0].target_date == date(2025, 1, 1)
    assert tasks[0].source == "nws"


def test_collection_tasks_to_dataframe_has_stable_columns() -> None:
    tasks = build_collection_tasks(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
        sources=["nws"],
    )

    df = collection_tasks_to_dataframe(tasks)

    assert list(df.columns) == ["city", "target_date", "source"]
    assert df.iloc[0].to_dict() == {
        "city": "denver",
        "target_date": date(2025, 1, 1),
        "source": "nws",
    }

"""Collection planning helpers for offline weather data jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta

import pandas as pd


@dataclass(frozen=True)
class CollectionTask:
    """One city/date/source unit of data collection work."""

    city: str
    target_date: date
    source: str


def date_range(start: date, end: date) -> list[date]:
    """Return inclusive dates from start through end."""
    if end < start:
        raise ValueError("end must be on or after start")
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def build_collection_tasks(
    *, cities: list[str], start: date, end: date, sources: list[str]
) -> list[CollectionTask]:
    """Build deterministic city/date/source collection tasks."""
    if not cities:
        raise ValueError("cities must not be empty")
    if not sources:
        raise ValueError("sources must not be empty")
    return [
        CollectionTask(city=city, target_date=target_date, source=source)
        for city in cities
        for target_date in date_range(start, end)
        for source in sources
    ]


def collection_tasks_to_dataframe(tasks: list[CollectionTask]) -> pd.DataFrame:
    """Convert collection tasks to a stable dataframe."""
    columns = ["city", "target_date", "source"]
    return pd.DataFrame([asdict(task) for task in tasks], columns=columns)

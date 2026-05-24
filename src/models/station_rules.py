"""Shared station/rule table for weather-market settlement metadata.

This module is mainline-safe: it reads station metadata only. It does not
fetch or depend on market prices, order books, or private audit results.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.config import PROJECT_ROOT
from src.fetchers.common import normalize_station

DEFAULT_STATION_RULES_PATH = PROJECT_ROOT / "config" / "station_rule_table.csv"
STATION_RULE_COLUMNS = [
    "city",
    "platform",
    "market_type",
    "settlement_station",
    "station_name",
    "timezone",
    "lst_offset",
    "dst_policy",
    "unit",
    "rounding_rule",
    "settlement_source",
    "rule_confidence",
    "notes",
]


@dataclass(frozen=True)
class StationRule:
    city: str
    platform: str
    market_type: str
    settlement_station: str
    station_name: str
    timezone: str
    lst_offset: int
    dst_policy: str
    unit: str
    rounding_rule: str
    settlement_source: str
    rule_confidence: str
    notes: str = ""


def load_station_rules(path: Path = DEFAULT_STATION_RULES_PATH) -> list[StationRule]:
    """Load the shared station/rule table."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(STATION_RULE_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"station rule table missing columns: {sorted(missing)}")
        return [_station_rule_from_row(row) for row in reader]


def station_rule_by_key(
    *,
    city: str,
    platform: str = "kalshi",
    market_type: str = "high",
    path: Path = DEFAULT_STATION_RULES_PATH,
) -> StationRule:
    """Return one rule by city/platform/market type."""
    city_key = city.strip().lower()
    platform_key = platform.strip().lower()
    market_type_key = market_type.strip().lower()
    for rule in load_station_rules(path):
        if (
            rule.city == city_key
            and rule.platform == platform_key
            and rule.market_type == market_type_key
        ):
            return rule
    raise KeyError(f"unknown station rule: {city}/{platform}/{market_type}")


def station_table_hash(path: Path = DEFAULT_STATION_RULES_PATH) -> str:
    """SHA-256 hash for manifest alignment with Bobby's private lane."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _station_rule_from_row(row: dict[str, str]) -> StationRule:
    try:
        lst_offset = int(str(row["lst_offset"]).strip())
    except ValueError as error:
        raise ValueError(f"invalid lst_offset for {row.get('city')!r}") from error
    return StationRule(
        city=str(row["city"]).strip().lower(),
        platform=str(row["platform"]).strip().lower(),
        market_type=str(row["market_type"]).strip().lower(),
        settlement_station=normalize_station(row["settlement_station"]),
        station_name=str(row["station_name"]).strip(),
        timezone=str(row["timezone"]).strip(),
        lst_offset=lst_offset,
        dst_policy=str(row["dst_policy"]).strip(),
        unit=str(row["unit"]).strip(),
        rounding_rule=str(row["rounding_rule"]).strip(),
        settlement_source=str(row["settlement_source"]).strip(),
        rule_confidence=str(row["rule_confidence"]).strip().lower(),
        notes=str(row.get("notes") or "").strip(),
    )

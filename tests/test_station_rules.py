from pathlib import Path

import pytest

from src.models.station_rules import (
    load_station_rules,
    station_rule_by_key,
    station_table_hash,
)


def test_load_station_rules_from_shared_table() -> None:
    rules = load_station_rules()

    chicago = next(rule for rule in rules if rule.city == "chicago")
    assert chicago.settlement_station == "KMDW"
    assert chicago.platform == "kalshi"
    assert chicago.market_type == "high"
    assert chicago.lst_offset == -6


def test_station_rule_by_key_normalizes_lookup() -> None:
    rule = station_rule_by_key(city=" Chicago ", platform="KALSHI", market_type="HIGH")

    assert rule.settlement_station == "KMDW"


def test_low_station_rules_are_available_but_not_high_confidence() -> None:
    rule = station_rule_by_key(city="nyc", platform="kalshi", market_type="low")

    assert rule.settlement_station == "KNYC"
    assert rule.market_type == "low"
    assert rule.rule_confidence == "medium"


def test_station_table_hash_is_stable_hex() -> None:
    digest = station_table_hash()

    assert len(digest) == 64
    int(digest, 16)


def test_missing_required_station_rule_columns_raises(tmp_path: Path) -> None:
    bad = tmp_path / "station_rule_table.csv"
    bad.write_text("city,platform\nnyc,kalshi\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        load_station_rules(bad)

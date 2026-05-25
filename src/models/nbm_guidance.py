"""Normalize NBM text guidance into mainline guidance rows.

This module is mainline-safe. It fetches public NOAA/NOMADS weather guidance
only and emits the same normalized guidance schema used by NWS guidance.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import pandas as pd

from src.models.guidance import GUIDANCE_COLUMNS, normalize_guidance_rows
from src.models.station_rules import (
    DEFAULT_STATION_RULES_PATH,
    StationRule,
    load_station_rules,
)

NBM_TEXT_SOURCE: Final = "nbm_text"
NOMADS_BLEND_BASE_URL: Final = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"
NBH_PRODUCT: Final = "nbh"
NBP_PRODUCT: Final = "nbp"
NBM_USER_AGENT: Final = "kalshimula-model-longrun/nbm-guidance"
PERCENTILE_ROWS: Final = {
    "TXNP1": "guidance_q10_f",
    "TXNP5": "guidance_q50_f",
    "TXNP9": "guidance_q90_f",
}


@dataclass(frozen=True)
class NbmTextBundle:
    product: str
    cycle_date: date
    cycle_hour: int
    text: str
    url: str


def latest_cycle_at_or_before(as_of_ts: datetime | str) -> tuple[date, int]:
    """Return the latest NBM cycle not after ``as_of_ts``.

    NBM text products are cycle-hour keyed. We avoid guessing publication lag
    here; callers should pass an ``as_of_ts`` after the cycle is known to be
    available, or validate availability by fetch success.
    """
    as_of = _parse_utc(as_of_ts)
    return as_of.date(), as_of.hour


def fetch_nbm_text_product(
    *,
    product: str,
    cycle_date: date,
    cycle_hour: int,
    base_url: str = NOMADS_BLEND_BASE_URL,
) -> NbmTextBundle:
    """Fetch one operational NBM text bulletin from NOMADS."""
    product_key = product.strip().lower()
    if product_key not in {NBH_PRODUCT, NBP_PRODUCT}:
        raise ValueError(f"unsupported NBM text product: {product}")
    if not 0 <= cycle_hour <= 23:
        raise ValueError(f"cycle_hour must be 0-23: {cycle_hour}")
    ymd = cycle_date.strftime("%Y%m%d")
    url = (
        f"{base_url.rstrip('/')}/blend.{ymd}/{cycle_hour:02d}/text/"
        f"blend_{product_key}tx.t{cycle_hour:02d}z"
    )
    request = urllib.request.Request(url, headers={"User-Agent": NBM_USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    return NbmTextBundle(product=product_key, cycle_date=cycle_date, cycle_hour=cycle_hour, text=text, url=url)


def build_nbm_guidance_rows(
    *,
    nbh_text: str,
    target: date,
    as_of_ts: datetime | str,
    rules: list[StationRule] | None = None,
    nbp_text: str | None = None,
    source: str = NBM_TEXT_SOURCE,
) -> pd.DataFrame:
    """Build normalized NBM guidance rows for all high-temperature station rules.

    ``NBH`` hourly station guidance is used as the deterministic point. If
    ``NBP`` percentile guidance is present for the same target settlement date,
    q10/q50/q90 are attached and q50 becomes the point. Otherwise the point is
    the max of hourly NBM temperature over the station's LST settlement date.
    """
    selected_rules = [
        rule
        for rule in (rules or load_station_rules(DEFAULT_STATION_RULES_PATH))
        if rule.platform == "kalshi" and rule.market_type == "high"
    ]
    as_of = _parse_utc(as_of_ts)
    rows = []
    for rule in selected_rules:
        hourly = _hourly_point_for_rule(nbh_text, rule=rule, target=target)
        percentiles = (
            _percentiles_for_rule(nbp_text, rule=rule, target=target)
            if nbp_text is not None
            else {}
        )
        has_percentile_point = "guidance_q50_f" in percentiles
        point = percentiles.get("guidance_q50_f", hourly.get("point"))
        if point is None:
            continue
        issue_ts = (
            percentiles.get("issue_ts") if has_percentile_point else hourly.get("issue_ts")
        ) or as_of
        valid_ts = (
            percentiles.get("valid_ts") if has_percentile_point else hourly.get("valid_ts")
        ) or as_of
        rows.append(
            {
                "city": rule.city,
                "source": source,
                "station_id": rule.settlement_station,
                "market_type": rule.market_type,
                "target_date": target.isoformat(),
                "issue_ts_utc": issue_ts.isoformat(),
                "valid_ts_utc": valid_ts.isoformat(),
                "available_ts_utc": issue_ts.isoformat(),
                "guidance_point_f": point,
                "guidance_q10_f": percentiles.get("guidance_q10_f", pd.NA),
                "guidance_q50_f": percentiles.get("guidance_q50_f", point),
                "guidance_q90_f": percentiles.get("guidance_q90_f", pd.NA),
                "actual_high_f": pd.NA,
                "raw_payload_hash": _hash_texts(nbh_text, nbp_text or ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=GUIDANCE_COLUMNS)
    return normalize_guidance_rows(pd.DataFrame(rows))


def fetch_nbm_guidance_rows(
    *,
    target: date,
    as_of_ts: datetime | str,
    station_rules_path: Path = DEFAULT_STATION_RULES_PATH,
    cities: list[str] | None = None,
    market_types: list[str] | None = None,
    base_url: str = NOMADS_BLEND_BASE_URL,
    max_cycle_lookback_hours: int = 6,
) -> pd.DataFrame:
    """Fetch current NBM text guidance and return normalized guidance rows."""
    nbh = _fetch_latest_available_product(
        product=NBH_PRODUCT,
        as_of_ts=as_of_ts,
        base_url=base_url,
        max_cycle_lookback_hours=max_cycle_lookback_hours,
    )
    try:
        nbp = fetch_nbm_text_product(
            product=NBP_PRODUCT,
            cycle_date=nbh.cycle_date,
            cycle_hour=nbh.cycle_hour,
            base_url=base_url,
        )
        nbp_text = nbp.text
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        nbp_text = None
    return build_nbm_guidance_rows(
        nbh_text=nbh.text,
        nbp_text=nbp_text,
        target=target,
        as_of_ts=as_of_ts,
        rules=_select_rules(
            load_station_rules(station_rules_path),
            cities=cities,
            market_types=market_types,
        ),
    )


def _fetch_latest_available_product(
    *,
    product: str,
    as_of_ts: datetime | str,
    base_url: str,
    max_cycle_lookback_hours: int,
) -> NbmTextBundle:
    as_of = _parse_utc(as_of_ts)
    for offset in range(max_cycle_lookback_hours + 1):
        cycle_ts = as_of - timedelta(hours=offset)
        try:
            return fetch_nbm_text_product(
                product=product,
                cycle_date=cycle_ts.date(),
                cycle_hour=cycle_ts.hour,
                base_url=base_url,
            )
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            continue
    raise FileNotFoundError(
        f"no available NBM {product} product within {max_cycle_lookback_hours}h of {as_of.isoformat()}"
    )


def write_nbm_guidance_rows(
    *,
    output_path: Path,
    target: date,
    as_of_ts: datetime | str,
    station_rules_path: Path = DEFAULT_STATION_RULES_PATH,
    cities: list[str] | None = None,
    market_types: list[str] | None = None,
    base_url: str = NOMADS_BLEND_BASE_URL,
) -> pd.DataFrame:
    """Fetch and write normalized NBM guidance rows."""
    rows = fetch_nbm_guidance_rows(
        target=target,
        as_of_ts=as_of_ts,
        station_rules_path=station_rules_path,
        cities=cities,
        market_types=market_types,
        base_url=base_url,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)
    return rows


def _select_rules(
    rules: list[StationRule],
    *,
    cities: list[str] | None,
    market_types: list[str] | None,
) -> list[StationRule]:
    city_set = {city.strip().lower() for city in cities or [] if city.strip()}
    market_set = {market.strip().lower() for market in market_types or [] if market.strip()}
    return [
        rule
        for rule in rules
        if (not city_set or rule.city in city_set)
        and (not market_set or rule.market_type in market_set)
    ]


def _hourly_point_for_rule(text: str, *, rule: StationRule, target: date) -> dict[str, object]:
    block = _station_block(text, rule.settlement_station)
    if block is None:
        return {}
    issue = _issue_ts(block)
    temps = _row_numbers(block, "TMP")
    if issue is None or not temps:
        return {}
    fhrs = [int(value) for value in _row_numbers(block, "FHR")]
    valid_points = []
    for idx, temp_f in enumerate(temps):
        fhr = fhrs[idx] if idx < len(fhrs) else idx + 1
        valid_ts = issue + timedelta(hours=fhr)
        settlement_date = (valid_ts + timedelta(hours=rule.lst_offset)).date()
        if settlement_date == target:
            valid_points.append((valid_ts, temp_f))
    if not valid_points:
        return {}
    valid_ts, point = max(valid_points, key=lambda item: item[1])
    return {"point": point, "issue_ts": issue, "valid_ts": valid_ts}


def _percentiles_for_rule(text: str, *, rule: StationRule, target: date) -> dict[str, object]:
    block = _station_block(text, rule.settlement_station)
    if block is None:
        return {}
    issue = _issue_ts(block)
    fhrs = _row_numbers(block, "FHR")
    if issue is None or not fhrs:
        return {}
    row_values = {name: _row_numbers(block, name) for name in PERCENTILE_ROWS}
    candidates: list[tuple[datetime, dict[str, float]]] = []
    for idx, fhr in enumerate(fhrs):
        valid_ts = issue + timedelta(hours=int(fhr))
        settlement_date = (valid_ts + timedelta(hours=rule.lst_offset)).date()
        if settlement_date != target:
            continue
        values: dict[str, float] = {}
        for row_name, column in PERCENTILE_ROWS.items():
            numbers = row_values.get(row_name, [])
            if idx < len(numbers):
                values[column] = numbers[idx]
        if values:
            candidates.append((valid_ts, values))
    if not candidates:
        return {}
    valid_ts, values = max(candidates, key=lambda item: len(item[1]))
    return {**values, "issue_ts": issue, "valid_ts": valid_ts}


def _station_block(text: str, station_id: str) -> str | None:
    pattern = re.compile(rf"(?m)^\s*{re.escape(station_id.upper())}\s+NBM V[^\n]*")
    match = pattern.search(text)
    if match is None:
        return None
    next_match = re.search(r"(?m)^\s*[A-Z0-9]{3,6}\s+NBM V", text[match.end() :])
    end = match.end() + next_match.start() if next_match else len(text)
    return text[match.start() : end]


def _issue_ts(block: str) -> datetime | None:
    first_line = next((line for line in block.splitlines() if line.strip()), "")
    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{2})(\d{2})\s+UTC", first_line)
    if match is None:
        return None
    month, day, year, hour, minute = (int(part) for part in match.groups())
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _row_numbers(block: str, row_name: str) -> list[float]:
    pattern = re.compile(rf"(?m)^\s*{re.escape(row_name)}\s+(.+)$")
    match = pattern.search(block)
    if match is None:
        return []
    return [float(value) for value in re.findall(r"-?\d+", match.group(1))]


def _parse_utc(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value)
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _hash_texts(*texts: str) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8", errors="replace"))
    return digest.hexdigest()

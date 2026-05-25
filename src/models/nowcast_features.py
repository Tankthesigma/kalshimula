"""Weather-only nowcast features from point-in-time station observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.fetchers.asos import (
    AsosHourlyObservation,
    fetch_asos_observation_csv,
    parse_asos_csv,
)
from src.models.station_rules import (
    DEFAULT_STATION_RULES_PATH,
    StationRule,
    load_station_rules,
    station_table_hash,
)

OBSERVATION_COLUMNS = [
    "station_id",
    "obs_ts_utc",
    "available_ts_utc",
    "temperature_f",
    "dewpoint_f",
    "wind_speed_kt",
    "wind_direction_deg",
    "gust_kt",
    "cloud_cover",
    "pressure_mb",
    "precip_in",
    "source",
]
DEFAULT_OBSERVATION_AVAILABILITY_LAG_MINUTES = 10
NOWCAST_FEATURE_COLUMNS = [
    "city",
    "platform",
    "market_type",
    "station_id",
    "target_date",
    "prediction_ts_utc",
    "prediction_time_local",
    "decision_time_label",
    "as_of_ts_utc",
    "latest_obs_ts_utc",
    "latest_temp_f",
    "latest_dewpoint_f",
    "high_so_far_f",
    "low_so_far_f",
    "latest_minus_high_so_far_f",
    "latest_minus_low_so_far_f",
    "temp_1h_slope_f",
    "temp_3h_slope_f",
    "dewpoint_depression_f",
    "wind_speed_kt",
    "cloud_cover",
    "hours_since_sunrise",
    "hours_to_solar_noon",
    "hours_to_sunset",
    "radiative_cooling_index",
    "remaining_heating_estimate_f",
    "remaining_cooling_estimate_f",
    "nowcast_veto_flag",
    "weather_reason_codes",
    "station_rule_confidence",
    "feature_hash",
]
OBSERVATION_COVERAGE_COLUMNS = [
    "city",
    "platform",
    "market_type",
    "station_id",
    "target_date",
    "decision_time_label",
    "as_of_ts_utc",
    "obs_count_available",
    "temp_obs_count_available",
    "first_obs_ts_utc",
    "latest_obs_ts_utc",
    "minutes_since_latest_obs",
    "high_so_far_f",
    "low_so_far_f",
    "coverage_ok",
    "coverage_reason_codes",
]


@dataclass(frozen=True)
class NowcastFeatureResult:
    observations: pd.DataFrame
    features: pd.DataFrame
    coverage: pd.DataFrame
    report: str
    manifest: dict[str, object]


def observations_to_frame(
    observations: list[AsosHourlyObservation],
    *,
    availability_lag_minutes: int = DEFAULT_OBSERVATION_AVAILABILITY_LAG_MINUTES,
) -> pd.DataFrame:
    """Convert parsed ASOS observations to the canonical observation store shape."""
    rows = []
    for obs in observations:
        obs_ts = _as_utc_naive(obs.valid_time)
        available_ts = obs_ts + timedelta(minutes=availability_lag_minutes)
        rows.append(
            {
                "station_id": obs.station,
                "obs_ts_utc": obs_ts.isoformat(),
                "available_ts_utc": available_ts.isoformat(),
                "temperature_f": obs.temp_f,
                "dewpoint_f": obs.dewpoint_f,
                "wind_speed_kt": obs.wind_speed_kt,
                "wind_direction_deg": obs.wind_direction_deg,
                "gust_kt": obs.gust_kt,
                "cloud_cover": obs.cloud_cover,
                "pressure_mb": obs.pressure_mb,
                "precip_in": obs.precip_in,
                "source": obs.source,
            }
        )
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)


def fetch_observations_for_rules(
    rules: list[StationRule], *, start: date, end: date
) -> pd.DataFrame:
    """Fetch ASOS observations for station rules and return canonical rows."""
    all_observations: list[pd.DataFrame] = []
    seen_stations: set[str] = set()
    for rule in rules:
        if rule.settlement_station in seen_stations:
            continue
        seen_stations.add(rule.settlement_station)
        try:
            text = fetch_asos_observation_csv(rule.settlement_station, start, end)
            parsed = parse_asos_csv(text, rule.settlement_station)
            all_observations.append(observations_to_frame(parsed))
        except Exception:  # noqa: BLE001
            all_observations.append(pd.DataFrame(columns=OBSERVATION_COLUMNS))
    if not all_observations:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    return pd.concat(all_observations, ignore_index=True)


def load_observation_store(path: Path) -> pd.DataFrame:
    """Load an ASOS observation store from CSV."""
    if not path.exists():
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    rows = pd.read_csv(path)
    missing = set(OBSERVATION_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"observation store missing columns: {sorted(missing)}")
    return rows.loc[:, OBSERVATION_COLUMNS].copy()


def merge_observation_store(
    existing: pd.DataFrame | None,
    new_rows: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge observation rows with deterministic de-duplication."""
    frames = [
        frame
        for frame in (existing, new_rows)
        if frame is not None and not frame.empty
    ]
    if not frames:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    for column in OBSERVATION_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    merged = merged.loc[:, OBSERVATION_COLUMNS]
    merged["_available_sort"] = pd.to_datetime(
        merged["available_ts_utc"],
        errors="coerce",
    )
    merged = (
        merged.sort_values(["station_id", "obs_ts_utc", "_available_sort"])
        .drop_duplicates(["station_id", "obs_ts_utc"], keep="last")
        .drop(columns=["_available_sort"])
        .sort_values(["station_id", "obs_ts_utc"])
        .reset_index(drop=True)
    )
    return merged


def write_observation_store(path: Path, observations: pd.DataFrame) -> None:
    """Write the canonical observation store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    merge_observation_store(None, observations).to_csv(path, index=False)


def build_nowcast_features(
    observations: pd.DataFrame,
    rules: list[StationRule],
    *,
    target_date: date,
    as_of_ts: datetime,
    decision_time_label: str,
) -> pd.DataFrame:
    """Build no-leak weather-only nowcast features for station rules.

    Only observations with ``available_ts_utc <= as_of_ts`` are used.
    """
    as_of_utc = _as_utc_naive(as_of_ts)
    obs = _clean_observations(observations)
    obs = obs[obs["available_ts_utc"] <= as_of_utc]
    rows = []
    for rule in rules:
        station_obs = _station_observations_for_target_date(
            obs,
            rule=rule,
            target_date=target_date,
        )
        rows.append(
            _feature_row(
                rule=rule,
                station_obs=station_obs,
                target_date=target_date,
                as_of_utc=as_of_utc,
                decision_time_label=decision_time_label,
            )
        )
    return pd.DataFrame(rows, columns=NOWCAST_FEATURE_COLUMNS)


def build_observation_coverage(
    observations: pd.DataFrame,
    rules: list[StationRule],
    *,
    target_date: date,
    as_of_ts: datetime,
    decision_time_label: str,
) -> pd.DataFrame:
    """Summarize ASOS observation coverage without changing feature schema."""
    as_of_utc = _as_utc_naive(as_of_ts)
    obs = _clean_observations(observations)
    obs = obs[obs["available_ts_utc"] <= as_of_utc]
    rows = []
    for rule in rules:
        station_obs = _station_observations_for_target_date(
            obs,
            rule=rule,
            target_date=target_date,
        )
        rows.append(
            _coverage_row(
                rule=rule,
                station_obs=station_obs,
                target_date=target_date,
                as_of_utc=as_of_utc,
                decision_time_label=decision_time_label,
            )
        )
    return pd.DataFrame(rows, columns=OBSERVATION_COVERAGE_COLUMNS)


def render_nowcast_feature_report(features: pd.DataFrame) -> str:
    """Render a compact weather-only nowcast feature report."""
    lines = [
        "# Nowcast Feature Report",
        "",
        "Weather-only features. No market prices, order books, private PnL labels, or trade instructions.",
        "",
    ]
    if features.empty:
        return "\n".join([*lines, "No rows.", ""])
    lines.append("| city | station | latest | high so far | low so far | veto | reasons |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for row in features.itertuples(index=False):
        lines.append(
            f"| {row.city} | {row.station_id} | {_fmt(row.latest_temp_f)} | "
            f"{_fmt(row.high_so_far_f)} | {_fmt(row.low_so_far_f)} | "
            f"{row.nowcast_veto_flag} | {row.weather_reason_codes} |"
        )
    return "\n".join(lines) + "\n"


def write_nowcast_features(
    *,
    output_dir: Path,
    target_date: date,
    as_of_ts: datetime,
    decision_time_label: str,
    observations: pd.DataFrame | None = None,
    observation_store_path: Path | None = None,
    update_observation_store: bool = False,
    station_rules_path: Path = DEFAULT_STATION_RULES_PATH,
    market_types: list[str] | None = None,
    cities: list[str] | None = None,
    fetch_live: bool = False,
    git_commit: str | None = None,
) -> NowcastFeatureResult:
    """Build and write nowcast feature artifacts."""
    rules = _filter_rules(
        load_station_rules(station_rules_path),
        market_types=market_types,
        cities=cities,
    )
    store_observations = (
        load_observation_store(observation_store_path)
        if observation_store_path is not None
        else pd.DataFrame(columns=OBSERVATION_COLUMNS)
    )
    if observations is None:
        if not fetch_live and observation_store_path is None:
            raise ValueError("observations are required unless fetch_live=True")
        observations = pd.DataFrame(columns=OBSERVATION_COLUMNS)
    if fetch_live:
        live_observations = fetch_observations_for_rules(
            rules,
            start=target_date,
            end=target_date,
        )
        observations = merge_observation_store(observations, live_observations)
    observations = merge_observation_store(store_observations, observations)
    if observation_store_path is not None and update_observation_store:
        write_observation_store(observation_store_path, observations)
    features = build_nowcast_features(
        observations,
        rules,
        target_date=target_date,
        as_of_ts=as_of_ts,
        decision_time_label=decision_time_label,
    )
    coverage = build_observation_coverage(
        observations,
        rules,
        target_date=target_date,
        as_of_ts=as_of_ts,
        decision_time_label=decision_time_label,
    )
    coverage = _reconcile_feature_coverage(features, coverage)
    report = render_nowcast_feature_report(features)
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "station_rules_path": str(station_rules_path),
        "observation_store_path": (
            str(observation_store_path) if observation_store_path is not None else None
        ),
        "observation_store_updated": bool(
            observation_store_path is not None and update_observation_store
        ),
        "station_table_hash": station_table_hash(station_rules_path),
        "target_date": target_date.isoformat(),
        "cities": sorted({rule.city for rule in rules}),
        "as_of_ts_utc": _utc_iso(as_of_ts),
        "decision_time_label": decision_time_label,
        "no_leak_max_observation_ts": _max_obs_ts(features),
        "row_counts": {
            "observations": int(len(observations)),
            "features": int(len(features)),
            "coverage": int(len(coverage)),
        },
        "coverage_summary": _coverage_summary(coverage),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output_dir / "asos_observations.csv", index=False)
    features.to_csv(output_dir / "nowcast_features.csv", index=False)
    coverage.to_csv(output_dir / "observation_coverage.csv", index=False)
    (output_dir / "nowcast_features_report.md").write_text(report, encoding="utf-8")
    (output_dir / "nowcast_features_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NowcastFeatureResult(
        observations=observations,
        features=features,
        coverage=coverage,
        report=report,
        manifest=manifest,
    )


def _filter_rules(
    rules: list[StationRule],
    *,
    market_types: list[str] | None,
    cities: list[str] | None = None,
) -> list[StationRule]:
    if market_types is None:
        market_type_set = {"high"}
    else:
        market_type_set = {market_type.strip().lower() for market_type in market_types}
    city_set = (
        {city.strip().lower() for city in cities if city.strip()}
        if cities is not None
        else None
    )
    return [
        rule
        for rule in rules
        if rule.market_type in market_type_set
        and (city_set is None or rule.city in city_set)
    ]


def _feature_row(
    *,
    rule: StationRule,
    station_obs: pd.DataFrame,
    target_date: date,
    as_of_utc: datetime,
    decision_time_label: str,
) -> dict[str, object]:
    reasons: list[str] = []
    latest = station_obs.tail(1)
    if latest.empty:
        reasons.append("missing_observations")
        local = _local_time(as_of_utc, rule)
        return _empty_feature_row(rule, target_date, as_of_utc, local, decision_time_label, reasons)

    latest_row = latest.iloc[0]
    latest_temp = _number_or_na(latest_row["temperature_f"])
    latest_obs_ts = latest_row["obs_ts_utc"]
    if pd.isna(latest_temp):
        reasons.append("missing_latest_temp")
    if as_of_utc - latest_obs_ts > timedelta(hours=2):
        reasons.append("stale_observation")

    temps = station_obs["temperature_f"].dropna().astype(float)
    high_so_far = float(temps.max()) if not temps.empty else pd.NA
    low_so_far = float(temps.min()) if not temps.empty else pd.NA
    one_hour_slope = _slope(station_obs, latest_obs_ts, hours=1)
    three_hour_slope = _slope(station_obs, latest_obs_ts, hours=3)
    dewpoint = _number_or_na(latest_row["dewpoint_f"])
    wind = _number_or_na(latest_row["wind_speed_kt"])
    cloud_cover = latest_row.get("cloud_cover")
    dewpoint_depression = (
        latest_temp - dewpoint
        if not pd.isna(latest_temp) and not pd.isna(dewpoint)
        else pd.NA
    )
    latest_minus_high = (
        latest_temp - high_so_far
        if not pd.isna(latest_temp) and not pd.isna(high_so_far)
        else pd.NA
    )
    latest_minus_low = (
        latest_temp - low_so_far
        if not pd.isna(latest_temp) and not pd.isna(low_so_far)
        else pd.NA
    )
    local = _local_time(as_of_utc, rule)
    solar = _solar_features(local)
    remaining_heating = _remaining_heating(local, one_hour_slope)
    remaining_cooling = _remaining_cooling(local, wind, cloud_cover)
    cooling_index = _radiative_cooling_index(wind, cloud_cover, dewpoint_depression)
    if pd.isna(dewpoint):
        reasons.append("missing_dewpoint")
    if pd.isna(wind):
        reasons.append("missing_wind")
    if not cloud_cover:
        reasons.append("missing_cloud_cover")
    veto = bool(reasons)
    feature_hash = _feature_hash(rule.settlement_station, as_of_utc, latest_obs_ts, high_so_far, low_so_far)
    return {
        "city": rule.city,
        "platform": rule.platform,
        "market_type": rule.market_type,
        "station_id": rule.settlement_station,
        "target_date": target_date.isoformat(),
        "prediction_ts_utc": _utc_iso(as_of_utc),
        "prediction_time_local": local.isoformat(),
        "decision_time_label": decision_time_label,
        "as_of_ts_utc": _utc_iso(as_of_utc),
        "latest_obs_ts_utc": latest_obs_ts.isoformat(),
        "latest_temp_f": latest_temp,
        "latest_dewpoint_f": dewpoint,
        "high_so_far_f": high_so_far,
        "low_so_far_f": low_so_far,
        "latest_minus_high_so_far_f": latest_minus_high,
        "latest_minus_low_so_far_f": latest_minus_low,
        "temp_1h_slope_f": one_hour_slope,
        "temp_3h_slope_f": three_hour_slope,
        "dewpoint_depression_f": dewpoint_depression,
        "wind_speed_kt": wind,
        "cloud_cover": cloud_cover,
        "hours_since_sunrise": solar["hours_since_sunrise"],
        "hours_to_solar_noon": solar["hours_to_solar_noon"],
        "hours_to_sunset": solar["hours_to_sunset"],
        "radiative_cooling_index": cooling_index,
        "remaining_heating_estimate_f": remaining_heating,
        "remaining_cooling_estimate_f": remaining_cooling,
        "nowcast_veto_flag": veto,
        "weather_reason_codes": ";".join(reasons),
        "station_rule_confidence": rule.rule_confidence,
        "feature_hash": feature_hash,
    }


def _empty_feature_row(
    rule: StationRule,
    target_date: date,
    as_of_utc: datetime,
    local: datetime,
    decision_time_label: str,
    reasons: list[str],
) -> dict[str, object]:
    return {
        "city": rule.city,
        "platform": rule.platform,
        "market_type": rule.market_type,
        "station_id": rule.settlement_station,
        "target_date": target_date.isoformat(),
        "prediction_ts_utc": _utc_iso(as_of_utc),
        "prediction_time_local": local.isoformat(),
        "decision_time_label": decision_time_label,
        "as_of_ts_utc": _utc_iso(as_of_utc),
        "latest_obs_ts_utc": pd.NA,
        "latest_temp_f": pd.NA,
        "latest_dewpoint_f": pd.NA,
        "high_so_far_f": pd.NA,
        "low_so_far_f": pd.NA,
        "latest_minus_high_so_far_f": pd.NA,
        "latest_minus_low_so_far_f": pd.NA,
        "temp_1h_slope_f": pd.NA,
        "temp_3h_slope_f": pd.NA,
        "dewpoint_depression_f": pd.NA,
        "wind_speed_kt": pd.NA,
        "cloud_cover": pd.NA,
        "hours_since_sunrise": _solar_features(local)["hours_since_sunrise"],
        "hours_to_solar_noon": _solar_features(local)["hours_to_solar_noon"],
        "hours_to_sunset": _solar_features(local)["hours_to_sunset"],
        "radiative_cooling_index": pd.NA,
        "remaining_heating_estimate_f": pd.NA,
        "remaining_cooling_estimate_f": pd.NA,
        "nowcast_veto_flag": True,
        "weather_reason_codes": ";".join(reasons),
        "station_rule_confidence": rule.rule_confidence,
        "feature_hash": _feature_hash(rule.settlement_station, as_of_utc, None, pd.NA, pd.NA),
    }


def _coverage_row(
    *,
    rule: StationRule,
    station_obs: pd.DataFrame,
    target_date: date,
    as_of_utc: datetime,
    decision_time_label: str,
) -> dict[str, object]:
    reasons: list[str] = []
    temps = station_obs["temperature_f"].dropna().astype(float) if not station_obs.empty else pd.Series(dtype=float)
    latest_obs_ts = pd.NA
    first_obs_ts = pd.NA
    minutes_since_latest = pd.NA
    if station_obs.empty:
        reasons.append("missing_observations")
    else:
        first_obs_ts = station_obs["obs_ts_utc"].min()
        latest_obs_ts = station_obs["obs_ts_utc"].max()
        minutes_since_latest = round((as_of_utc - latest_obs_ts).total_seconds() / 60, 3)
        if as_of_utc - latest_obs_ts > timedelta(hours=2):
            reasons.append("stale_observation")
    if temps.empty:
        reasons.append("missing_temperature")
    elif len(temps) < _expected_min_temp_obs(as_of_utc, rule, target_date):
        reasons.append("thin_temperature_coverage")
    return {
        "city": rule.city,
        "platform": rule.platform,
        "market_type": rule.market_type,
        "station_id": rule.settlement_station,
        "target_date": target_date.isoformat(),
        "decision_time_label": decision_time_label,
        "as_of_ts_utc": _utc_iso(as_of_utc),
        "obs_count_available": int(len(station_obs)),
        "temp_obs_count_available": int(len(temps)),
        "first_obs_ts_utc": first_obs_ts.isoformat() if not pd.isna(first_obs_ts) else pd.NA,
        "latest_obs_ts_utc": latest_obs_ts.isoformat() if not pd.isna(latest_obs_ts) else pd.NA,
        "minutes_since_latest_obs": minutes_since_latest,
        "high_so_far_f": float(temps.max()) if not temps.empty else pd.NA,
        "low_so_far_f": float(temps.min()) if not temps.empty else pd.NA,
        "coverage_ok": not reasons,
        "coverage_reason_codes": ";".join(reasons),
    }


def _clean_observations(observations: pd.DataFrame) -> pd.DataFrame:
    obs = observations.copy()
    if obs.empty:
        out = pd.DataFrame(columns=OBSERVATION_COLUMNS)
        out["obs_ts_utc"] = pd.to_datetime(out["obs_ts_utc"])
        out["available_ts_utc"] = pd.to_datetime(out["available_ts_utc"])
        return out
    required = {"station_id", "obs_ts_utc", "available_ts_utc", "temperature_f"}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")
    obs["station_id"] = obs["station_id"].astype(str).str.strip().str.upper()
    obs["obs_ts_utc"] = _parse_utc_timestamp_series(obs["obs_ts_utc"])
    obs["available_ts_utc"] = _parse_utc_timestamp_series(obs["available_ts_utc"])
    for col in ("temperature_f", "dewpoint_f", "wind_speed_kt", "wind_direction_deg", "gust_kt", "pressure_mb", "precip_in"):
        if col in obs.columns:
            obs[col] = pd.to_numeric(obs[col], errors="coerce")
    return obs.dropna(subset=["station_id", "obs_ts_utc", "available_ts_utc"])


def _station_observations_for_target_date(
    observations: pd.DataFrame,
    *,
    rule: StationRule,
    target_date: date,
) -> pd.DataFrame:
    """Return station rows in the market settlement date, not UTC date.

    Daily weather markets settle on station-local/LST days. Filtering by UTC
    date leaks prior-evening observations into western cities; e.g. Phoenix
    00:51Z on May 4 is still May 3 local evening.
    """
    if observations.empty:
        return observations
    station_obs = observations[observations["station_id"] == rule.settlement_station].copy()
    if station_obs.empty:
        return station_obs
    station_obs["_settlement_date"] = (
        station_obs["obs_ts_utc"] + timedelta(hours=rule.lst_offset)
    ).dt.date
    return (
        station_obs[station_obs["_settlement_date"] == target_date]
        .drop(columns=["_settlement_date"])
        .sort_values("obs_ts_utc")
    )


def _as_utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _parse_utc_timestamp_series(values: pd.Series) -> pd.Series:
    """Parse ISO timestamps plus common epoch integer encodings.

    Pandas treats bare integers as nanoseconds by default. That turns epoch
    seconds like ``1770000000`` into 1970-era timestamps, which can silently make
    same-day nowcast rows look missing. Canonical stores should write ISO strings,
    but this accepts seconds, milliseconds, microseconds, and nanoseconds.
    """
    text_values = values.astype("string").str.strip()
    numeric = pd.to_numeric(text_values, errors="coerce")
    parsed = pd.to_datetime(text_values, errors="coerce", utc=True)
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        parsed.loc[numeric_mask] = [
            _parse_epoch_numeric(float(value))
            for value in numeric.loc[numeric_mask].tolist()
        ]
    return parsed.dt.tz_localize(None)


def _parse_epoch_numeric(value: float) -> pd.Timestamp:
    magnitude = abs(value)
    if magnitude < 1e11:
        unit = "s"
    elif magnitude < 1e14:
        unit = "ms"
    elif magnitude < 1e17:
        unit = "us"
    else:
        unit = "ns"
    return pd.to_datetime(value, unit=unit, utc=True)


def _local_time(as_of_utc: datetime, rule: StationRule) -> datetime:
    aware = as_of_utc.replace(tzinfo=UTC)
    return aware.astimezone(ZoneInfo(rule.timezone))


def _solar_features(local: datetime) -> dict[str, float]:
    hour = local.hour + local.minute / 60 + local.second / 3600
    return {
        "hours_since_sunrise": hour - 6.0,
        "hours_to_solar_noon": 12.0 - hour,
        "hours_to_sunset": 18.0 - hour,
    }


def _slope(station_obs: pd.DataFrame, latest_obs_ts: datetime, *, hours: int) -> object:
    prior = station_obs[station_obs["obs_ts_utc"] <= latest_obs_ts - timedelta(hours=hours)]
    prior = prior.dropna(subset=["temperature_f"]).sort_values("obs_ts_utc")
    latest = station_obs[station_obs["obs_ts_utc"] == latest_obs_ts]["temperature_f"].dropna()
    if prior.empty or latest.empty:
        return pd.NA
    return float(latest.iloc[-1]) - float(prior.iloc[-1]["temperature_f"])


def _remaining_heating(local: datetime, one_hour_slope: object) -> object:
    hour = local.hour + local.minute / 60
    heating_hours = max(0.0, min(15.0, 15.0 - hour))
    slope = float(one_hour_slope) if not pd.isna(one_hour_slope) else 1.0
    return max(0.0, min(8.0, heating_hours * max(0.0, slope) * 0.6))


def _remaining_cooling(local: datetime, wind: object, cloud_cover: object) -> object:
    hour = local.hour + local.minute / 60
    if 6 <= hour <= 18:
        return 0.0
    base = 4.0 if hour >= 18 else 1.5
    wind_value = float(wind) if not pd.isna(wind) else 8.0
    cloud = str(cloud_cover or "").upper()
    if "OVC" in cloud or "BKN" in cloud:
        base *= 0.5
    if wind_value <= 4:
        base *= 1.25
    return round(base, 3)


def _radiative_cooling_index(wind: object, cloud_cover: object, dewpoint_depression: object) -> object:
    if pd.isna(wind) or pd.isna(dewpoint_depression):
        return pd.NA
    cloud = str(cloud_cover or "").upper()
    cloud_penalty = 0.5 if "OVC" in cloud or "BKN" in cloud else 1.0
    wind_score = max(0.0, 1.0 - min(float(wind), 15.0) / 15.0)
    dry_score = min(max(float(dewpoint_depression), 0.0), 25.0) / 25.0
    return round((0.6 * wind_score + 0.4 * dry_score) * cloud_penalty, 3)


def _number_or_na(value: object) -> object:
    return pd.NA if pd.isna(value) else float(value)


def _expected_min_temp_obs(
    as_of_utc: datetime,
    rule: StationRule,
    target_date: date,
) -> int:
    local = _local_time(as_of_utc, rule)
    if local.date() != target_date:
        return 1
    hours_elapsed = max(0, local.hour + (1 if local.minute >= 30 else 0))
    return max(1, min(24, hours_elapsed // 2))


def _coverage_summary(coverage: pd.DataFrame) -> dict[str, object]:
    if coverage.empty:
        return {
            "rows": 0,
            "coverage_ok_rows": 0,
            "thin_or_missing_rows": 0,
        }
    ok = coverage["coverage_ok"].astype(bool)
    return {
        "rows": int(len(coverage)),
        "coverage_ok_rows": int(ok.sum()),
        "thin_or_missing_rows": int((~ok).sum()),
    }


def _reconcile_feature_coverage(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Mark coverage weak if feature extrema exceed the coverage obs extrema."""
    if features.empty or coverage.empty:
        return coverage
    output = coverage.copy()
    key_cols = [
        "city",
        "platform",
        "market_type",
        "station_id",
        "target_date",
        "decision_time_label",
    ]
    feature_map = {
        tuple(str(row[col]) for col in key_cols): row
        for row in features.to_dict(orient="records")
    }
    for index, row in output.iterrows():
        feature = feature_map.get(tuple(str(row[col]) for col in key_cols))
        if feature is None:
            continue
        reasons = [
            reason
            for reason in str(row.get("coverage_reason_codes") or "").split(";")
            if reason
        ]
        feature_high = _number_or_none(feature.get("high_so_far_f"))
        coverage_high = _number_or_none(row.get("high_so_far_f"))
        if (
            feature_high is not None
            and coverage_high is not None
            and feature_high > coverage_high + 1e-9
        ):
            reasons.append("feature_high_so_far_exceeds_observed_max")
        feature_low = _number_or_none(feature.get("low_so_far_f"))
        coverage_low = _number_or_none(row.get("low_so_far_f"))
        if (
            feature_low is not None
            and coverage_low is not None
            and feature_low < coverage_low - 1e-9
        ):
            reasons.append("feature_low_so_far_below_observed_min")
        if reasons:
            output.at[index, "coverage_ok"] = False
            output.at[index, "coverage_reason_codes"] = ";".join(dict.fromkeys(reasons))
    return output


def _number_or_none(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _feature_hash(*values: object) -> str:
    import hashlib

    text = "|".join("" if pd.isna(value) else str(value) for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _max_obs_ts(features: pd.DataFrame) -> str | None:
    if features.empty or "latest_obs_ts_utc" not in features:
        return None
    values = pd.to_datetime(features["latest_obs_ts_utc"], errors="coerce").dropna()
    if values.empty:
        return None
    return values.max().isoformat()


def _fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)

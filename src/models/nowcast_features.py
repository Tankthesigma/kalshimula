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


@dataclass(frozen=True)
class NowcastFeatureResult:
    observations: pd.DataFrame
    features: pd.DataFrame
    report: str
    manifest: dict[str, object]


def observations_to_frame(observations: list[AsosHourlyObservation]) -> pd.DataFrame:
    """Convert parsed ASOS observations to the canonical observation store shape."""
    rows = []
    for obs in observations:
        obs_ts = _as_utc_naive(obs.valid_time)
        rows.append(
            {
                "station_id": obs.station,
                "obs_ts_utc": obs_ts.isoformat(),
                "available_ts_utc": obs_ts.isoformat(),
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
        if obs.empty:
            station_obs = obs
        else:
            station_obs = obs[
                (obs["station_id"] == rule.settlement_station)
                & (obs["obs_ts_utc"].dt.date == target_date)
            ].sort_values("obs_ts_utc")
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
    fetch_live: bool = False,
    git_commit: str | None = None,
) -> NowcastFeatureResult:
    """Build and write nowcast feature artifacts."""
    rules = _filter_rules(load_station_rules(station_rules_path), market_types=market_types)
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
        "as_of_ts_utc": _as_utc_naive(as_of_ts).isoformat(),
        "decision_time_label": decision_time_label,
        "no_leak_max_observation_ts": _max_obs_ts(features),
        "row_counts": {
            "observations": int(len(observations)),
            "features": int(len(features)),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output_dir / "asos_observations.csv", index=False)
    features.to_csv(output_dir / "nowcast_features.csv", index=False)
    (output_dir / "nowcast_features_report.md").write_text(report, encoding="utf-8")
    (output_dir / "nowcast_features_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NowcastFeatureResult(
        observations=observations,
        features=features,
        report=report,
        manifest=manifest,
    )


def _filter_rules(
    rules: list[StationRule], *, market_types: list[str] | None
) -> list[StationRule]:
    if market_types is None:
        market_type_set = {"high"}
    else:
        market_type_set = {market_type.strip().lower() for market_type in market_types}
    return [rule for rule in rules if rule.market_type in market_type_set]


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
        "prediction_ts_utc": as_of_utc.isoformat(),
        "prediction_time_local": local.isoformat(),
        "decision_time_label": decision_time_label,
        "as_of_ts_utc": as_of_utc.isoformat(),
        "latest_obs_ts_utc": latest_obs_ts.isoformat(),
        "latest_temp_f": latest_temp,
        "latest_dewpoint_f": dewpoint,
        "high_so_far_f": high_so_far,
        "low_so_far_f": low_so_far,
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
        "prediction_ts_utc": as_of_utc.isoformat(),
        "prediction_time_local": local.isoformat(),
        "decision_time_label": decision_time_label,
        "as_of_ts_utc": as_of_utc.isoformat(),
        "latest_obs_ts_utc": pd.NA,
        "latest_temp_f": pd.NA,
        "latest_dewpoint_f": pd.NA,
        "high_so_far_f": pd.NA,
        "low_so_far_f": pd.NA,
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
    obs["obs_ts_utc"] = pd.to_datetime(obs["obs_ts_utc"], errors="coerce").dt.tz_localize(None)
    obs["available_ts_utc"] = pd.to_datetime(
        obs["available_ts_utc"], errors="coerce"
    ).dt.tz_localize(None)
    for col in ("temperature_f", "dewpoint_f", "wind_speed_kt", "wind_direction_deg", "gust_kt", "pressure_mb", "precip_in"):
        if col in obs.columns:
            obs[col] = pd.to_numeric(obs[col], errors="coerce")
    return obs.dropna(subset=["station_id", "obs_ts_utc", "available_ts_utc"])


def _as_utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


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

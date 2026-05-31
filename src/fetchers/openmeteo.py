"""Open-Meteo fetchers.

Three host endpoints we hit:

* Ensemble API (ensemble-api.open-meteo.com) — multi-member ensembles, one row
  per day, columns `temperature_2m_max`, `temperature_2m_max_member01`, ...
* Standard Forecast API (api.open-meteo.com) — single deterministic series per
  call, used for AIFS, GraphCast, HRRR.
* Historical Forecast API (historical-forecast-api.open-meteo.com) — same shape
  as the forecast API but returns model runs that were issued in the past.

Everything is in Fahrenheit by force (`temperature_unit=fahrenheit`). Daily
aggregation is whatever Open-Meteo returns for the station's local timezone;
LST-correct re-aggregation happens later in `features.py`. For Milestone A
that's fine — we only need today/tomorrow.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Final

import httpx
import pandas as pd

from src.cache import JsonCache

ENSEMBLE_URL: Final = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_URL: Final = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL: Final = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)
MAX_ATTEMPTS: Final = 5
MAX_RETRY_DELAY_SECONDS: Final = 300.0
DEFAULT_RATE_LIMIT_DELAY_SECONDS: Final = 60.0
REQUEST_SPACING_SECONDS: Final = 0.1
HISTORICAL_REQUEST_SPACING_SECONDS: Final = 0.5

# Each entry: (slug, open-meteo `models=` value, endpoint, kind)
# kind ∈ {"ensemble", "deterministic"} controls how we parse the response.
SOURCES: Final[list[tuple[str, str, str, str]]] = [
    ("gfs_ens", "gfs_seamless", ENSEMBLE_URL, "ensemble"),
    ("ecmwf_ens", "ecmwf_ifs025", ENSEMBLE_URL, "ensemble"),
    ("icon_ens", "icon_seamless", ENSEMBLE_URL, "ensemble"),
    ("gem_ens", "gem_global", ENSEMBLE_URL, "ensemble"),
    ("aifs", "ecmwf_aifs025_single", FORECAST_URL, "deterministic"),
    ("graphcast", "gfs_graphcast025", FORECAST_URL, "deterministic"),
    ("hrrr", "gfs_hrrr", FORECAST_URL, "deterministic"),
]
_rate_limit_lock = threading.Lock()
_next_request_at = 0.0


@dataclass(frozen=True)
class ModelDailyHigh:
    """Daily-high forecasts from a single Open-Meteo model.

    Members for an ensemble; a one-element list for a deterministic model.
    """

    source: str
    target_date: date
    members_f: list[float]

    @property
    def n_members(self) -> int:
        return len(self.members_f)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError | httpx.TimeoutException)


def _get(url: str, params: dict) -> dict:
    last_error: httpx.HTTPError | None = None
    with httpx.Client(timeout=30.0) as client:
        for attempt in range(MAX_ATTEMPTS):
            _wait_for_request_slot(url)
            try:
                response = client.get(url, params=params)
                if response.is_success:
                    return response.json()
                response.raise_for_status()
            except httpx.HTTPError as error:
                last_error = error
                if (
                    _is_daily_limit_error(error)
                    or not _is_retryable(error)
                    or attempt == MAX_ATTEMPTS - 1
                ):
                    raise
                delay = _retry_delay_seconds(error, attempt)
                if _is_rate_limit_error(error):
                    _set_global_cooldown(delay)
                _sleep(delay)
        if last_error is not None:
            raise last_error
    raise RuntimeError("Open-Meteo request failed without an exception")


def _wait_for_request_slot(url: str) -> None:
    spacing = (
        HISTORICAL_REQUEST_SPACING_SECONDS
        if url == HISTORICAL_FORECAST_URL
        else REQUEST_SPACING_SECONDS
    )
    with _rate_limit_lock:
        global _next_request_at
        now = _monotonic()
        wait_seconds = max(_next_request_at - now, 0.0)
        _next_request_at = max(now, _next_request_at) + spacing
    if wait_seconds > 0:
        _sleep(wait_seconds)


def _set_global_cooldown(delay_seconds: float) -> None:
    with _rate_limit_lock:
        global _next_request_at
        _next_request_at = max(_next_request_at, _monotonic() + delay_seconds)


def _retry_delay_seconds(error: httpx.HTTPError, attempt: int) -> float:
    if _is_rate_limit_error(error):
        return _retry_after_seconds(error) or DEFAULT_RATE_LIMIT_DELAY_SECONDS
    return min(2.0**attempt, 30.0)


def _retry_after_seconds(error: httpx.HTTPError) -> float | None:
    response = getattr(error, "response", None)
    if response is None:
        return None
    raw_value = response.headers.get("retry-after")
    if raw_value is None:
        return None
    try:
        return min(max(float(raw_value), 0.0), MAX_RETRY_DELAY_SECONDS)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(raw_value)
    except (TypeError, ValueError):
        return None
    delay = retry_at.timestamp() - time.time()
    return min(max(delay, 0.0), MAX_RETRY_DELAY_SECONDS)


def _is_rate_limit_error(error: httpx.HTTPError) -> bool:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 429


def _is_daily_limit_error(error: httpx.HTTPError) -> bool:
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) != 429:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    reason = str(payload.get("reason", "")).lower()
    return "daily api request limit exceeded" in reason


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _monotonic() -> float:
    return time.monotonic()


def _response_cache() -> JsonCache:
    root = Path(
        os.environ.get(
            "OPENMETEO_RESPONSE_CACHE_DIR",
            str(Path(gettempdir()) / "kalshimula-openmeteo-response-cache"),
        )
    )
    return JsonCache(root)


def _cache_params(url: str, params: dict) -> dict[str, object]:
    return {"url": url, **params}


def _cached_payload(url: str, params: dict) -> dict | None:
    payload = _response_cache().get("openmeteo_response", _cache_params(url, params))
    return payload if isinstance(payload, dict) else None


def _store_cached_payload(url: str, params: dict, payload: dict) -> None:
    _response_cache().set("openmeteo_response", _cache_params(url, params), payload)


def _common_params(lat: float, lon: float, target: date) -> dict:
    return _range_params(lat, lon, start=target, end=target)


def _range_params(lat: float, lon: float, *, start: date, end: date) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }


def _parse_ensemble(
    payload: dict, target: date, source: str
) -> ModelDailyHigh:
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    if not times:
        return ModelDailyHigh(source=source, target_date=target, members_f=[])
    try:
        idx = times.index(target.isoformat())
    except ValueError:
        idx = 0
    members: list[float] = []
    base = daily.get("temperature_2m_max")
    if base is not None and idx < len(base) and base[idx] is not None:
        members.append(float(base[idx]))
    for key, values in daily.items():
        if key.startswith("temperature_2m_max_member") and idx < len(values):
            v = values[idx]
            if v is not None:
                members.append(float(v))
    return ModelDailyHigh(source=source, target_date=target, members_f=members)


def _parse_ensemble_range(payload: dict, source: str) -> list[ModelDailyHigh]:
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    rows = []
    for idx, raw_time in enumerate(times):
        target = date.fromisoformat(raw_time)
        members: list[float] = []
        base = daily.get("temperature_2m_max")
        if base is not None and idx < len(base) and base[idx] is not None:
            members.append(float(base[idx]))
        for key, values in daily.items():
            if key.startswith("temperature_2m_max_member") and idx < len(values):
                value = values[idx]
                if value is not None:
                    members.append(float(value))
        rows.append(ModelDailyHigh(source=source, target_date=target, members_f=members))
    return rows


def _parse_deterministic(
    payload: dict, target: date, source: str
) -> ModelDailyHigh:
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    values = daily.get("temperature_2m_max", [])
    if not times or not values:
        return ModelDailyHigh(source=source, target_date=target, members_f=[])
    try:
        idx = times.index(target.isoformat())
    except ValueError:
        idx = 0
    v = values[idx]
    members = [float(v)] if v is not None else []
    return ModelDailyHigh(source=source, target_date=target, members_f=members)


def _parse_deterministic_range(payload: dict, source: str) -> list[ModelDailyHigh]:
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    values = daily.get("temperature_2m_max", [])
    rows = []
    for idx, raw_time in enumerate(times):
        target = date.fromisoformat(raw_time)
        value = values[idx] if idx < len(values) else None
        members = [float(value)] if value is not None else []
        rows.append(ModelDailyHigh(source=source, target_date=target, members_f=members))
    return rows


def fetch_source(
    source_slug: str,
    *,
    lat: float,
    lon: float,
    target: date,
    use_historical: bool = False,
) -> ModelDailyHigh:
    """Fetch one model's daily-high for one (lat, lon, target_date).

    `use_historical=True` routes to the historical-forecast endpoint, which is
    needed when `target` is in the past and we want the forecast that was issued
    *at the time*, not the latest re-run.
    """
    entry = next((s for s in SOURCES if s[0] == source_slug), None)
    if entry is None:
        raise ValueError(f"Unknown source {source_slug!r}")
    _, model_param, default_url, kind = entry

    url = HISTORICAL_FORECAST_URL if use_historical else default_url
    params = _common_params(lat, lon, target) | {"models": model_param}
    try:
        payload = _get(url, params)
        _store_cached_payload(url, params, payload)
    except httpx.HTTPStatusError as e:
        # 400 from a model that doesn't support the date range — treat as empty.
        if e.response.status_code in (400, 404):
            return ModelDailyHigh(source=source_slug, target_date=target, members_f=[])
        cached = _cached_payload(url, params)
        if cached is not None:
            payload = cached
        else:
            raise
    except (httpx.TransportError, httpx.TimeoutException):
        cached = _cached_payload(url, params)
        if cached is not None:
            payload = cached
        else:
            raise

    if kind == "ensemble":
        return _parse_ensemble(payload, target, source_slug)
    return _parse_deterministic(payload, target, source_slug)


def fetch_source_range(
    source_slug: str,
    *,
    lat: float,
    lon: float,
    start: date,
    end: date,
    use_historical: bool = False,
) -> list[ModelDailyHigh]:
    """Fetch one model's daily-high forecasts for a date range."""
    entry = next((s for s in SOURCES if s[0] == source_slug), None)
    if entry is None:
        raise ValueError(f"Unknown source {source_slug!r}")
    _, model_param, default_url, kind = entry

    url = HISTORICAL_FORECAST_URL if use_historical else default_url
    params = _range_params(lat, lon, start=start, end=end) | {"models": model_param}
    try:
        payload = _get(url, params)
        _store_cached_payload(url, params, payload)
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (400, 404):
            return [
                ModelDailyHigh(source=source_slug, target_date=target, members_f=[])
                for target in _date_range(start, end)
            ]
        cached = _cached_payload(url, params)
        if cached is not None:
            payload = cached
        else:
            raise
    except (httpx.TransportError, httpx.TimeoutException):
        cached = _cached_payload(url, params)
        if cached is not None:
            payload = cached
        else:
            raise

    if kind == "ensemble":
        return _parse_ensemble_range(payload, source_slug)
    return _parse_deterministic_range(payload, source_slug)


def fetch_all_sources(
    *, lat: float, lon: float, target: date
) -> list[ModelDailyHigh]:
    """Fetch every Open-Meteo source for a single (lat, lon, target_date).

    For Milestone A we hit the live forecast endpoints. Historical-forecast
    routing is added later when the target is more than a few days in the past.
    """
    use_historical = target < date.today() - timedelta(days=2)
    out: list[ModelDailyHigh] = []
    for slug, *_ in SOURCES:
        out.append(
            fetch_source(
                slug, lat=lat, lon=lon, target=target, use_historical=use_historical
            )
        )
    return out


def members_dataframe(sources: list[ModelDailyHigh]) -> pd.DataFrame:
    """Long-format member table across all sources. One row per ensemble member.

    Columns: `source`, `target_date`, `temp_f`.
    """
    rows = []
    for s in sources:
        for v in s.members_f:
            rows.append(
                {"source": s.source, "target_date": s.target_date, "temp_f": v}
            )
    return pd.DataFrame(rows, columns=["source", "target_date", "temp_f"])


def _date_range(start: date, end: date) -> list[date]:
    days = (end - start).days
    if days < 0:
        return []
    return [start + timedelta(days=offset) for offset in range(days + 1)]

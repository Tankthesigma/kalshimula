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

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

ENSEMBLE_URL: Final = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_URL: Final = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL: Final = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)

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


@retry(
    retry=retry_if_exception_type((httpx.HTTPError,)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.0, min=1, max=30),
    reraise=True,
)
def _get(url: str, params: dict) -> dict:
    with httpx.Client(timeout=30.0) as client:
        r = client.get(url, params=params)
        if not r.is_success and not _is_retryable(
            httpx.HTTPStatusError("status", request=r.request, response=r)
        ):
            r.raise_for_status()
        r.raise_for_status()
        return r.json()


def _common_params(lat: float, lon: float, target: date) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
        "start_date": target.isoformat(),
        "end_date": target.isoformat(),
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
    except httpx.HTTPStatusError as e:
        # 400 from a model that doesn't support the date range — treat as empty.
        if e.response.status_code in (400, 404):
            return ModelDailyHigh(source=source_slug, target_date=target, members_f=[])
        raise

    if kind == "ensemble":
        return _parse_ensemble(payload, target, source_slug)
    return _parse_deterministic(payload, target, source_slug)


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

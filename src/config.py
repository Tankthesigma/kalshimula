"""Load station config + environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Station:
    slug: str
    name: str
    nws_station: str
    ghcnd_id: str
    lat: float
    lon: float
    tz: str
    lst_offset_hours: int

    @property
    def ghcnd_bare(self) -> str:
        """Station id without the `GHCND:` prefix, for Access Data Service."""
        return self.ghcnd_id.removeprefix("GHCND:")


@lru_cache(maxsize=1)
def load_stations() -> dict[str, Station]:
    path = PROJECT_ROOT / "config" / "stations.yaml"
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {
        slug: Station(slug=slug, **payload) for slug, payload in raw["cities"].items()
    }


def get_station(slug: str) -> Station:
    stations = load_stations()
    if slug not in stations:
        raise KeyError(
            f"Unknown city slug {slug!r}. Known: {sorted(stations.keys())}"
        )
    return stations[slug]


def nws_user_agent() -> str:
    ua = os.environ.get("NWS_USER_AGENT", "").strip()
    if not ua:
        return "WeatherPredictor/1.0 (no-email-configured)"
    return ua

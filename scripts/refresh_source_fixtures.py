"""Refresh the snapshotted live-API fixtures under ``tests/fixtures/``.

Local/manual helper — not invoked by CI. Run when a fixture-pinned test
fails and you suspect the upstream shape changed (vs. the parser
breaking). Re-runs the same captures used in PR #3 (NCEI), PR #7 (NWS),
and PR #8 (POWER) so the embedded ``expected_high_f`` values self-update
to whatever the live API returns *right now*.

Defaults:

- Writes with LF newlines explicitly (Windows captures don't drift to CRLF).
- Targets three representative cities per source — denver / nyc / miami —
  matching the existing fixture set.
- Replaces the current dated NWS/POWER fixture pool instead of appending a
  new set every time it runs.
- Uses the canonical NCEI historical dates already asserted by the NCEI
  fixture tests; NWS uses today (forecast only), and POWER uses 10 days ago
  (~3-5 day lag).

Run from repo root::

    python -m scripts.refresh_source_fixtures
    # or: python scripts/refresh_source_fixtures.py

Prints one ``ok`` line per fixture written and exits 0 on success. Does
not commit or stage anything — review the diff with ``git status`` /
``git diff`` and decide whether to commit the refresh.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Allow both ``python scripts/refresh_source_fixtures.py`` and
# ``python -m scripts.refresh_source_fixtures`` to resolve ``src.*``.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

from src.config import get_station  # noqa: E402
from src.fetchers.ncei import NCEI_DATA_URL  # noqa: E402
from src.fetchers.ncei import parse_daily_high as parse_ncei  # noqa: E402
from src.fetchers.nws import (  # noqa: E402
    NWS_POINTS_URL,
    forecast_url_from_points_payload,
    parse_daily_high_forecast,
)
from src.fetchers.power import POWER_DAILY_URL  # noqa: E402
from src.fetchers.power import parse_daily_high as parse_power  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# Representative spread — cold winter, mild, tropical.
CITIES = ("denver", "nyc", "miami")
NCEI_CASES: tuple[tuple[str, date], ...] = (
    ("nyc", date(2025, 1, 1)),
    ("denver", date(2025, 7, 15)),
    ("miami", date(2025, 1, 1)),
    # Mirror the chicago negative-temperature edge case for NCEI on a fixed
    # winter day so the parser keeps proving it handles negative Celsius.
    ("chicago", date(2025, 1, 15)),
    # Empty-result fixture: a far-future date NCEI has no data for.
    ("nyc", date(2030, 1, 1)),
)


def _write_lf_json(path: Path, payload: dict | list) -> None:
    """Write JSON with LF newlines + trailing newline, indent 2, sorted keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _ncei_filename(slug: str, target: date, empty: bool = False) -> str:
    # NCEI fixture convention uses underscores in the date (matching the
    # files committed in PR #3). NWS/POWER fixtures use ISO hyphens; that
    # split is intentional — don't unify without renaming the existing set.
    stem = f"ncei_{slug}_{target.strftime('%Y_%m_%d')}"
    return f"{stem}_empty.json" if empty else f"{stem}.json"


def _clear_fixtures(pattern: str) -> None:
    for path in FIXTURES_DIR.glob(pattern):
        path.unlink()


def _refresh_ncei(client: httpx.Client) -> None:
    for slug, target in NCEI_CASES:
        is_empty = (slug, target) == ("nyc", date(2030, 1, 1))
        name = _ncei_filename(slug, target, empty=is_empty)
        station = get_station(slug)
        params = {
            "dataset": "daily-summaries",
            "stations": station.ghcnd_bare,
            "dataTypes": "TMAX",
            "startDate": target.isoformat(),
            "endDate": target.isoformat(),
            "format": "json",
            "units": "metric",
        }
        response = client.get(NCEI_DATA_URL, params=params)
        response.raise_for_status()
        body = response.json()
        _write_lf_json(FIXTURES_DIR / name, body)
        print(
            f"ok  ncei  {name}  rows={len(body) if isinstance(body, list) else '-'}"
        )


def _refresh_nws(client: httpx.Client) -> None:
    today = date.today()
    _clear_fixtures("nws_*.json")
    headers = {
        "User-Agent": "weather-predictor-fixtures/1.0 (refresh)",
        "Accept": "application/geo+json",
    }
    for slug in CITIES:
        station = get_station(slug)
        points_response = client.get(
            NWS_POINTS_URL.format(lat=station.lat, lon=station.lon),
            headers=headers,
        )
        points_response.raise_for_status()
        points_payload = points_response.json()
        # Trim to the fields the parser uses + a few human-readable extras.
        points_trim = {
            "properties": {
                key: points_payload.get("properties", {}).get(key)
                for key in (
                    "forecast",
                    "forecastHourly",
                    "gridId",
                    "gridX",
                    "gridY",
                    "timeZone",
                )
            }
        }
        forecast_url = forecast_url_from_points_payload(points_payload)
        forecast_response = client.get(forecast_url, headers=headers)
        forecast_response.raise_for_status()
        forecast_payload = forecast_response.json()
        parsed = parse_daily_high_forecast(forecast_payload, today, station.nws_station)
        fixture = {
            "captured_for_slug": slug,
            "captured_for_station": station.nws_station,
            "captured_target_date": today.isoformat(),
            "expected_high_f": parsed.high_f,
            "points_response": points_trim,
            "forecast_response": forecast_payload,
        }
        path = FIXTURES_DIR / f"nws_{slug}_{today.isoformat()}.json"
        _write_lf_json(path, fixture)
        print(f"ok  nws   {path.name}  high_f={parsed.high_f}")


def _refresh_power(client: httpx.Client) -> None:
    target = date.today() - timedelta(days=10)
    _clear_fixtures("power_*.json")
    for slug in CITIES:
        station = get_station(slug)
        params = {
            "parameters": "T2M_MAX",
            "community": "RE",
            "longitude": station.lon,
            "latitude": station.lat,
            "start": target.strftime("%Y%m%d"),
            "end": target.strftime("%Y%m%d"),
            "format": "JSON",
        }
        response = client.get(POWER_DAILY_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        parsed = parse_power(payload, target, station.nws_station)
        fixture = {
            "captured_for_slug": slug,
            "captured_for_station": station.nws_station,
            "captured_target_date": target.isoformat(),
            "expected_high_f": parsed.high_f,
            "payload": payload,
        }
        path = FIXTURES_DIR / f"power_{slug}_{target.isoformat()}.json"
        _write_lf_json(path, fixture)
        print(f"ok  power {path.name}  high_f={parsed.high_f}")


def _confirm_parser_round_trip() -> None:
    """Quick parse on each refreshed NCEI fixture with its real target/station."""
    for slug, target in NCEI_CASES:
        station = get_station(slug)
        is_empty = (slug, target) == ("nyc", date(2030, 1, 1))
        path = FIXTURES_DIR / _ncei_filename(slug, target, empty=is_empty)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            parse_ncei(payload, target, station.ghcnd_bare)


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30.0) as client:
        _refresh_ncei(client)
        _refresh_nws(client)
        _refresh_power(client)
    _confirm_parser_round_trip()
    print(
        "\ndone. review changes with `git status` / `git diff tests/fixtures/` "
        "then commit if the refresh looks healthy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

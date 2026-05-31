from __future__ import annotations

from datetime import date

import httpx
import pytest

from src.fetchers import openmeteo


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, params):
        self.calls += 1
        response = self.responses.pop(0)
        response.request = httpx.Request("GET", url)
        return response


def _reset_rate_limit(monkeypatch) -> list[float]:
    now = [0.0]
    sleeps = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(openmeteo, "_next_request_at", 0.0)
    monkeypatch.setattr(openmeteo, "_monotonic", lambda: now[0])
    monkeypatch.setattr(openmeteo, "_sleep", fake_sleep)
    return sleeps


def test_get_respects_retry_after_on_rate_limit(monkeypatch) -> None:
    sleeps = _reset_rate_limit(monkeypatch)
    client = FakeClient(
        [
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    monkeypatch.setattr(openmeteo.httpx, "Client", lambda timeout: client)

    payload = openmeteo._get(openmeteo.FORECAST_URL, {"models": "gfs"})

    assert payload == {"ok": True}
    assert client.calls == 2
    assert sleeps == [2.0]


def test_get_does_not_retry_daily_request_limit(monkeypatch) -> None:
    sleeps = _reset_rate_limit(monkeypatch)
    client = FakeClient(
        [
            httpx.Response(
                429,
                json={
                    "reason": "Daily API request limit exceeded. Please try again tomorrow.",
                    "error": True,
                },
            )
        ]
    )
    monkeypatch.setattr(openmeteo.httpx, "Client", lambda timeout: client)

    with pytest.raises(httpx.HTTPStatusError):
        openmeteo._get(openmeteo.HISTORICAL_FORECAST_URL, {"models": "gfs"})

    assert client.calls == 1
    assert sleeps == []


def test_get_does_not_retry_non_retryable_status(monkeypatch) -> None:
    sleeps = _reset_rate_limit(monkeypatch)
    client = FakeClient([httpx.Response(400, json={"reason": "bad request"})])
    monkeypatch.setattr(openmeteo.httpx, "Client", lambda timeout: client)

    with pytest.raises(httpx.HTTPStatusError):
        openmeteo._get(openmeteo.FORECAST_URL, {"models": "bad"})

    assert client.calls == 1
    assert sleeps == []


def test_retry_after_seconds_accepts_http_date(monkeypatch) -> None:
    monkeypatch.setattr(openmeteo.time, "time", lambda: 1000.0)
    response = httpx.Response(
        429,
        headers={"Retry-After": "Thu, 01 Jan 1970 00:20:00 GMT"},
        request=httpx.Request("GET", openmeteo.FORECAST_URL),
    )
    error = httpx.HTTPStatusError("rate limited", request=response.request, response=response)

    assert openmeteo._retry_after_seconds(error) == 200.0


def test_fetch_source_range_parses_ensemble_payload(monkeypatch) -> None:
    payload = {
        "daily": {
            "time": ["2025-01-01", "2025-01-02"],
            "temperature_2m_max": [70.0, 71.0],
            "temperature_2m_max_member01": [69.0, 72.0],
            "temperature_2m_max_member02": [None, 73.0],
        }
    }
    monkeypatch.setattr(openmeteo, "_get", lambda url, params: payload)

    rows = openmeteo.fetch_source_range(
        "gfs_ens",
        lat=1,
        lon=2,
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        use_historical=True,
    )

    assert rows == [
        openmeteo.ModelDailyHigh(
            source="gfs_ens", target_date=date(2025, 1, 1), members_f=[70.0, 69.0]
        ),
        openmeteo.ModelDailyHigh(
            source="gfs_ens",
            target_date=date(2025, 1, 2),
            members_f=[71.0, 72.0, 73.0],
        ),
    ]


def test_fetch_source_range_treats_unsupported_range_as_empty(monkeypatch) -> None:
    request = httpx.Request("GET", openmeteo.FORECAST_URL)
    response = httpx.Response(400, request=request)

    def fake_get(url, params):
        raise httpx.HTTPStatusError("bad range", request=request, response=response)

    monkeypatch.setattr(openmeteo, "_get", fake_get)

    rows = openmeteo.fetch_source_range(
        "aifs",
        lat=1,
        lon=2,
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
    )

    assert rows == [
        openmeteo.ModelDailyHigh(source="aifs", target_date=date(2025, 1, 1), members_f=[]),
        openmeteo.ModelDailyHigh(source="aifs", target_date=date(2025, 1, 2), members_f=[]),
    ]


def test_fetch_source_uses_cached_payload_after_rate_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENMETEO_RESPONSE_CACHE_DIR", str(tmp_path))

    payload = {
        "daily": {
            "time": ["2025-01-01"],
            "temperature_2m_max": [70.0],
            "temperature_2m_max_member01": [69.0],
        }
    }

    monkeypatch.setattr(openmeteo, "_get", lambda url, params: payload)
    first = openmeteo.fetch_source("gfs_ens", lat=1, lon=2, target=date(2025, 1, 1))
    assert first.members_f == [70.0, 69.0]

    request = httpx.Request("GET", openmeteo.ENSEMBLE_URL)
    response = httpx.Response(429, request=request)

    def rate_limited(url, params):
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr(openmeteo, "_get", rate_limited)
    second = openmeteo.fetch_source("gfs_ens", lat=1, lon=2, target=date(2025, 1, 1))
    assert second.members_f == [70.0, 69.0]

"""Tests for src.fetchers.ncei — parser- and fetcher-focused, fully offline."""

from __future__ import annotations

from datetime import date

import pytest

from src.config import Station
from src.fetchers import ncei
from src.fetchers.ncei import NceiDailyHigh, fetch_daily_high, parse_daily_high

STATION = "USW00094728"
TARGET = date(2025, 1, 2)


def _row(
    *,
    date_str: str = "2025-01-02",
    datatype: str = "TMAX",
    value: object = 100,
    station: str = STATION,
) -> dict:
    return {
        "date": date_str,
        "datatype": datatype,
        "value": value,
        "station": station,
    }


def _payload(*rows: dict) -> dict:
    return {"results": list(rows)}


def _station() -> Station:
    return Station(
        slug="nyc",
        name="NYC",
        nws_station="KNYC",
        ghcnd_id="GHCND:USW00094728",
        lat=40.78,
        lon=-73.97,
        tz="America/New_York",
        lst_offset_hours=-5,
    )


class TestParseDailyHigh:
    def test_matching_tmax_converts_tenths_celsius_to_fahrenheit(self):
        # 100 tenths C = 10 C = 50 F
        result = parse_daily_high(_payload(_row(value=100)), TARGET, STATION)
        assert isinstance(result, NceiDailyHigh)
        assert result.station == STATION
        assert result.target_date == TARGET
        assert result.source == "ncei"
        assert result.high_f == pytest.approx(50.0)

    def test_lowercase_datatype_match(self):
        result = parse_daily_high(_payload(_row(datatype="tmax", value=100)), TARGET, STATION)
        assert result.high_f == pytest.approx(50.0)

    def test_mixed_case_datatype_match(self):
        result = parse_daily_high(_payload(_row(datatype="TMax", value=100)), TARGET, STATION)
        assert result.high_f == pytest.approx(50.0)

    def test_non_tmax_rows_ignored(self):
        result = parse_daily_high(
            _payload(
                _row(datatype="TMIN", value=10),
                _row(datatype="PRCP", value=0),
            ),
            TARGET,
            STATION,
        )
        assert result.high_f is None

    def test_multiple_matching_tmax_returns_max(self):
        result = parse_daily_high(
            _payload(
                _row(value=100),  # 50 F
                _row(value=156),  # 60.08 F
                _row(value=120),  # 53.6 F
            ),
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(60.08)

    def test_missing_results_returns_none(self):
        assert parse_daily_high({}, TARGET, STATION).high_f is None
        assert parse_daily_high({"results": []}, TARGET, STATION).high_f is None

    def test_results_not_a_list_returns_none(self):
        assert parse_daily_high({"results": "oops"}, TARGET, STATION).high_f is None

    def test_bad_or_missing_values_ignored(self):
        result = parse_daily_high(
            _payload(
                _row(value=None),
                _row(value=""),
                _row(value="M"),
                _row(value="abc"),
            ),
            TARGET,
            STATION,
        )
        assert result.high_f is None

    def test_datetime_style_date_strings(self):
        result = parse_daily_high(
            _payload(_row(date_str="2025-01-02T00:00:00", value=200)),
            TARGET,
            STATION,
        )
        # 200 tenths C = 20 C = 68 F
        assert result.high_f == pytest.approx(68.0)

    def test_different_date_ignored(self):
        result = parse_daily_high(
            _payload(_row(date_str="2025-01-03", value=200)),
            TARGET,
            STATION,
        )
        assert result.high_f is None

    def test_non_dict_row_skipped(self):
        result = parse_daily_high(
            {"results": ["bad", _row(value=100)]},
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(50.0)

    def test_mixed_good_and_bad_rows(self):
        result = parse_daily_high(
            _payload(
                _row(value=None),
                _row(value=156),  # 60.08 F
                _row(datatype="TMIN", value=200),
            ),
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(60.08)

    def test_raw_list_payload(self):
        # Access Data Service sometimes returns a bare top-level list.
        result = parse_daily_high([_row(value=200)], TARGET, STATION)  # type: ignore[arg-type]
        assert result.high_f == pytest.approx(68.0)

    def test_single_dict_row_payload(self):
        # Older paths occasionally return one row not wrapped in {"results": ...}.
        result = parse_daily_high(
            {"date": "2025-01-02", "datatype": "TMAX", "value": 100},
            TARGET,
            STATION,
        )
        assert result.high_f == pytest.approx(50.0)

    def test_station_field_can_be_missing(self):
        row = {"date": "2025-01-02", "datatype": "TMAX", "value": 100}
        result = parse_daily_high({"results": [row]}, TARGET, STATION)
        assert result.high_f == pytest.approx(50.0)

    def test_non_dict_payload_returns_none(self):
        for bad in (None, "oops", 42):
            result = parse_daily_high(bad, TARGET, STATION)  # type: ignore[arg-type]
            assert result.high_f is None

    def test_non_string_datatype_skipped(self):
        result = parse_daily_high(
            _payload(_row(datatype=None, value=100)),  # type: ignore[arg-type]
            TARGET,
            STATION,
        )
        assert result.high_f is None


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self.calls: list[tuple[str, dict, dict | None]] = []
        self._payload = payload

    def __call__(self, timeout):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, headers=None, params=None):
        self.calls.append((url, params, headers))
        return _FakeResponse(self._payload)


class TestFetchDailyHigh:
    def test_calls_access_data_service_with_expected_params(self, monkeypatch):
        payload = {"results": [_row(value=200)]}
        client = _FakeClient(payload)
        monkeypatch.setattr(ncei.httpx, "Client", client)

        result = fetch_daily_high(_station(), TARGET)

        assert result.high_f == pytest.approx(68.0)
        assert result.station == "USW00094728"
        assert len(client.calls) == 1
        url, params, _ = client.calls[0]
        assert url == "https://www.ncei.noaa.gov/access/services/data/v1"
        assert params["dataset"] == "daily-summaries"
        assert params["stations"] == "USW00094728"
        assert params["dataTypes"] == "TMAX"
        assert params["startDate"] == "2025-01-02"
        assert params["endDate"] == "2025-01-02"
        assert params["format"] == "json"
        assert params["units"] == "metric"

    def test_bare_list_response_works(self, monkeypatch):
        client = _FakeClient([_row(value=100)])
        monkeypatch.setattr(ncei.httpx, "Client", client)
        result = fetch_daily_high(_station(), TARGET)
        assert result.high_f == pytest.approx(50.0)

    def test_http_error_propagates(self, monkeypatch):
        class _BoomResponse:
            def raise_for_status(self):
                raise RuntimeError("ncei 503")

            def json(self):
                raise AssertionError

        class _BoomClient:
            def __call__(self, timeout):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, *a, **k):
                return _BoomResponse()

        monkeypatch.setattr(ncei.httpx, "Client", _BoomClient())
        with pytest.raises(RuntimeError, match="ncei 503"):
            fetch_daily_high(_station(), TARGET)

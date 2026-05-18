"""Tests for src.smoke_weather_cli — fully offline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src import smoke_weather_cli
from src.smoke_weather import SmokeResult

TARGET = date(2025, 1, 2)


def _ok(city: str, source: str, high_f: float) -> SmokeResult:
    return SmokeResult(
        city=city,
        target_date=TARGET,
        source=source,
        ok=True,
        high_f=high_f,
        error=None,
    )


def _err(city: str, source: str, msg: str) -> SmokeResult:
    return SmokeResult(
        city=city,
        target_date=TARGET,
        source=source,
        ok=False,
        high_f=None,
        error=msg,
    )


class TestParseCities:
    def test_explicit_cities_parsed(self):
        assert smoke_weather_cli._parse_cities("denver,nyc") == ["denver", "nyc"]

    def test_explicit_cities_strip_whitespace(self):
        assert smoke_weather_cli._parse_cities("  denver  , nyc ") == ["denver", "nyc"]

    def test_empty_chunks_dropped(self):
        assert smoke_weather_cli._parse_cities("denver,,nyc,") == ["denver", "nyc"]

    def test_default_uses_load_stations(self, monkeypatch):
        monkeypatch.setattr(
            smoke_weather_cli,
            "load_stations",
            lambda: {"denver": object(), "nyc": object()},
        )
        assert smoke_weather_cli._parse_cities(None) == ["denver", "nyc"]


class TestMain:
    def test_returns_zero_when_all_ok(self, monkeypatch, capsys):
        monkeypatch.setattr(
            smoke_weather_cli,
            "smoke_cities",
            lambda cities, target: [
                _ok("denver", "nws", 75.0),
                _ok("denver", "ncei", 70.0),
                _ok("denver", "power", 72.0),
            ],
        )
        rc = smoke_weather_cli.main(["--cities", "denver", "--date", "2025-01-02"])
        captured = capsys.readouterr().out
        assert rc == 0
        assert "denver" in captured
        assert "nws" in captured

    def test_returns_one_when_any_error(self, monkeypatch):
        monkeypatch.setattr(
            smoke_weather_cli,
            "smoke_cities",
            lambda cities, target: [
                _ok("denver", "nws", 75.0),
                _err("denver", "ncei", "RuntimeError: 503"),
            ],
        )
        rc = smoke_weather_cli.main(["--cities", "denver", "--date", "2025-01-02"])
        assert rc == 1

    def test_writes_csv_when_out_given(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            smoke_weather_cli,
            "smoke_cities",
            lambda cities, target: [_ok("denver", "nws", 75.0)],
        )
        out = tmp_path / "smoke.csv"
        rc = smoke_weather_cli.main(
            ["--cities", "denver", "--date", "2025-01-02", "--out", str(out)]
        )
        assert rc == 0
        assert out.exists()
        df = pd.read_csv(out)
        assert "city" in df.columns and "source" in df.columns
        assert df.iloc[0]["city"] == "denver"
        assert df.iloc[0]["source"] == "nws"

    def test_writes_csv_creates_parent_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            smoke_weather_cli,
            "smoke_cities",
            lambda cities, target: [_ok("denver", "nws", 75.0)],
        )
        out = tmp_path / "nested" / "deeper" / "smoke.csv"
        rc = smoke_weather_cli.main(
            ["--cities", "denver", "--date", "2025-01-02", "--out", str(out)]
        )
        assert rc == 0
        assert out.exists()

    def test_default_cities_uses_load_stations(self, monkeypatch):
        captured: dict[str, list[str]] = {}
        monkeypatch.setattr(
            smoke_weather_cli,
            "load_stations",
            lambda: {"denver": None, "nyc": None, "miami": None},
        )

        def fake(cities, target):
            captured["cities"] = cities
            return [_ok(cities[0], "nws", 75.0)]

        monkeypatch.setattr(smoke_weather_cli, "smoke_cities", fake)
        rc = smoke_weather_cli.main(["--date", "2025-01-02"])
        assert rc == 0
        assert captured["cities"] == ["denver", "miami", "nyc"]

    def test_empty_results_returns_zero(self, monkeypatch):
        monkeypatch.setattr(
            smoke_weather_cli, "smoke_cities", lambda cities, target: []
        )
        rc = smoke_weather_cli.main(["--cities", "denver", "--date", "2025-01-02"])
        # No rows means no failures.
        assert rc == 0

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            smoke_weather_cli.main(["--cities", "denver", "--date", "not-a-date"])

    def test_missing_date_arg_raises(self):
        with pytest.raises(SystemExit):
            smoke_weather_cli.main(["--cities", "denver"])

    def test_does_not_write_csv_when_out_not_given(self, monkeypatch, tmp_path):
        # Negative control: --out is optional.
        monkeypatch.setattr(
            smoke_weather_cli,
            "smoke_cities",
            lambda cities, target: [_ok("denver", "nws", 75.0)],
        )
        before = list(Path(tmp_path).iterdir())
        rc = smoke_weather_cli.main(["--cities", "denver", "--date", "2025-01-02"])
        after = list(Path(tmp_path).iterdir())
        assert rc == 0
        assert before == after

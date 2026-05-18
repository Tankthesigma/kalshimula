"""Tests for src.source_quality — pure DataFrame computations."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.source_quality import (
    REQUIRED_COLUMNS,
    SUMMARY_COLUMNS,
    read_smoke_results,
    summarize_source_quality,
    write_source_quality,
)


def _row(
    *,
    city: str = "denver",
    target_date: str = "2025-01-02",
    source: str,
    ok: bool,
    high_f: float | None = None,
    error: str | None = None,
) -> dict:
    return {
        "city": city,
        "target_date": target_date,
        "source": source,
        "ok": ok,
        "high_f": high_f,
        "error": error,
    }


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))


class TestSummarizeSourceQuality:
    def test_basic_ok_rate(self):
        df = _df(
            [
                _row(source="nws", ok=True, high_f=70.0),
                _row(source="nws", ok=True, high_f=72.0),
                _row(source="nws", ok=False, error="boom"),
                _row(source="ncei", ok=True, high_f=68.0),
            ]
        )
        summary = summarize_source_quality(df)
        assert list(summary.columns) == list(SUMMARY_COLUMNS)
        nws = summary[summary["source"] == "nws"].iloc[0]
        assert int(nws["n"]) == 3
        assert int(nws["ok_count"]) == 2
        assert int(nws["error_count"]) == 1
        assert int(nws["missing_high_count"]) == 0
        assert nws["ok_rate"] == pytest.approx(2 / 3)
        assert nws["missing_high_rate"] == 0.0

        ncei = summary[summary["source"] == "ncei"].iloc[0]
        assert int(ncei["n"]) == 1
        assert int(ncei["ok_count"]) == 1
        assert int(ncei["error_count"]) == 0
        assert ncei["ok_rate"] == 1.0

    def test_missing_high_among_ok_counted(self):
        df = _df(
            [
                _row(source="power", ok=True, high_f=68.0),
                _row(source="power", ok=True, high_f=None),
                _row(source="power", ok=True, high_f=None),
                _row(source="power", ok=False, error="503"),
            ]
        )
        summary = summarize_source_quality(df)
        row = summary[summary["source"] == "power"].iloc[0]
        assert int(row["n"]) == 4
        assert int(row["ok_count"]) == 3
        assert int(row["error_count"]) == 1
        assert int(row["missing_high_count"]) == 2
        assert row["missing_high_rate"] == pytest.approx(2 / 3)

    def test_all_errors_zero_ok_rate(self):
        df = _df(
            [
                _row(source="nws", ok=False, error="boom"),
                _row(source="nws", ok=False, error="boom"),
            ]
        )
        summary = summarize_source_quality(df)
        row = summary.iloc[0]
        assert int(row["ok_count"]) == 0
        assert int(row["error_count"]) == 2
        assert row["ok_rate"] == 0.0
        assert row["missing_high_rate"] == 0.0  # well-defined, not NaN

    def test_empty_dataframe_returns_stable_columns(self):
        df = _df([])
        summary = summarize_source_quality(df)
        assert list(summary.columns) == list(SUMMARY_COLUMNS)
        assert len(summary) == 0

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"city": [], "source": [], "ok": []})
        with pytest.raises(ValueError) as excinfo:
            summarize_source_quality(df)
        assert "missing required columns" in str(excinfo.value)

    def test_sources_returned_sorted(self):
        df = _df(
            [
                _row(source="power", ok=True, high_f=70.0),
                _row(source="ncei", ok=True, high_f=70.0),
                _row(source="nws", ok=True, high_f=70.0),
            ]
        )
        summary = summarize_source_quality(df)
        assert list(summary["source"]) == ["ncei", "nws", "power"]

    def test_nan_high_f_treated_as_missing(self):
        # CSV round-trip converts None to NaN. Both must read as missing.
        df = _df(
            [
                _row(source="power", ok=True, high_f=70.0),
                _row(source="power", ok=True, high_f=float("nan")),
            ]
        )
        summary = summarize_source_quality(df)
        row = summary.iloc[0]
        assert int(row["missing_high_count"]) == 1


class TestReadAndWrite:
    def test_round_trip_through_csv(self, tmp_path):
        df = _df(
            [
                _row(source="nws", ok=True, high_f=70.0),
                _row(source="ncei", ok=False, error="boom"),
            ]
        )
        smoke_path = tmp_path / "smoke.csv"
        out_path = tmp_path / "summary.csv"
        df.to_csv(smoke_path, index=False)

        summary = write_source_quality(smoke_path, out_path)
        assert out_path.exists()
        loaded = pd.read_csv(out_path)
        assert list(loaded.columns) == list(SUMMARY_COLUMNS)
        assert len(loaded) == len(summary)
        assert set(loaded["source"]) == {"nws", "ncei"}

    def test_read_smoke_results(self, tmp_path):
        df = _df([_row(source="nws", ok=True, high_f=70.0)])
        path = tmp_path / "smoke.csv"
        df.to_csv(path, index=False)
        loaded = read_smoke_results(path)
        assert list(loaded.columns) == list(REQUIRED_COLUMNS)
        assert len(loaded) == 1

    def test_write_creates_parent_dir(self, tmp_path):
        df = _df([_row(source="nws", ok=True, high_f=70.0)])
        smoke_path = tmp_path / "smoke.csv"
        out_path = tmp_path / "nested" / "deeper" / "summary.csv"
        df.to_csv(smoke_path, index=False)
        write_source_quality(smoke_path, out_path)
        assert out_path.exists()

    def test_ok_column_can_be_string_after_csv(self, tmp_path):
        # CSV round-trip can render True/False as strings depending on dtype;
        # the summary must still handle this without erroring.
        smoke_path = tmp_path / "smoke.csv"
        smoke_path.write_text(
            "city,target_date,source,ok,high_f,error\n"
            "denver,2025-01-02,nws,True,70.0,\n"
            "denver,2025-01-02,nws,False,,boom\n",
            encoding="utf-8",
        )
        loaded = read_smoke_results(smoke_path)
        summary = summarize_source_quality(loaded)
        row = summary[summary["source"] == "nws"].iloc[0]
        # pandas reads "True"/"False" as bool dtype, so this works cleanly.
        assert int(row["ok_count"]) >= 1


def test_summary_columns_constant_matches_signature():
    # The constant the CLI and tests rely on must not silently drift.
    assert SUMMARY_COLUMNS == (
        "source",
        "n",
        "ok_count",
        "error_count",
        "missing_high_count",
        "ok_rate",
        "missing_high_rate",
    )
    # And the required columns must include every input field.
    assert "ok" in REQUIRED_COLUMNS
    assert "high_f" in REQUIRED_COLUMNS
    assert math.isnan(float("nan"))  # smoke: make sure pytest didn't drop math import

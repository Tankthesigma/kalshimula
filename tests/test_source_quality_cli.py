"""Tests for src.source_quality_cli — fully offline."""

from __future__ import annotations

import pandas as pd
import pytest

from src import source_quality_cli


def _write_smoke_csv(path, rows: list[dict]) -> None:
    df = pd.DataFrame(
        rows,
        columns=["city", "target_date", "source", "ok", "high_f", "error"],
    )
    df.to_csv(path, index=False)


def test_main_writes_summary(tmp_path, capsys):
    smoke = tmp_path / "smoke.csv"
    out = tmp_path / "summary.csv"
    _write_smoke_csv(
        smoke,
        [
            {
                "city": "denver",
                "target_date": "2025-01-02",
                "source": "nws",
                "ok": True,
                "high_f": 70.0,
                "error": "",
            },
            {
                "city": "denver",
                "target_date": "2025-01-02",
                "source": "nws",
                "ok": False,
                "high_f": "",
                "error": "boom",
            },
            {
                "city": "denver",
                "target_date": "2025-01-02",
                "source": "ncei",
                "ok": True,
                "high_f": "",
                "error": "",
            },
        ],
    )

    rc = source_quality_cli.main(["--input", str(smoke), "--out", str(out)])
    assert rc == 0
    assert out.exists()

    captured = capsys.readouterr().out
    assert "summarized" in captured

    summary = pd.read_csv(out)
    assert set(summary.columns) >= {
        "source",
        "n",
        "ok_count",
        "error_count",
        "missing_high_count",
        "ok_rate",
        "missing_high_rate",
    }
    assert set(summary["source"]) == {"nws", "ncei"}


def test_main_creates_output_parent_dir(tmp_path):
    smoke = tmp_path / "smoke.csv"
    out = tmp_path / "nested" / "deeper" / "summary.csv"
    _write_smoke_csv(
        smoke,
        [
            {
                "city": "denver",
                "target_date": "2025-01-02",
                "source": "nws",
                "ok": True,
                "high_f": 70.0,
                "error": "",
            }
        ],
    )
    rc = source_quality_cli.main(["--input", str(smoke), "--out", str(out)])
    assert rc == 0
    assert out.exists()


def test_main_missing_required_arg(tmp_path):
    with pytest.raises(SystemExit):
        source_quality_cli.main(["--input", str(tmp_path / "smoke.csv")])


def test_main_propagates_invalid_input(tmp_path):
    # An input CSV missing required columns must raise from summarize, not be
    # silently swallowed.
    smoke = tmp_path / "bad.csv"
    smoke.write_text("city,source\ndenver,nws\n", encoding="utf-8")
    out = tmp_path / "summary.csv"
    with pytest.raises(ValueError):
        source_quality_cli.main(["--input", str(smoke), "--out", str(out)])

import argparse
from datetime import date

import pandas as pd
import pytest

from src import batch_collect_cli
from src.datasets.backtest import make_backtest_row


def test_parse_cities_trims_and_filters() -> None:
    assert batch_collect_cli._parse_cities("denver, chicago ,,nyc") == [
        "denver",
        "chicago",
        "nyc",
    ]


def test_parse_cities_rejects_empty() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        batch_collect_cli._parse_cities(" , ")


def test_collect_many_cities_combines_rows(monkeypatch, tmp_path) -> None:
    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        assert openmeteo_mode == "sources"
        return type(
            "Result",
            (),
            {
                "rows": [
                    make_backtest_row(
                        city=city,
                        target_date=start,
                        source="openmeteo_naive",
                        point_f=70,
                        actual_high_f=68,
                    )
                ]
            },
        )()

    monkeypatch.setattr(
        batch_collect_cli, "collect_backtest_rows", fake_collect_backtest_rows
    )

    df = batch_collect_cli.collect_many_cities(
        cities=["denver", "chicago"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
        cache_root=tmp_path,
        openmeteo_mode="sources",
    )

    assert list(df["city"]) == ["denver", "chicago"]
    assert list(df["absolute_error_f"]) == [2.0, 2.0]


def test_write_batch_outputs_writes_rows_and_summary(tmp_path) -> None:
    rows_path = tmp_path / "rows.csv"
    summary_path = tmp_path / "summary.csv"
    df = pd.DataFrame(
        [
            {
                "city": "denver",
                "target_date": date(2025, 1, 1),
                "source": "openmeteo_naive",
                "point_f": 70,
                "actual_high_f": 68,
                "absolute_error_f": 2,
            }
        ]
    )

    summary = batch_collect_cli.write_batch_outputs(df, rows_path, summary_path)

    assert rows_path.exists()
    assert summary_path.exists()
    assert summary.iloc[0]["mae"] == 2.0


def test_batch_collect_cli_main(monkeypatch, tmp_path, capsys) -> None:
    rows_path = tmp_path / "rows.csv"
    summary_path = tmp_path / "summary.csv"

    def fake_collect_many_cities(*, cities, start, end, cache_root, openmeteo_mode):
        assert cities == ["denver", "chicago"]
        assert start == date(2025, 1, 1)
        assert end == date(2025, 1, 1)
        assert cache_root == tmp_path / "cache"
        assert openmeteo_mode == "both"
        return pd.DataFrame(
            [
                {
                    "city": "denver",
                    "target_date": date(2025, 1, 1),
                    "source": "openmeteo_naive",
                    "point_f": 70,
                    "actual_high_f": 68,
                    "absolute_error_f": 2,
                }
            ]
        )

    monkeypatch.setattr(batch_collect_cli, "collect_many_cities", fake_collect_many_cities)

    code = batch_collect_cli.main(
        [
            "--cities",
            "denver,chicago",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-01",
            "--rows-out",
            str(rows_path),
            "--summary-out",
            str(summary_path),
            "--cache",
            str(tmp_path / "cache"),
            "--openmeteo-mode",
            "both",
        ]
    )

    assert code == 0
    assert rows_path.exists()
    assert summary_path.exists()
    assert "Wrote 1 rows" in capsys.readouterr().out

from datetime import date

from src import collect_cli
from src.collect import CollectionResult


def test_collect_cli_writes_output(monkeypatch, tmp_path, capsys) -> None:
    out = tmp_path / "rows.csv"

    def fake_collect_backtest_rows(*, city, start, end, cache_root):
        assert city == "denver"
        assert start == date(2025, 1, 1)
        assert end == date(2025, 1, 1)
        assert cache_root == tmp_path / "cache"
        return CollectionResult(city=city, start=start, end=end, rows=[])

    def fake_write_collection_csv(result, path):
        assert result.city == "denver"
        path.write_text("city,target_date,source,point_f,actual_high_f,absolute_error_f\n")

    monkeypatch.setattr(collect_cli, "collect_backtest_rows", fake_collect_backtest_rows)
    monkeypatch.setattr(collect_cli, "write_collection_csv", fake_write_collection_csv)

    code = collect_cli.main(
        [
            "--city",
            "denver",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-01",
            "--out",
            str(out),
            "--cache",
            str(tmp_path / "cache"),
        ]
    )

    assert code == 0
    assert "Wrote 0 rows" in capsys.readouterr().out

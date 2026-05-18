from datetime import date

from src import historical_runner_cli
from src.historical_runner import HistoricalRunResult


def test_historical_runner_cli_calls_pipeline(monkeypatch, tmp_path, capsys) -> None:
    def fake_run_historical_pipeline(*, cities, start, end, test_start, out_dir, cache_root, alpha):
        assert cities == ["denver", "chicago"]
        assert start == date(2025, 1, 1)
        assert end == date(2025, 1, 3)
        assert test_start == date(2025, 1, 3)
        assert out_dir == tmp_path / "run"
        assert cache_root == tmp_path / "cache"
        assert alpha == 0.2
        return HistoricalRunResult(
            rows_path=out_dir / "rows.csv",
            summary_path=out_dir / "summary.csv",
            report_dir=out_dir / "model_report",
            train_eval_dir=out_dir / "train_eval",
            n_rows=3,
            n_summary_rows=2,
        )

    monkeypatch.setattr(
        historical_runner_cli, "run_historical_pipeline", fake_run_historical_pipeline
    )

    code = historical_runner_cli.main(
        [
            "--cities",
            "denver,chicago",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-03",
            "--test-start",
            "2025-01-03",
            "--out-dir",
            str(tmp_path / "run"),
            "--cache",
            str(tmp_path / "cache"),
        ]
    )

    assert code == 0
    assert "Wrote 3 rows" in capsys.readouterr().out

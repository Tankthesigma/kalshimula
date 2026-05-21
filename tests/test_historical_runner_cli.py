from datetime import date

from src import historical_runner_cli
from src.historical_runner import HistoricalRunResult


def test_historical_runner_cli_calls_pipeline(monkeypatch, tmp_path, capsys) -> None:
    def fake_run_historical_pipeline(
        *,
        cities,
        start,
        end,
        test_start,
        out_dir,
        cache_root,
        alpha,
        bias_strategy,
        bias_recent_days,
        openmeteo_mode,
        progress,
        workers,
        chunk_days,
    ):
        assert cities == ["denver", "chicago"]
        assert start == date(2025, 1, 1)
        assert end == date(2025, 1, 3)
        assert test_start == date(2025, 1, 3)
        assert out_dir == tmp_path / "run"
        assert cache_root == tmp_path / "cache"
        assert alpha == 0.13
        assert bias_strategy == "recent"
        assert bias_recent_days == 180
        assert openmeteo_mode == "both"
        assert progress is print
        assert workers == 3
        assert chunk_days == 5
        return HistoricalRunResult(
            rows_path=out_dir / "rows.csv",
            summary_path=out_dir / "summary.csv",
            errors_path=out_dir / "errors.csv",
            report_dir=out_dir / "model_report",
            train_eval_dir=out_dir / "train_eval",
            n_rows=3,
            n_summary_rows=2,
            n_errors=1,
            n_skipped=4,
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
            "--alpha",
            "0.13",
            "--bias-strategy",
            "recent",
            "--bias-recent-days",
            "180",
            "--openmeteo-mode",
            "both",
            "--workers",
            "3",
            "--chunk-days",
            "5",
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "Wrote 3 rows" in out
    assert "4 skipped, 1 errors" in out

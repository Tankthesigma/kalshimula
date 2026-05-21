from datetime import date

import pandas as pd
import pytest

from src import historical_runner
from src.collect import CollectionResult
from src.datasets.backtest import make_backtest_row
from src.historical_runner import run_historical_pipeline


def _row(city: str, target: date, point_f: float = 70) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": city,
                "target_date": target,
                "source": "openmeteo_naive",
                "point_f": point_f,
                "actual_high_f": 68,
                "absolute_error_f": abs(point_f - 68),
            }
        ]
    )


def _collection(city: str, target: date, point_f: float = 70) -> CollectionResult:
    return CollectionResult(
        city=city,
        start=target,
        end=target,
        rows=[
            make_backtest_row(
                city=city,
                target_date=target,
                source="openmeteo_naive",
                point_f=point_f,
                actual_high_f=68,
            )
        ],
    )


def test_run_historical_pipeline_writes_artifacts_incrementally(
    monkeypatch, tmp_path
) -> None:
    calls = []
    snapshots = []

    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        assert start == end
        assert cache_root == tmp_path / "cache"
        assert openmeteo_mode == "naive"
        calls.append((city, start))
        return _collection(city, start, point_f=70 + len(calls))

    def fake_write_rows(rows, path):
        snapshots.append(len(rows))
        rows.to_csv(path, index=False)

    monkeypatch.setattr(
        historical_runner, "collect_backtest_rows", fake_collect_backtest_rows
    )
    monkeypatch.setattr(historical_runner, "_write_rows", fake_write_rows)

    result = run_historical_pipeline(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 3),
        test_start=date(2025, 1, 3),
        out_dir=tmp_path / "run",
        cache_root=tmp_path / "cache",
    )

    assert calls == [
        ("denver", date(2025, 1, 1)),
        ("denver", date(2025, 1, 2)),
        ("denver", date(2025, 1, 3)),
    ]
    assert snapshots == [1, 2, 3]
    assert result.n_rows == 3
    assert result.n_summary_rows == 1
    assert result.n_errors == 0
    assert result.rows_path.exists()
    assert result.summary_path.exists()
    assert (result.report_dir / "raw_summary.csv").exists()
    assert (result.report_dir / "corrected_evaluation.csv").exists()
    assert (result.train_eval_dir / "evaluation.csv").exists()


def test_run_historical_pipeline_passes_bias_options_to_train_eval(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        return _collection(city, start)

    def fake_write_train_eval_outputs(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        historical_runner, "collect_backtest_rows", fake_collect_backtest_rows
    )
    monkeypatch.setattr(
        historical_runner, "write_train_eval_outputs", fake_write_train_eval_outputs
    )

    run_historical_pipeline(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 3),
        test_start=date(2025, 1, 3),
        out_dir=tmp_path / "run",
        cache_root=tmp_path / "cache",
        alpha=0.13,
        bias_strategy="recent",
        bias_recent_days=180,
    )

    assert captured["alpha"] == pytest.approx(0.13)
    assert captured["bias_strategy"] == "recent"
    assert captured["bias_recent_days"] == 180


def test_run_historical_pipeline_writes_source_selection_when_validation_enabled(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        return _collection(city, start)

    def fake_write_train_eval_outputs(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "city": "denver",
                    "source": "openmeteo_naive",
                    "method": "recent_180d",
                    "validation_mae": 0.8,
                }
            ]
        ).to_csv(output_dir / "validation_scores.csv", index=False)
        pd.DataFrame(
            [
                {
                    "city": "denver",
                    "source": "openmeteo_naive",
                    "mae_raw": 1.0,
                    "mae_corrected": 0.7,
                    "interval_coverage_raw": 0.8,
                    "interval_width_raw": 3.0,
                }
            ]
        ).to_csv(output_dir / "evaluation.csv", index=False)

    def fake_write_source_selection_outputs(**kwargs):
        captured.update(kwargs)
        kwargs["output_dir"].mkdir(parents=True)

    monkeypatch.setattr(
        historical_runner, "collect_backtest_rows", fake_collect_backtest_rows
    )
    monkeypatch.setattr(
        historical_runner, "write_train_eval_outputs", fake_write_train_eval_outputs
    )
    monkeypatch.setattr(
        historical_runner,
        "write_source_selection_outputs",
        fake_write_source_selection_outputs,
    )

    result = run_historical_pipeline(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 3),
        test_start=date(2025, 1, 3),
        validation_start=date(2025, 1, 2),
        out_dir=tmp_path / "run",
        cache_root=tmp_path / "cache",
    )

    assert captured["validation_scores_path"] == (
        result.train_eval_dir / "validation_scores.csv"
    )
    assert captured["evaluation_path"] == result.train_eval_dir / "evaluation.csv"
    assert captured["output_dir"] == result.source_selection_dir


def test_run_historical_pipeline_skips_source_selection_without_validation(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        return _collection(city, start)

    def fake_write_source_selection_outputs(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        historical_runner, "collect_backtest_rows", fake_collect_backtest_rows
    )
    monkeypatch.setattr(
        historical_runner,
        "write_source_selection_outputs",
        fake_write_source_selection_outputs,
    )

    run_historical_pipeline(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 3),
        test_start=date(2025, 1, 3),
        out_dir=tmp_path / "run",
        cache_root=tmp_path / "cache",
    )

    assert calls == []


def test_run_historical_pipeline_passes_openmeteo_mode_to_collect(
    monkeypatch, tmp_path
) -> None:
    modes = []

    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        modes.append(openmeteo_mode)
        return _collection(city, start)

    monkeypatch.setattr(
        historical_runner, "collect_backtest_rows", fake_collect_backtest_rows
    )

    run_historical_pipeline(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        test_start=date(2025, 1, 2),
        out_dir=tmp_path / "run",
        cache_root=tmp_path / "cache",
        openmeteo_mode="both",
    )

    assert modes == ["both", "both"]


def test_completed_city_dates_respects_openmeteo_mode() -> None:
    rows = pd.DataFrame(
        [
            {
                "city": "denver",
                "target_date": date(2025, 1, 1),
                "source": "openmeteo_naive",
            },
            {"city": "denver", "target_date": date(2025, 1, 2), "source": "gfs_ens"},
            {
                "city": "denver",
                "target_date": date(2025, 1, 3),
                "source": "openmeteo_naive",
            },
            {"city": "denver", "target_date": date(2025, 1, 3), "source": "gfs_ens"},
        ]
    )

    assert historical_runner._completed_city_dates(
        rows, openmeteo_mode="naive"
    ) == {
        ("denver", "2025-01-01"),
        ("denver", "2025-01-03"),
    }
    assert historical_runner._completed_city_dates(
        rows, openmeteo_mode="sources"
    ) == {
        ("denver", "2025-01-02"),
        ("denver", "2025-01-03"),
    }
    assert historical_runner._completed_city_dates(rows, openmeteo_mode="both") == {
        ("denver", "2025-01-03")
    }


def test_run_historical_pipeline_skips_existing_rows_on_rerun(
    monkeypatch, tmp_path
) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    pd.concat(
        [
            _row("denver", date(2025, 1, 1)),
            _row("denver", date(2025, 1, 2)),
        ],
        ignore_index=True,
    ).to_csv(out_dir / "rows.csv", index=False)
    calls = []
    progress = []

    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        calls.append((city, start))
        return _collection(city, start, point_f=73)

    monkeypatch.setattr(
        historical_runner, "collect_backtest_rows", fake_collect_backtest_rows
    )

    result = run_historical_pipeline(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 3),
        test_start=date(2025, 1, 3),
        out_dir=out_dir,
        cache_root=tmp_path / "cache",
        progress=progress.append,
    )

    rows = pd.read_csv(out_dir / "rows.csv")
    assert calls == [("denver", date(2025, 1, 3))]
    assert len(rows) == 3
    assert result.n_skipped == 2
    assert any(message.startswith("skip denver 2025-01-01") for message in progress)


def test_run_historical_pipeline_clears_stale_error_after_success(
    monkeypatch, tmp_path
) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    _row("denver", date(2025, 1, 1)).to_csv(out_dir / "rows.csv", index=False)
    pd.DataFrame(
        [
            {
                "city": "denver",
                "target_date": "2025-01-02",
                "error_type": "RuntimeError",
                "message": "temporary failure",
            }
        ]
    ).to_csv(out_dir / "errors.csv", index=False)

    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        return _collection(city, start, point_f=73)

    monkeypatch.setattr(
        historical_runner, "collect_backtest_rows", fake_collect_backtest_rows
    )

    result = run_historical_pipeline(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        test_start=date(2025, 1, 2),
        out_dir=out_dir,
        cache_root=tmp_path / "cache",
    )

    assert result.n_errors == 0
    assert not (out_dir / "errors.csv").exists()


def test_run_historical_pipeline_continues_after_source_failure(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        calls.append((city, start))
        if start == date(2025, 1, 2):
            raise RuntimeError("source timed out")
        return _collection(city, start)

    monkeypatch.setattr(
        historical_runner, "collect_backtest_rows", fake_collect_backtest_rows
    )

    result = run_historical_pipeline(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 3),
        test_start=date(2025, 1, 3),
        out_dir=tmp_path / "run",
        cache_root=tmp_path / "cache",
    )

    rows = pd.read_csv(result.rows_path)
    errors = pd.read_csv(result.errors_path)
    assert calls == [
        ("denver", date(2025, 1, 1)),
        ("denver", date(2025, 1, 2)),
        ("denver", date(2025, 1, 3)),
    ]
    assert len(rows) == 2
    assert result.n_errors == 1
    assert errors.iloc[0]["city"] == "denver"
    assert errors.iloc[0]["target_date"] == "2025-01-02"
    assert errors.iloc[0]["error_type"] == "RuntimeError"


def test_run_historical_pipeline_can_collect_chunks_in_parallel(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        calls.append((city, start))
        return _collection(city, start)

    monkeypatch.setattr(
        historical_runner, "collect_backtest_rows", fake_collect_backtest_rows
    )

    result = run_historical_pipeline(
        cities=["denver", "chicago"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        test_start=date(2025, 1, 2),
        out_dir=tmp_path / "run",
        cache_root=tmp_path / "cache",
        workers=2,
    )

    rows = pd.read_csv(result.rows_path)
    assert sorted(calls) == [
        ("chicago", date(2025, 1, 1)),
        ("chicago", date(2025, 1, 2)),
        ("denver", date(2025, 1, 1)),
        ("denver", date(2025, 1, 2)),
    ]
    assert len(rows) == 4
    assert result.n_rows == 4
    assert result.n_errors == 0


def test_run_historical_pipeline_rejects_invalid_worker_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="workers"):
        run_historical_pipeline(
            cities=["denver"],
            start=date(2025, 1, 1),
            end=date(2025, 1, 1),
            test_start=date(2025, 1, 1),
            out_dir=tmp_path / "run",
            cache_root=tmp_path / "cache",
            workers=0,
        )


def test_run_historical_pipeline_collects_sequential_ranges_when_chunk_days_set(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        calls.append((city, start, end))
        return CollectionResult(
            city=city,
            start=start,
            end=end,
            rows=[
                make_backtest_row(
                    city=city,
                    target_date=target,
                    source="openmeteo_naive",
                    point_f=70,
                    actual_high_f=68,
                )
                for target in historical_runner.date_range(start, end)
            ],
        )

    monkeypatch.setattr(
        historical_runner, "collect_backtest_rows", fake_collect_backtest_rows
    )

    result = run_historical_pipeline(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 5),
        test_start=date(2025, 1, 4),
        out_dir=tmp_path / "run",
        cache_root=tmp_path / "cache",
        chunk_days=2,
    )

    assert calls == [
        ("denver", date(2025, 1, 1), date(2025, 1, 2)),
        ("denver", date(2025, 1, 3), date(2025, 1, 4)),
        ("denver", date(2025, 1, 5), date(2025, 1, 5)),
    ]
    assert result.n_rows == 5


def test_run_historical_pipeline_rejects_invalid_chunk_days(tmp_path) -> None:
    with pytest.raises(ValueError, match="chunk_days"):
        run_historical_pipeline(
            cities=["denver"],
            start=date(2025, 1, 1),
            end=date(2025, 1, 1),
            test_start=date(2025, 1, 1),
            out_dir=tmp_path / "run",
            cache_root=tmp_path / "cache",
            chunk_days=0,
        )


def test_run_historical_pipeline_handles_empty_rows(monkeypatch, tmp_path) -> None:
    def fake_collect_backtest_rows(*, city, start, end, cache_root, openmeteo_mode):
        return CollectionResult(city=city, start=start, end=end, rows=[])

    monkeypatch.setattr(
        historical_runner, "collect_backtest_rows", fake_collect_backtest_rows
    )

    result = run_historical_pipeline(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
        test_start=date(2025, 1, 1),
        out_dir=tmp_path / "run",
        cache_root=tmp_path / "cache",
    )

    assert result.n_rows == 0
    assert (result.report_dir / "raw_summary.csv").exists()
    assert (result.train_eval_dir / "evaluation.csv").exists()

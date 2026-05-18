from datetime import date

import pandas as pd

from src import historical_runner
from src.historical_runner import run_historical_pipeline


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": "denver",
                "target_date": date(2025, 1, 1),
                "source": "openmeteo_naive",
                "point_f": 70,
                "actual_high_f": 68,
                "absolute_error_f": 2,
            },
            {
                "city": "denver",
                "target_date": date(2025, 1, 2),
                "source": "openmeteo_naive",
                "point_f": 72,
                "actual_high_f": 71,
                "absolute_error_f": 1,
            },
            {
                "city": "denver",
                "target_date": date(2025, 1, 3),
                "source": "openmeteo_naive",
                "point_f": 73,
                "actual_high_f": 73,
                "absolute_error_f": 0,
            },
        ]
    )


def test_run_historical_pipeline_writes_artifacts(monkeypatch, tmp_path) -> None:
    def fake_collect_many_cities(*, cities, start, end, cache_root):
        assert cities == ["denver"]
        assert start == date(2025, 1, 1)
        assert end == date(2025, 1, 3)
        assert cache_root == tmp_path / "cache"
        return _rows()

    monkeypatch.setattr(historical_runner, "collect_many_cities", fake_collect_many_cities)

    result = run_historical_pipeline(
        cities=["denver"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 3),
        test_start=date(2025, 1, 3),
        out_dir=tmp_path / "run",
        cache_root=tmp_path / "cache",
    )

    assert result.n_rows == 3
    assert result.n_summary_rows == 1
    assert result.rows_path.exists()
    assert result.summary_path.exists()
    assert (result.report_dir / "raw_summary.csv").exists()
    assert (result.report_dir / "corrected_evaluation.csv").exists()
    assert (result.train_eval_dir / "evaluation.csv").exists()


def test_run_historical_pipeline_handles_empty_rows(monkeypatch, tmp_path) -> None:
    def fake_collect_many_cities(*, cities, start, end, cache_root):
        return pd.DataFrame(
            columns=["city", "target_date", "source", "point_f", "actual_high_f", "absolute_error_f"]
        )

    monkeypatch.setattr(historical_runner, "collect_many_cities", fake_collect_many_cities)

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

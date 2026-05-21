"""Historical run orchestration for collection and model reports."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.batch_collect_cli import write_batch_outputs
from src.collect import OPENMETEO_NAIVE_SOURCE, collect_backtest_rows
from src.datasets.backtest import backtest_rows_to_dataframe
from src.datasets.collection import date_range
from src.fetchers import openmeteo
from src.models.report import write_model_report
from src.models.source_selection import write_source_selection_outputs
from src.models.train_eval import write_train_eval_outputs

ROW_COLUMNS = [
    "city",
    "target_date",
    "source",
    "point_f",
    "actual_high_f",
    "absolute_error_f",
]
ERROR_COLUMNS = ["city", "target_date", "error_type", "message"]
ProgressLogger = Callable[[str], None]
Chunk = tuple[str, date]


class _RateLimitStop(Exception):
    """Internal signal to stop collection and preserve resumable artifacts."""


@dataclass(frozen=True)
class HistoricalRunResult:
    """Paths and row counts produced by a historical run."""

    rows_path: Path
    summary_path: Path
    errors_path: Path | None
    report_dir: Path
    train_eval_dir: Path
    source_selection_dir: Path
    n_rows: int
    n_summary_rows: int
    n_errors: int = 0
    n_skipped: int = 0


def run_historical_pipeline(
    *,
    cities: list[str],
    start: date,
    end: date,
    test_start: date,
    validation_start: date | None = None,
    out_dir: Path,
    cache_root: Path,
    alpha: float = 0.2,
    bias_strategy: str = "seasonal",
    bias_recent_days: int | None = None,
    openmeteo_mode: str = "naive",
    progress: ProgressLogger | None = None,
    workers: int = 1,
    chunk_days: int = 1,
) -> HistoricalRunResult:
    """Collect rows and write all model/report artifacts for a historical run.

    Collection is chunked one city/date at a time. Existing rows are loaded from
    ``rows.csv`` so interrupted runs can be resumed without redoing completed
    city/date chunks.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.csv"
    summary_path = out_dir / "summary.csv"
    errors_path = out_dir / "errors.csv"
    report_dir = out_dir / "model_report"
    train_eval_dir = out_dir / "train_eval"
    source_selection_dir = out_dir / "source_selection"

    rows = _read_existing_rows(rows_path)
    completed = _completed_city_dates(rows, openmeteo_mode=openmeteo_mode)
    errors = _read_existing_errors(errors_path)
    n_skipped = 0

    pending, n_skipped = _pending_chunks(
        cities=cities,
        start=start,
        end=end,
        completed=completed,
        progress=progress,
    )
    _log(progress, f"{len(pending)} chunks pending ({n_skipped} skipped)")

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")

    try:
        if workers == 1 and chunk_days == 1:
            for city, target in pending:
                rows, errors = _collect_and_write_chunk(
                    city=city,
                    target=target,
                    cache_root=cache_root,
                    openmeteo_mode=openmeteo_mode,
                    rows=rows,
                    completed=completed,
                    errors=errors,
                    rows_path=rows_path,
                    errors_path=errors_path,
                    progress=progress,
                )
        elif workers == 1:
            for city, range_start, range_end in _pending_ranges(pending, chunk_days):
                rows, errors = _collect_and_write_range(
                    city=city,
                    start=range_start,
                    end=range_end,
                    cache_root=cache_root,
                    openmeteo_mode=openmeteo_mode,
                    rows=rows,
                    completed=completed,
                    errors=errors,
                    rows_path=rows_path,
                    errors_path=errors_path,
                    progress=progress,
                )
    except _RateLimitStop:
        _log(progress, "rate limit detected; stopping run for later resume")

    if workers > 1:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {
            executor.submit(
                _collect_chunk, city, target, cache_root, openmeteo_mode
            ): (city, target)
            for city, target in pending
        }
        try:
            for future in as_completed(futures):
                city, target = futures[future]
                try:
                    chunk = future.result()
                except Exception as error:
                    errors = _append_error(errors, city, target, error)
                    _write_errors(errors, errors_path)
                    _log(
                        progress,
                        f"failed {city} {target.isoformat()}: {type(error).__name__}: {error}",
                    )
                    if _is_rate_limit_error(error):
                        _log(progress, "rate limit detected; stopping run for later resume")
                        for pending_future in futures:
                            pending_future.cancel()
                        executor.shutdown(wait=True, cancel_futures=True)
                        break
                    continue

                new_errors = _drop_chunk_errors(errors, city, target)
                if len(new_errors) != len(errors):
                    errors = new_errors
                    _write_errors(errors, errors_path)
                if not chunk.empty:
                    rows = pd.concat([rows, chunk], ignore_index=True)
                completed.add((city, target.isoformat()))
                _write_rows(rows, rows_path)
                _log_row_progress(progress, len(rows), workers=workers)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    rows = _normalize_rows(rows)
    summary = write_batch_outputs(rows, rows_path, summary_path)
    if rows.empty:
        _write_empty_artifact_dirs(report_dir, train_eval_dir)
    else:
        write_model_report(input_path=rows_path, output_dir=report_dir, alpha=alpha)
        write_train_eval_outputs(
            input_path=rows_path,
            output_dir=train_eval_dir,
            test_start=test_start.isoformat(),
            alpha=alpha,
            bias_strategy=bias_strategy,
            bias_recent_days=bias_recent_days,
            validation_start=(
                validation_start.isoformat() if validation_start is not None else None
            ),
        )
        if validation_start is not None:
            write_source_selection_outputs(
                validation_scores_path=train_eval_dir / "validation_scores.csv",
                evaluation_path=train_eval_dir / "evaluation.csv",
                output_dir=source_selection_dir,
            )

    return HistoricalRunResult(
        rows_path=rows_path,
        summary_path=summary_path,
        errors_path=errors_path,
        report_dir=report_dir,
        train_eval_dir=train_eval_dir,
        source_selection_dir=source_selection_dir,
        n_rows=len(rows),
        n_summary_rows=len(summary),
        n_errors=len(errors),
        n_skipped=n_skipped,
    )


def _pending_chunks(
    *,
    cities: list[str],
    start: date,
    end: date,
    completed: set[tuple[str, str]],
    progress: ProgressLogger | None,
) -> tuple[list[Chunk], int]:
    pending: list[Chunk] = []
    n_skipped = 0
    for city in cities:
        for target in date_range(start, end):
            key = (city, target.isoformat())
            if key in completed:
                n_skipped += 1
                _log_skip_progress(progress, city, target, n_skipped)
                continue
            pending.append((city, target))
    return pending, n_skipped


def _collect_and_write_chunk(
    *,
    city: str,
    target: date,
    cache_root: Path,
    openmeteo_mode: str,
    rows: pd.DataFrame,
    completed: set[tuple[str, str]],
    errors: pd.DataFrame,
    rows_path: Path,
    errors_path: Path,
    progress: ProgressLogger | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _log(progress, f"collect {city} {target.isoformat()}")
    try:
        chunk = _collect_chunk(city, target, cache_root, openmeteo_mode)
    except Exception as error:
        errors = _append_error(errors, city, target, error)
        _write_errors(errors, errors_path)
        _log(
            progress,
            f"failed {city} {target.isoformat()}: {type(error).__name__}: {error}",
        )
        if _is_rate_limit_error(error):
            raise _RateLimitStop from error
        return rows, errors

    new_errors = _drop_chunk_errors(errors, city, target)
    if len(new_errors) != len(errors):
        errors = new_errors
        _write_errors(errors, errors_path)
    if not chunk.empty:
        rows = pd.concat([rows, chunk], ignore_index=True)
        completed.add((city, target.isoformat()))
    _write_rows(rows, rows_path)
    _log_row_progress(progress, len(rows), workers=1)
    return rows, errors


def _collect_chunk(
    city: str, target: date, cache_root: Path, openmeteo_mode: str
) -> pd.DataFrame:
    result = collect_backtest_rows(
        city=city,
        start=target,
        end=target,
        cache_root=cache_root,
        openmeteo_mode=openmeteo_mode,
    )
    return backtest_rows_to_dataframe(result.rows)


def _pending_ranges(pending: list[Chunk], chunk_days: int) -> list[tuple[str, date, date]]:
    ranges: list[tuple[str, date, date]] = []
    if not pending:
        return ranges

    by_city: dict[str, list[date]] = {}
    for city, target in pending:
        by_city.setdefault(city, []).append(target)

    for city, targets in by_city.items():
        sorted_targets = sorted(targets)
        range_start = sorted_targets[0]
        previous = sorted_targets[0]
        count = 1
        for target in sorted_targets[1:]:
            contiguous = (target - previous).days == 1
            if not contiguous or count >= chunk_days:
                ranges.append((city, range_start, previous))
                range_start = target
                count = 1
            else:
                count += 1
            previous = target
        ranges.append((city, range_start, previous))

    return ranges


def _collect_and_write_range(
    *,
    city: str,
    start: date,
    end: date,
    cache_root: Path,
    openmeteo_mode: str,
    rows: pd.DataFrame,
    completed: set[tuple[str, str]],
    errors: pd.DataFrame,
    rows_path: Path,
    errors_path: Path,
    progress: ProgressLogger | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _log(progress, f"collect {city} {start.isoformat()}..{end.isoformat()}")
    try:
        result = collect_backtest_rows(
            city=city,
            start=start,
            end=end,
            cache_root=cache_root,
            openmeteo_mode=openmeteo_mode,
        )
        chunk = backtest_rows_to_dataframe(result.rows)
    except Exception as error:
        for target in date_range(start, end):
            errors = _append_error(errors, city, target, error)
        _write_errors(errors, errors_path)
        _log(
            progress,
            f"failed {city} {start.isoformat()}..{end.isoformat()}: {type(error).__name__}: {error}",
        )
        if _is_rate_limit_error(error):
            raise _RateLimitStop from error
        return rows, errors

    if not chunk.empty:
        rows = pd.concat([rows, chunk], ignore_index=True)
        for target in pd.to_datetime(chunk["target_date"]).dt.date:
            completed.add((city, target.isoformat()))
            errors = _drop_chunk_errors(errors, city, target)
        _write_errors(errors, errors_path)

    _write_rows(rows, rows_path)
    _log(progress, f"wrote {len(rows)} total rows")
    return rows, errors


def _read_existing_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=ROW_COLUMNS)
    return _normalize_rows(pd.read_csv(path, parse_dates=["target_date"]))


def _write_rows(rows: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_rows(rows).to_csv(path, index=False)


def _normalize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(rows, columns=ROW_COLUMNS)
    normalized = rows.copy()
    normalized["target_date"] = pd.to_datetime(normalized["target_date"]).dt.date
    return normalized


def _completed_city_dates(
    rows: pd.DataFrame, *, openmeteo_mode: str = "naive"
) -> set[tuple[str, str]]:
    if rows.empty:
        return set()
    df = rows.copy()
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
    source_slugs = {slug for slug, *_ in openmeteo.SOURCES}
    completed: set[tuple[str, str]] = set()
    for (city, target), group in df.groupby(["city", "target_date"], sort=False):
        sources = set(group["source"].astype(str))
        if _has_expected_source(sources, target, source_slugs, openmeteo_mode):
            completed.add((str(city), target.isoformat()))
    return completed


def _has_expected_source(
    sources: set[str],
    target: date,
    source_slugs: set[str],
    openmeteo_mode: str,
) -> bool:
    if target >= date.today():
        return "nws" in sources
    has_naive = OPENMETEO_NAIVE_SOURCE in sources
    has_source = bool(sources & source_slugs)
    if openmeteo_mode == "naive":
        return has_naive
    if openmeteo_mode == "sources":
        return has_source
    if openmeteo_mode == "both":
        return has_naive and has_source
    raise ValueError(f"unknown openmeteo_mode {openmeteo_mode!r}")


def _read_existing_errors(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=ERROR_COLUMNS)
    return pd.read_csv(path)


def _append_error(
    errors: pd.DataFrame, city: str, target: date, error: Exception
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "city": city,
                "target_date": target.isoformat(),
                "error_type": type(error).__name__,
                "message": str(error),
            }
        ],
        columns=ERROR_COLUMNS,
    )
    return pd.concat([errors, row], ignore_index=True)


def _write_errors(errors: pd.DataFrame, path: Path) -> None:
    if errors.empty:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(path, index=False)


def _drop_chunk_errors(errors: pd.DataFrame, city: str, target: date) -> pd.DataFrame:
    if errors.empty:
        return errors
    target_text = target.isoformat()
    mask = (errors["city"].astype(str) == city) & (
        errors["target_date"].astype(str) == target_text
    )
    if not mask.any():
        return errors
    return errors.loc[~mask].reset_index(drop=True)


def _is_rate_limit_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 429


def _log(progress: ProgressLogger | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _log_row_progress(
    progress: ProgressLogger | None, row_count: int, *, workers: int
) -> None:
    if workers == 1 or row_count <= 10 or row_count % 25 == 0:
        _log(progress, f"wrote {row_count} total rows")


def _log_skip_progress(
    progress: ProgressLogger | None, city: str, target: date, n_skipped: int
) -> None:
    if n_skipped <= 10 or n_skipped % 100 == 0:
        _log(progress, f"skip {city} {target.isoformat()} already collected")


def _write_empty_artifact_dirs(report_dir: Path, train_eval_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    train_eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(report_dir / "raw_summary.csv", index=False)
    pd.DataFrame().to_csv(report_dir / "bias_table.csv", index=False)
    pd.DataFrame().to_csv(report_dir / "corrected_evaluation.csv", index=False)
    pd.DataFrame().to_csv(report_dir / "intervals.csv", index=False)
    pd.DataFrame().to_csv(train_eval_dir / "evaluation.csv", index=False)

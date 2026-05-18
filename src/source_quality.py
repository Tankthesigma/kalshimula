"""Source-quality reporting on smoke-result CSVs.

Reads the long-format smoke CSV emitted by :mod:`src.smoke_weather_cli` and
summarizes per-source reliability: how many calls succeeded, how many raised,
and how many succeeded but produced no usable high. Pure DataFrame in / pure
DataFrame out — no network, no fetcher imports.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "city",
    "target_date",
    "source",
    "ok",
    "high_f",
    "error",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "source",
    "n",
    "ok_count",
    "error_count",
    "missing_high_count",
    "ok_rate",
    "missing_high_rate",
)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"smoke results dataframe is missing required columns: {missing}"
        )


def summarize_source_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-source reliability stats from a long-format smoke frame.

    Definitions:
    - ``n``: rows for this source.
    - ``ok_count``: rows where ``ok`` is truthy.
    - ``error_count``: rows where ``ok`` is falsy.
    - ``missing_high_count``: rows where ``ok`` is truthy *but* ``high_f`` is
      missing — i.e. the fetcher returned cleanly but the source had no data.
    - ``ok_rate``: ``ok_count / n``.
    - ``missing_high_rate``: ``missing_high_count / ok_count`` (0 when
      ``ok_count`` is 0, so the denominator is well-defined).
    """
    _validate_columns(df)

    if df.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))

    work = df.copy()
    work["ok_bool"] = work["ok"].astype(bool)
    # high_f may already be NaN; treat any missing as missing.
    work["high_missing"] = work["high_f"].isna()

    rows = []
    for source, group in work.groupby("source", sort=True):
        n = int(len(group))
        ok_count = int(group["ok_bool"].sum())
        error_count = n - ok_count
        missing_high_count = int(
            (group["ok_bool"] & group["high_missing"]).sum()
        )
        ok_rate = ok_count / n if n else 0.0
        missing_high_rate = (
            missing_high_count / ok_count if ok_count else 0.0
        )
        rows.append(
            {
                "source": source,
                "n": n,
                "ok_count": ok_count,
                "error_count": error_count,
                "missing_high_count": missing_high_count,
                "ok_rate": ok_rate,
                "missing_high_rate": missing_high_rate,
            }
        )
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def read_smoke_results(path: Path) -> pd.DataFrame:
    """Read a smoke-result CSV produced by :mod:`src.smoke_weather_cli`."""
    return pd.read_csv(path)


def write_source_quality(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Read smoke CSV at ``input_path``, summarize, write to ``output_path``."""
    df = read_smoke_results(input_path)
    summary = summarize_source_quality(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary

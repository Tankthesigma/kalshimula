"""CLI for summarizing selected model-policy artifacts from a run directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt_number(value: Any, *, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def _fmt_percent(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _first_row(table: pd.DataFrame) -> pd.Series | None:
    if table.empty:
        return None
    return table.iloc[0]


def _recommended_row(table: pd.DataFrame, *, split: str | None = None) -> pd.Series | None:
    if table.empty or "recommended" not in table.columns:
        return None
    filtered = table[table["recommended"].astype(str).str.lower() == "true"]
    if split is not None and "split" in filtered.columns:
        filtered = filtered[filtered["split"].astype(str) == split]
    return _first_row(filtered)


def _source_policy_lines(recommended_sources: pd.DataFrame) -> list[str]:
    if recommended_sources.empty:
        return ["Source policy: missing source_selection/recommended_sources.csv"]
    required = {"selected_source", "city"}
    if missing := required - set(recommended_sources.columns):
        return [f"Source policy: unreadable, missing columns {sorted(missing)}"]

    counts = (
        recommended_sources.groupby("selected_source", sort=True)
        .size()
        .reset_index(name="cities")
        .sort_values(["cities", "selected_source"], ascending=[False, True])
    )
    parts = [
        f"{row.selected_source}={int(row.cities)} cities"
        for row in counts.itertuples(index=False)
    ]
    policy = "n/a"
    if "recommended_policy" in recommended_sources.columns:
        policies = sorted(str(value) for value in recommended_sources["recommended_policy"].dropna().unique())
        policy = ", ".join(policies) if policies else "n/a"
    return [f"Source policy: {'; '.join(parts)} ({policy})"]


def _model_policy_lines(model_policy: pd.DataFrame) -> list[str]:
    row = _first_row(model_policy)
    if row is None:
        return ["Model policy: missing model_policy/model_policy.csv"]
    recent = row.get("bias_recent_days", "n/a")
    recent_text = "n/a" if pd.isna(recent) else str(int(float(recent)))
    return [
        "Model policy: "
        f"{row.get('policy', 'n/a')} "
        f"(bias={row.get('bias_strategy', 'n/a')}, "
        f"recent_days={recent_text}, alpha={_fmt_number(row.get('alpha'))})"
    ]


def _bias_policy_lines(comparison: pd.DataFrame) -> list[str]:
    row = _recommended_row(comparison)
    if row is None:
        return ["Bias policy metrics: missing recommended row"]
    return [
        "Bias policy metrics: "
        f"validation MAE={_fmt_number(row.get('validation_mae_corrected'))}F, "
        f"test MAE={_fmt_number(row.get('test_mae_corrected'))}F, "
        f"test coverage={_fmt_percent(row.get('test_interval_coverage_raw'))}, "
        f"test width={_fmt_number(row.get('test_interval_width_raw'))}F"
    ]


def _interval_policy_lines(comparison: pd.DataFrame) -> list[str]:
    row = _recommended_row(comparison, split="test")
    if row is None:
        return ["Interval policy metrics: missing recommended test row"]
    return [
        "Interval policy metrics: "
        f"{row.get('policy', 'n/a')} alpha={row.get('alpha', 'n/a')}, "
        f"test coverage={_fmt_percent(row.get('interval_coverage_raw'))}, "
        f"test width={_fmt_number(row.get('interval_width_raw'))}F, "
        f"target={_fmt_percent(row.get('target_coverage'))}"
    ]


def _threshold_lines(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["Threshold probabilities: missing probability_calibration/threshold_calibration_summary.csv"]
    required = {"split", "n_events", "brier_score", "expected_calibration_error"}
    if missing := required - set(summary.columns):
        return [f"Threshold probabilities: unreadable, missing columns {sorted(missing)}"]

    lines = ["Threshold probabilities:"]
    for _, row in summary.iterrows():
        lines.append(
            "  "
            f"{row['split']}: events={int(row['n_events']):,}, "
            f"brier={_fmt_number(row['brier_score'])}, "
            f"ece={_fmt_number(row['expected_calibration_error'])}, "
            f"pred={_fmt_percent(row.get('mean_predicted_probability', pd.NA))}, "
            f"obs={_fmt_percent(row.get('observed_frequency', pd.NA))}"
        )
    return lines


def _data_lines(rows: pd.DataFrame) -> list[str]:
    if rows.empty:
        return ["Data: missing or empty rows.csv"]
    cities = rows["city"].nunique() if "city" in rows.columns else "n/a"
    sources = rows["source"].nunique() if "source" in rows.columns else "n/a"
    if "target_date" in rows.columns:
        dates = pd.to_datetime(rows["target_date"], errors="coerce").dropna()
        date_range = (
            f"{dates.min().date()} to {dates.max().date()}" if not dates.empty else "n/a"
        )
    else:
        date_range = "n/a"
    return [f"Data: {len(rows):,} rows, {cities} cities, {sources} sources, dates {date_range}"]


def build_model_policy_report(run_dir: Path) -> str:
    """Build a compact text report from final model-policy artifacts."""
    rows = _read_csv_if_exists(run_dir / "rows.csv")
    recommended_sources = _read_csv_if_exists(run_dir / "source_selection" / "recommended_sources.csv")
    model_policy = _read_csv_if_exists(run_dir / "model_policy" / "model_policy.csv")
    bias_comparison = _read_csv_if_exists(run_dir / "model_policy" / "bias_policy_comparison.csv")
    interval_comparison = _read_csv_if_exists(
        run_dir / "model_policy" / "interval_policy_comparison.csv"
    )
    threshold_summary = _read_csv_if_exists(
        run_dir / "probability_calibration" / "threshold_calibration_summary.csv"
    )

    lines = [f"Run: {run_dir}"]
    lines.extend(_data_lines(rows))
    lines.extend(_source_policy_lines(recommended_sources))
    lines.extend(_model_policy_lines(model_policy))
    lines.extend(_bias_policy_lines(bias_comparison))
    lines.extend(_interval_policy_lines(interval_comparison))
    lines.extend(_threshold_lines(threshold_summary))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model_policy_report",
        description="Summarize selected source, bias, interval, and threshold calibration artifacts.",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional text file to write in addition to stdout.",
    )
    args = parser.parse_args(argv)

    report = build_model_policy_report(args.run_dir)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

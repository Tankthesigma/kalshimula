"""Source-vs-consensus diagnostics for Open-Meteo source rows.

These helpers are descriptive model diagnostics. They do not use market prices
and should not be interpreted as trading signals without a separate private
market-data audit.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PRIMARY_SOURCES = (
    "gfs_ens",
    "ecmwf_ens",
    "icon_ens",
    "gem_ens",
    "aifs",
    "graphcast",
    "hrrr",
)
CONSENSUS_SOURCE = "openmeteo_naive"
DEFAULT_OFFSETS = (-6, -4, -2, 0, 2, 4, 6)

DAILY_SOURCE_DELTA_COLUMNS = [
    "city",
    "target_date",
    "month",
    "source",
    "blend_source",
    "source_point_f",
    "blend_point_f",
    "abs_delta_f",
    "signed_delta_f",
    "delta_sign",
    "actual_high_f",
    "source_residual_f",
    "blend_residual_f",
    "source_abs_error_f",
    "blend_abs_error_f",
    "contrarian_correct",
]
SOURCE_METRIC_COLUMNS = [
    "city",
    "source",
    "n_days",
    "source_mae",
    "blend_mae",
    "mae_delta",
    "source_bias",
    "blend_bias",
    "mean_abs_delta_f",
    "mean_signed_delta_f",
    "contrarian_n",
    "contrarian_correct_n",
    "contrarian_correct_rate",
    "contrarian_correct_ci_lower_95",
    "contrarian_correct_ci_upper_95",
    "promoted",
    "promote_reason",
]
MONTHLY_SOURCE_METRIC_COLUMNS = ["month", *SOURCE_METRIC_COLUMNS]
THRESHOLD_GRID_COLUMNS = [
    "city",
    "source",
    "offset_f",
    "n_days",
    "mean_source_prob_above",
    "mean_blend_prob_above",
    "mean_prob_delta",
    "mean_abs_prob_delta",
    "source_brier",
    "blend_brier",
    "brier_delta",
    "source_edge_direction_correct_rate",
]


@dataclass(frozen=True)
class SourceContrarianDiagnostics:
    """Artifacts from source-vs-consensus diagnostics."""

    daily_source_deltas: pd.DataFrame
    monthly_source_metrics: pd.DataFrame
    source_contrarian_summary: pd.DataFrame
    source_threshold_grid: pd.DataFrame
    contrarian_value_index: str
    manifest: dict[str, object]


def build_source_contrarian_diagnostics(
    rows: pd.DataFrame,
    *,
    offsets: Iterable[int | float] = DEFAULT_OFFSETS,
    input_path: str | None = None,
    input_sha256: str | None = None,
    command_args: dict[str, object] | None = None,
    git_commit: str | None = None,
) -> SourceContrarianDiagnostics:
    """Build all source-vs-consensus diagnostic tables."""
    offset_values = tuple(float(offset) for offset in offsets)
    daily = build_daily_source_deltas(rows)
    monthly = build_monthly_source_metrics(daily)
    summary = build_source_contrarian_summary(daily)
    threshold_grid = build_source_threshold_grid(daily, offsets=offset_values)
    report = render_contrarian_value_index(summary, monthly, threshold_grid)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "input_path": input_path,
        "input_sha256": input_sha256,
        "command_args": command_args or {},
        "offsets": list(offset_values),
        "row_counts": {
            "input_rows": int(len(rows)),
            "daily_source_deltas": int(len(daily)),
            "monthly_source_metrics": int(len(monthly)),
            "source_contrarian_summary": int(len(summary)),
            "source_threshold_grid": int(len(threshold_grid)),
        },
    }
    return SourceContrarianDiagnostics(
        daily_source_deltas=daily,
        monthly_source_metrics=monthly,
        source_contrarian_summary=summary,
        source_threshold_grid=threshold_grid,
        contrarian_value_index=report,
        manifest=manifest,
    )


def build_daily_source_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare each individual source row to same-city/date consensus."""
    required = {"city", "target_date", "source", "point_f", "actual_high_f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"rows missing required columns: {sorted(missing)}")
    if rows.empty:
        return pd.DataFrame(columns=DAILY_SOURCE_DELTA_COLUMNS)

    clean = rows.loc[:, list(required)].copy()
    clean["city"] = clean["city"].astype(str)
    clean["source"] = clean["source"].astype(str)
    clean["target_date"] = pd.to_datetime(clean["target_date"], errors="coerce")
    clean["point_f"] = pd.to_numeric(clean["point_f"], errors="coerce")
    clean["actual_high_f"] = pd.to_numeric(clean["actual_high_f"], errors="coerce")
    clean = clean.dropna(subset=["target_date", "point_f", "actual_high_f"])
    if clean.empty:
        return pd.DataFrame(columns=DAILY_SOURCE_DELTA_COLUMNS)

    output_rows: list[dict[str, object]] = []
    for (city, target_date), group in clean.groupby(["city", "target_date"], sort=True):
        actual = float(group["actual_high_f"].iloc[0])
        consensus = _consensus_row(group)
        for source_row in group.sort_values("source").itertuples(index=False):
            source = str(source_row.source)
            if source == CONSENSUS_SOURCE:
                continue
            blend_point, blend_source = _blend_for_source(group, source, consensus)
            if blend_point is None:
                continue

            source_point = float(source_row.point_f)
            signed_delta = source_point - blend_point
            abs_delta = abs(signed_delta)
            actual_vs_blend = actual - blend_point
            delta_sign = _sign(signed_delta)
            contrarian_correct = (
                pd.NA
                if abs_delta == 0
                else bool(delta_sign == _sign(actual_vs_blend))
            )
            source_residual = actual - source_point
            blend_residual = actual - blend_point
            output_rows.append(
                {
                    "city": city,
                    "target_date": target_date.date().isoformat(),
                    "month": int(target_date.month),
                    "source": source,
                    "blend_source": blend_source,
                    "source_point_f": source_point,
                    "blend_point_f": blend_point,
                    "abs_delta_f": abs_delta,
                    "signed_delta_f": signed_delta,
                    "delta_sign": delta_sign,
                    "actual_high_f": actual,
                    "source_residual_f": source_residual,
                    "blend_residual_f": blend_residual,
                    "source_abs_error_f": abs(source_residual),
                    "blend_abs_error_f": abs(blend_residual),
                    "contrarian_correct": contrarian_correct,
                }
            )
    return pd.DataFrame(output_rows, columns=DAILY_SOURCE_DELTA_COLUMNS)


def build_monthly_source_metrics(daily_source_deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize source-vs-consensus diagnostics by city/source/month."""
    if daily_source_deltas.empty:
        return pd.DataFrame(columns=MONTHLY_SOURCE_METRIC_COLUMNS)
    rows = []
    for (month, city, source), group in daily_source_deltas.groupby(
        ["month", "city", "source"], sort=True
    ):
        rows.append({"month": int(month), **_metric_row(city, source, group)})
    return pd.DataFrame(rows, columns=MONTHLY_SOURCE_METRIC_COLUMNS)


def build_source_contrarian_summary(daily_source_deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize source-vs-consensus diagnostics by city/source."""
    if daily_source_deltas.empty:
        return pd.DataFrame(columns=SOURCE_METRIC_COLUMNS)
    rows = []
    for (city, source), group in daily_source_deltas.groupby(["city", "source"], sort=True):
        rows.append(_metric_row(city, source, group))
    return pd.DataFrame(rows, columns=SOURCE_METRIC_COLUMNS)


def build_source_threshold_grid(
    daily_source_deltas: pd.DataFrame,
    *,
    offsets: Iterable[int | float] = DEFAULT_OFFSETS,
) -> pd.DataFrame:
    """Build descriptive source/blend threshold probability diagnostics.

    Probabilities are in-sample empirical estimates from the same residuals used
    for evaluation. This is useful for source diagnostics, but it is not a live
    trading or PnL estimate.
    """
    if daily_source_deltas.empty:
        return pd.DataFrame(columns=THRESHOLD_GRID_COLUMNS)

    rows = []
    for (city, source), group in daily_source_deltas.groupby(["city", "source"], sort=True):
        source_residuals = group["source_residual_f"].astype(float)
        blend_residuals = group["blend_residual_f"].astype(float)
        for offset in offsets:
            events = []
            source_probs = []
            blend_probs = []
            for row in group.itertuples(index=False):
                threshold = round(float(row.blend_point_f)) + float(offset)
                outcome = 1.0 if float(row.actual_high_f) >= threshold else 0.0
                source_cutoff = threshold - float(row.source_point_f)
                blend_cutoff = threshold - float(row.blend_point_f)
                source_prob = float((source_residuals >= source_cutoff).mean())
                blend_prob = float((blend_residuals >= blend_cutoff).mean())
                events.append(outcome)
                source_probs.append(source_prob)
                blend_probs.append(blend_prob)

            event_series = pd.Series(events, dtype=float)
            source_series = pd.Series(source_probs, dtype=float)
            blend_series = pd.Series(blend_probs, dtype=float)
            prob_delta = source_series - blend_series
            source_brier = _brier(source_series, event_series)
            blend_brier = _brier(blend_series, event_series)
            rows.append(
                {
                    "city": city,
                    "source": source,
                    "offset_f": float(offset),
                    "n_days": int(len(group)),
                    "mean_source_prob_above": float(source_series.mean()),
                    "mean_blend_prob_above": float(blend_series.mean()),
                    "mean_prob_delta": float(prob_delta.mean()),
                    "mean_abs_prob_delta": float(prob_delta.abs().mean()),
                    "source_brier": source_brier,
                    "blend_brier": blend_brier,
                    "brier_delta": source_brier - blend_brier,
                    "source_edge_direction_correct_rate": _direction_correct_rate(
                        prob_delta,
                        event_series - blend_series,
                    ),
                }
            )
    return pd.DataFrame(rows, columns=THRESHOLD_GRID_COLUMNS)


def render_contrarian_value_index(
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    threshold_grid: pd.DataFrame,
) -> str:
    """Render a compact markdown report for source diagnostics."""
    lines = [
        "# Source Contrarian Diagnostics",
        "",
        "**Warning:** This is a source diagnostic, not a trading signal. It uses no Kalshi prices or market data. It becomes market-relevant only if Bobby's private audit shows market prices resemble the consensus being compared here.",
        "",
        "## Promotion Rule",
        "",
        "Promoted rows require `contrarian_correct_rate >= 0.55`, `contrarian_correct_ci_lower_95 > 0.50`, `mean_abs_delta_f >= 1.0`, and `n_days >= 180`.",
        "",
    ]
    lines.extend(_summary_table("Top Promoted City/Source Combos", _top_promoted(summary)))
    lines.extend(
        _summary_table(
            "High-MAE But High-Contrarian-Value Combos",
            summary.sort_values(
                ["contrarian_correct_ci_lower_95", "source_mae"],
                ascending=[False, False],
            ).head(10),
        )
    )
    avoid = summary.sort_values(
        ["source_mae", "contrarian_correct_rate", "mean_abs_delta_f"],
        ascending=[False, True, False],
    ).head(10)
    lines.extend(_summary_table("Source Combos To Avoid", avoid))
    consensus_wins = summary[summary["mae_delta"] > 0].sort_values(
        ["mae_delta", "blend_mae"],
        ascending=[False, True],
    )
    lines.extend(_summary_table("Where Consensus Beats The Source", consensus_wins.head(10)))
    if not threshold_grid.empty:
        tail = threshold_grid.sort_values("mean_abs_prob_delta", ascending=False).head(10)
        lines.extend(_threshold_table("Largest Threshold Probability Deltas", tail))
    lines.extend(
        [
            "## Notes For Bobby's Private Kalshi Audit",
            "",
            "- Use promoted combos as candidates, not proof.",
            "- Test whether market prices resemble consensus before assigning money meaning.",
            "- Compare promoted filters against gfs_ens, openmeteo_naive, blend_equal, and selected-source policies.",
            "- Kill the filter if it removes winners as often as losers or fails after fees/slippage.",
            "",
            "## Manual Paper-Check Checklist",
            "",
            "- Check only cities with enough sample and stable source behavior first.",
            "- Prefer combos where the source both disagrees meaningfully and has a CI lower bound above 50%.",
            "- Treat low-sample monthly spikes as research leads, not decisions.",
            "- Confirm tomorrow's forecast packet and source availability before any manual market review.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_source_contrarian_diagnostics(
    *,
    input_path: Path,
    output_dir: Path,
    offsets: Iterable[int | float] = DEFAULT_OFFSETS,
    command_args: dict[str, object] | None = None,
    git_commit: str | None = None,
) -> SourceContrarianDiagnostics:
    """Read rows, write diagnostics CSVs, markdown, and manifest."""
    rows = pd.read_csv(input_path)
    input_sha256 = _sha256(input_path)
    result = build_source_contrarian_diagnostics(
        rows,
        offsets=offsets,
        input_path=str(input_path),
        input_sha256=input_sha256,
        command_args=command_args,
        git_commit=git_commit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.daily_source_deltas.to_csv(output_dir / "daily_source_deltas.csv", index=False)
    result.monthly_source_metrics.to_csv(output_dir / "monthly_source_metrics.csv", index=False)
    result.source_contrarian_summary.to_csv(
        output_dir / "source_contrarian_summary.csv", index=False
    )
    result.source_threshold_grid.to_csv(output_dir / "source_threshold_grid.csv", index=False)
    (output_dir / "contrarian_value_index.md").write_text(
        result.contrarian_value_index,
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _consensus_row(group: pd.DataFrame) -> float | None:
    consensus = group[group["source"] == CONSENSUS_SOURCE]
    if consensus.empty:
        return None
    return float(consensus.iloc[0]["point_f"])


def _blend_for_source(
    group: pd.DataFrame,
    source: str,
    consensus: float | None,
) -> tuple[float | None, str]:
    if consensus is not None:
        return consensus, CONSENSUS_SOURCE
    primary = group[group["source"].isin(PRIMARY_SOURCES)]
    if source in PRIMARY_SOURCES:
        primary = primary[primary["source"] != source]
    if primary.empty:
        return None, "computed_equal_primary"
    return float(primary["point_f"].astype(float).mean()), "computed_equal_primary"


def _metric_row(city: str, source: str, group: pd.DataFrame) -> dict[str, object]:
    contrarian = group["contrarian_correct"].dropna()
    contrarian_n = int(len(contrarian))
    contrarian_correct_n = int(contrarian.astype(bool).sum()) if contrarian_n else 0
    rate = contrarian_correct_n / contrarian_n if contrarian_n else pd.NA
    ci_lower, ci_upper = wilson_interval(contrarian_correct_n, contrarian_n)
    mean_abs_delta = float(group["abs_delta_f"].astype(float).mean())
    promoted, reason = _promotion(rate, ci_lower, mean_abs_delta, int(len(group)))
    source_mae = float(group["source_abs_error_f"].astype(float).mean())
    blend_mae = float(group["blend_abs_error_f"].astype(float).mean())
    return {
        "city": city,
        "source": source,
        "n_days": int(len(group)),
        "source_mae": source_mae,
        "blend_mae": blend_mae,
        "mae_delta": source_mae - blend_mae,
        "source_bias": float(group["source_residual_f"].astype(float).mean()),
        "blend_bias": float(group["blend_residual_f"].astype(float).mean()),
        "mean_abs_delta_f": mean_abs_delta,
        "mean_signed_delta_f": float(group["signed_delta_f"].astype(float).mean()),
        "contrarian_n": contrarian_n,
        "contrarian_correct_n": contrarian_correct_n,
        "contrarian_correct_rate": rate,
        "contrarian_correct_ci_lower_95": ci_lower,
        "contrarian_correct_ci_upper_95": ci_upper,
        "promoted": promoted,
        "promote_reason": reason,
    }


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[object, object]:
    """Return the Wilson score confidence interval for a binomial rate."""
    if n <= 0:
        return pd.NA, pd.NA
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _promotion(
    rate: object,
    ci_lower: object,
    mean_abs_delta: float,
    n_days: int,
) -> tuple[bool, str]:
    failures = []
    if n_days < 180:
        failures.append("low_n")
    if mean_abs_delta < 1.0:
        failures.append("low_delta")
    if pd.isna(rate) or float(rate) < 0.55:
        failures.append("rate_below_threshold")
    if pd.isna(ci_lower) or float(ci_lower) <= 0.50:
        failures.append("ci_not_above_half")
    if failures:
        return False, ";".join(failures)
    return True, "promoted"


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _brier(probabilities: pd.Series, outcomes: pd.Series) -> float:
    return float(((probabilities.astype(float) - outcomes.astype(float)) ** 2).mean())


def _direction_correct_rate(prob_delta: pd.Series, outcome_delta: pd.Series) -> object:
    valid = prob_delta[prob_delta != 0].index
    if len(valid) == 0:
        return pd.NA
    correct = [
        _sign(float(prob_delta.loc[index])) == _sign(float(outcome_delta.loc[index]))
        for index in valid
    ]
    return float(sum(correct) / len(correct))


def _top_promoted(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    promoted = summary[summary["promoted"] == True]  # noqa: E712
    return promoted.sort_values(
        ["contrarian_correct_ci_lower_95", "contrarian_correct_rate", "mean_abs_delta_f"],
        ascending=[False, False, False],
    ).head(10)


def _summary_table(title: str, rows: pd.DataFrame) -> list[str]:
    lines = [f"## {title}", ""]
    columns = [
        "city",
        "source",
        "n_days",
        "source_mae",
        "blend_mae",
        "mae_delta",
        "mean_abs_delta_f",
        "contrarian_correct_rate",
        "contrarian_correct_ci_lower_95",
        "promote_reason",
    ]
    if rows.empty:
        return [*lines, "No rows.", ""]
    lines.append("| city | source | n | source MAE | blend MAE | MAE delta | abs delta | contrarian rate | CI lower | reason |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows.loc[:, columns].itertuples(index=False):
        lines.append(
            f"| {row.city} | {row.source} | {row.n_days} | "
            f"{_fmt(row.source_mae)} | {_fmt(row.blend_mae)} | {_fmt(row.mae_delta)} | "
            f"{_fmt(row.mean_abs_delta_f)} | {_fmt(row.contrarian_correct_rate)} | "
            f"{_fmt(row.contrarian_correct_ci_lower_95)} | {row.promote_reason} |"
        )
    lines.append("")
    return lines


def _threshold_table(title: str, rows: pd.DataFrame) -> list[str]:
    lines = [f"## {title}", ""]
    if rows.empty:
        return [*lines, "No rows.", ""]
    lines.append("| city | source | offset | n | mean prob delta | mean abs prob delta | brier delta |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.city} | {row.source} | {_fmt(row.offset_f)} | {row.n_days} | "
            f"{_fmt(row.mean_prob_delta)} | {_fmt(row.mean_abs_prob_delta)} | "
            f"{_fmt(row.brier_delta)} |"
        )
    lines.append("")
    return lines


def _fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Leakage-safe climate/trend feature diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "city",
    "target_date",
    "day_of_year",
    "day_of_year_sin",
    "day_of_year_cos",
    "actual_high_f",
    "historical_normal_f",
    "rolling_30d_actual_f",
    "rolling_90d_actual_f",
    "rolling_180d_actual_f",
    "rolling_30d_anomaly_f",
    "rolling_90d_anomaly_f",
    "rolling_180d_anomaly_f",
    "yearly_city_trend_f",
    "recent_warming_anomaly_f",
    "climatology_error_baseline_f",
    "feature_missing",
]
SUMMARY_COLUMNS = [
    "city",
    "n_days",
    "normal_available_rate",
    "rolling_30d_available_rate",
    "rolling_90d_available_rate",
    "rolling_180d_available_rate",
    "trend_available_rate",
    "mean_recent_warming_anomaly_f",
    "climatology_mae",
    "climatology_bias",
]


@dataclass(frozen=True)
class ClimateFeatureDiagnostics:
    """Climate/trend feature rows and city summary."""

    features: pd.DataFrame
    summary: pd.DataFrame
    report: str


def build_climate_feature_diagnostics(rows: pd.DataFrame) -> ClimateFeatureDiagnostics:
    """Build leakage-safe climate/trend features from actual high rows.

    Every feature for a target date is computed from rows strictly before that
    date. The actual high remains in the output for diagnostics only.
    """
    daily = _daily_actuals(rows)
    feature_rows = []
    for city, group in daily.groupby("city", sort=True):
        group = group.sort_values("target_date").reset_index(drop=True)
        for index, row in group.iterrows():
            prior = group.iloc[:index]
            actual = float(row["actual_high_f"])
            day_of_year = int(row["target_date"].dayofyear)
            normal = _historical_normal(prior, day_of_year)
            rolling_30 = _rolling_mean(prior, 30)
            rolling_90 = _rolling_mean(prior, 90)
            rolling_180 = _rolling_mean(prior, 180)
            trend = _yearly_trend(prior)
            recent_warming = (
                rolling_30 - rolling_180
                if not pd.isna(rolling_30) and not pd.isna(rolling_180)
                else pd.NA
            )
            normal_missing = pd.isna(normal)
            feature_rows.append(
                {
                    "city": city,
                    "target_date": row["target_date"].date().isoformat(),
                    "day_of_year": day_of_year,
                    "day_of_year_sin": math.sin(2 * math.pi * day_of_year / 366),
                    "day_of_year_cos": math.cos(2 * math.pi * day_of_year / 366),
                    "actual_high_f": actual,
                    "historical_normal_f": normal,
                    "rolling_30d_actual_f": rolling_30,
                    "rolling_90d_actual_f": rolling_90,
                    "rolling_180d_actual_f": rolling_180,
                    "rolling_30d_anomaly_f": actual - rolling_30 if not pd.isna(rolling_30) else pd.NA,
                    "rolling_90d_anomaly_f": actual - rolling_90 if not pd.isna(rolling_90) else pd.NA,
                    "rolling_180d_anomaly_f": actual - rolling_180 if not pd.isna(rolling_180) else pd.NA,
                    "yearly_city_trend_f": trend,
                    "recent_warming_anomaly_f": recent_warming,
                    "climatology_error_baseline_f": actual - normal if not normal_missing else pd.NA,
                    "feature_missing": bool(
                        normal_missing
                        or pd.isna(rolling_30)
                        or pd.isna(rolling_90)
                        or pd.isna(rolling_180)
                        or pd.isna(trend)
                    ),
                }
            )
    features = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    summary = summarize_climate_features(features)
    return ClimateFeatureDiagnostics(
        features=features,
        summary=summary,
        report=render_climate_feature_report(summary),
    )


def summarize_climate_features(features: pd.DataFrame) -> pd.DataFrame:
    """Summarize climate feature availability and baseline error by city."""
    if features.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows = []
    for city, group in features.groupby("city", sort=True):
        normal_error = group["climatology_error_baseline_f"].dropna().astype(float)
        rows.append(
            {
                "city": city,
                "n_days": int(len(group)),
                "normal_available_rate": float(group["historical_normal_f"].notna().mean()),
                "rolling_30d_available_rate": float(group["rolling_30d_actual_f"].notna().mean()),
                "rolling_90d_available_rate": float(group["rolling_90d_actual_f"].notna().mean()),
                "rolling_180d_available_rate": float(group["rolling_180d_actual_f"].notna().mean()),
                "trend_available_rate": float(group["yearly_city_trend_f"].notna().mean()),
                "mean_recent_warming_anomaly_f": _mean_or_na(group["recent_warming_anomaly_f"]),
                "climatology_mae": float(normal_error.abs().mean()) if not normal_error.empty else pd.NA,
                "climatology_bias": float(normal_error.mean()) if not normal_error.empty else pd.NA,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def render_climate_feature_report(summary: pd.DataFrame) -> str:
    """Render climate feature diagnostic markdown."""
    lines = [
        "# Climate Feature Diagnostics",
        "",
        "These features are leakage-safe: each target date uses only prior actual highs. This report does not claim climate/trend features improve forecasts unless walk-forward metrics prove it.",
        "",
    ]
    if summary.empty:
        return "\n".join([*lines, "No rows.", ""])
    lines.append("| city | n | normal rate | 180d rate | trend rate | climatology MAE | recent warming |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.city} | {row.n_days} | {_fmt(row.normal_available_rate)} | "
            f"{_fmt(row.rolling_180d_available_rate)} | {_fmt(row.trend_available_rate)} | "
            f"{_fmt(row.climatology_mae)} | {_fmt(row.mean_recent_warming_anomaly_f)} |"
        )
    lines.extend(
        [
            "",
            "Use this as an audit input only. Climate trend is a slow baseline adjustment, not a daily trading signal.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_climate_feature_diagnostics(*, input_path: Path, output_dir: Path) -> ClimateFeatureDiagnostics:
    """Read rows and write climate feature diagnostics."""
    diagnostics = build_climate_feature_diagnostics(pd.read_csv(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.features.to_csv(output_dir / "climate_features.csv", index=False)
    diagnostics.summary.to_csv(output_dir / "climate_feature_summary.csv", index=False)
    (output_dir / "climate_feature_diagnostics.md").write_text(
        diagnostics.report,
        encoding="utf-8",
    )
    return diagnostics


def _daily_actuals(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"city", "target_date", "actual_high_f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"rows missing required columns: {sorted(missing)}")
    daily = rows.loc[:, list(required)].copy()
    daily["city"] = daily["city"].astype(str)
    daily["target_date"] = pd.to_datetime(daily["target_date"], errors="coerce")
    daily["actual_high_f"] = pd.to_numeric(daily["actual_high_f"], errors="coerce")
    daily = daily.dropna(subset=["target_date", "actual_high_f"])
    return (
        daily.groupby(["city", "target_date"], as_index=False)["actual_high_f"]
        .mean()
        .sort_values(["city", "target_date"])
    )


def _historical_normal(prior: pd.DataFrame, day_of_year: int, window: int = 15) -> object:
    if prior.empty:
        return pd.NA
    distance = (prior["target_date"].dt.dayofyear - day_of_year).abs()
    distance = distance.where(distance <= 183, 366 - distance)
    matches = prior[distance <= window]
    if matches.empty:
        return pd.NA
    return float(matches["actual_high_f"].mean())


def _rolling_mean(prior: pd.DataFrame, days: int) -> object:
    if len(prior) < days:
        return pd.NA
    return float(prior.tail(days)["actual_high_f"].mean())


def _yearly_trend(prior: pd.DataFrame, min_days: int = 365) -> object:
    if len(prior) < min_days:
        return pd.NA
    x = (prior["target_date"] - prior["target_date"].min()).dt.days.astype(float) / 365.25
    y = prior["actual_high_f"].astype(float)
    if float(x.var()) == 0:
        return pd.NA
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def _mean_or_na(values: pd.Series) -> object:
    valid = values.dropna().astype(float)
    if valid.empty:
        return pd.NA
    return float(valid.mean())


def _fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"

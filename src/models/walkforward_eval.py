"""Leakage-safe walk-forward evaluation for source policies."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.models.source_contrarian_diagnostics import PRIMARY_SOURCES

DEFAULT_THRESHOLDS = (-6, -4, -2, 0, 2, 4, 6)
PREDICTION_COLUMNS = [
    "window_start",
    "window_end",
    "train_start",
    "train_end",
    "city",
    "source",
    "target_date",
    "point_f",
    "bias_correction_f",
    "corrected_point_f",
    "actual_high_f",
    "error_f",
    "absolute_error_f",
    "interval_lower_f",
    "interval_upper_f",
    "interval_covered",
]
EVENT_COLUMNS = [
    "window_start",
    "window_end",
    "city",
    "source",
    "target_date",
    "threshold_f",
    "offset_f",
    "predicted_probability",
    "recalibrated_probability",
    "outcome",
    "actual_high_f",
    "corrected_point_f",
]
SUMMARY_COLUMNS = [
    "city",
    "source",
    "n_predictions",
    "n_events",
    "mae",
    "bias",
    "coverage_80",
    "brier_raw",
    "brier_recal",
    "ece_raw",
    "ece_recal",
    "logloss_raw",
    "logloss_recal",
    "worst_window_mae",
    "best_window_mae",
    "stability_score",
]


@dataclass(frozen=True)
class WalkforwardResult:
    """Artifacts from walk-forward evaluation."""

    events: pd.DataFrame
    predictions: pd.DataFrame
    window_summary: pd.DataFrame
    city_source_summary: pd.DataFrame
    threshold_summary: pd.DataFrame
    policy_leaderboard: pd.DataFrame
    report: str
    manifest: dict[str, object]


def evaluate_walkforward(
    rows: pd.DataFrame,
    *,
    cities: Iterable[str],
    sources: Iterable[str],
    train_window_days: int,
    test_window_days: int,
    step_days: int,
    threshold_offsets: Iterable[int | float] = DEFAULT_THRESHOLDS,
    holdout_start: str | None = None,
    holdout_end: str | None = None,
    purge_days_before: int = 0,
    purge_days_after: int = 0,
    input_path: str | None = None,
    input_sha256: str | None = None,
    command_args: dict[str, object] | None = None,
    git_commit: str | None = None,
) -> WalkforwardResult:
    """Evaluate policies with rolling train windows and future-only tests."""
    if train_window_days <= 0 or test_window_days <= 0 or step_days <= 0:
        raise ValueError("train, test, and step windows must be positive")
    offsets = tuple(float(offset) for offset in threshold_offsets)
    if not offsets:
        raise ValueError("threshold_offsets must contain at least one value")

    clean = _clean_rows(rows)
    city_values = tuple(str(city) for city in cities)
    source_values = tuple(str(source) for source in sources)
    clean = clean[clean["city"].isin(city_values)]
    if clean.empty:
        raise ValueError("no rows matched requested cities")

    holdout = _holdout_window(holdout_start, holdout_end, purge_days_before, purge_days_after)
    all_predictions: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    min_date = clean["target_date"].min().normalize()
    max_date = clean["target_date"].max().normalize()
    test_start = min_date + pd.Timedelta(days=train_window_days)
    while test_start <= max_date:
        train_start = test_start - pd.Timedelta(days=train_window_days)
        test_end = test_start + pd.Timedelta(days=test_window_days)
        train = clean[(clean["target_date"] >= train_start) & (clean["target_date"] < test_start)]
        test = clean[(clean["target_date"] >= test_start) & (clean["target_date"] < test_end)]
        if holdout is not None:
            train = _exclude_holdout(train, holdout)
            test = _exclude_holdout(test, holdout)

        for city in city_values:
            city_train = train[train["city"] == city]
            city_test = test[test["city"] == city]
            if city_train.empty or city_test.empty:
                continue
            for source in source_values:
                train_policy = _policy_rows(city_train, source, weight_train=city_train)
                test_policy = _policy_rows(city_test, source, weight_train=city_train)
                if len(train_policy) < 14 or test_policy.empty:
                    continue
                predictions, events = _evaluate_policy_window(
                    train_policy=train_policy,
                    test_policy=test_policy,
                    offsets=offsets,
                    window_start=test_start,
                    window_end=test_end - pd.Timedelta(days=1),
                    train_start=train_start,
                    train_end=test_start - pd.Timedelta(days=1),
                    city=city,
                    source=source,
                )
                all_predictions.append(predictions)
                all_events.append(events)
        test_start += pd.Timedelta(days=step_days)

    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    events = (
        pd.concat(all_events, ignore_index=True)
        if all_events
        else pd.DataFrame(columns=EVENT_COLUMNS)
    )
    window_summary = _window_summary(predictions, events)
    city_source_summary = _city_source_summary(predictions, events)
    threshold_summary = _threshold_summary(events)
    policy_leaderboard = _policy_leaderboard(city_source_summary)
    report = render_walkforward_report(policy_leaderboard, city_source_summary, window_summary)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "input_path": input_path,
        "input_sha256": input_sha256,
        "command_args": command_args or {},
        "row_counts": {
            "input_rows": int(len(rows)),
            "clean_rows": int(len(clean)),
            "walkforward_predictions": int(len(predictions)),
            "walkforward_events": int(len(events)),
            "window_summary": int(len(window_summary)),
            "city_source_summary": int(len(city_source_summary)),
        },
    }
    return WalkforwardResult(
        events=events,
        predictions=predictions,
        window_summary=window_summary,
        city_source_summary=city_source_summary,
        threshold_summary=threshold_summary,
        policy_leaderboard=policy_leaderboard,
        report=report,
        manifest=manifest,
    )


def write_walkforward_outputs(
    *,
    rows_path: Path,
    output_dir: Path,
    cities: Iterable[str],
    sources: Iterable[str],
    train_window_days: int,
    test_window_days: int,
    step_days: int,
    threshold_offsets: Iterable[int | float] = DEFAULT_THRESHOLDS,
    holdout_start: str | None = None,
    holdout_end: str | None = None,
    purge_days_before: int = 0,
    purge_days_after: int = 0,
    command_args: dict[str, object] | None = None,
    git_commit: str | None = None,
) -> WalkforwardResult:
    """Read rows and write walk-forward artifacts."""
    rows = pd.read_csv(rows_path)
    result = evaluate_walkforward(
        rows,
        cities=cities,
        sources=sources,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        step_days=step_days,
        threshold_offsets=threshold_offsets,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        purge_days_before=purge_days_before,
        purge_days_after=purge_days_after,
        input_path=str(rows_path),
        input_sha256=_sha256(rows_path),
        command_args=command_args,
        git_commit=git_commit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.events.to_csv(output_dir / "walkforward_events.csv", index=False)
    result.predictions.to_csv(output_dir / "walkforward_predictions.csv", index=False)
    result.window_summary.to_csv(output_dir / "walkforward_window_summary.csv", index=False)
    result.city_source_summary.to_csv(
        output_dir / "walkforward_city_source_summary.csv",
        index=False,
    )
    result.threshold_summary.to_csv(output_dir / "walkforward_threshold_summary.csv", index=False)
    result.policy_leaderboard.to_csv(output_dir / "walkforward_policy_leaderboard.csv", index=False)
    (output_dir / "walkforward_report.md").write_text(result.report, encoding="utf-8")
    (output_dir / "walkforward_manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def render_walkforward_report(
    policy_leaderboard: pd.DataFrame,
    city_source_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> str:
    """Render a compact markdown report."""
    lines = [
        "# Walk-Forward Model Evaluation",
        "",
        "This report is leakage-safe by construction: each test window is calibrated only from rows before the test start.",
        "",
        "## Policy Leaderboard",
        "",
    ]
    lines.extend(_leaderboard_table(policy_leaderboard.head(12)))
    lines.extend(["## Worst City/Source Rows", ""])
    worst = city_source_summary.sort_values(["mae", "brier_raw"], ascending=[False, False]).head(12)
    lines.extend(_leaderboard_table(worst))
    lines.extend(["## Worst Windows", ""])
    lines.extend(_window_table(window_summary.sort_values("mae", ascending=False).head(12)))
    return "\n".join(lines) + "\n"


def _clean_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"city", "target_date", "source", "point_f", "actual_high_f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"rows missing required columns: {sorted(missing)}")
    clean = rows.loc[:, list(required)].copy()
    clean["city"] = clean["city"].astype(str)
    clean["source"] = clean["source"].astype(str)
    clean["target_date"] = pd.to_datetime(clean["target_date"], errors="coerce")
    clean["point_f"] = pd.to_numeric(clean["point_f"], errors="coerce")
    clean["actual_high_f"] = pd.to_numeric(clean["actual_high_f"], errors="coerce")
    return clean.dropna(subset=["target_date", "point_f", "actual_high_f"])


def _policy_rows(rows: pd.DataFrame, source: str, *, weight_train: pd.DataFrame) -> pd.DataFrame:
    if source in set(rows["source"]):
        out = rows[rows["source"] == source].copy()
        return out.sort_values("target_date")
    if source == "blend_equal":
        return _blend_rows(rows, weights=None)
    if source == "blend_mae_90d":
        weights = _recent_mae_weights(weight_train)
        return _blend_rows(rows, weights=weights)
    return pd.DataFrame(columns=rows.columns)


def _blend_rows(rows: pd.DataFrame, weights: dict[str, float] | None) -> pd.DataFrame:
    primary = rows[rows["source"].isin(PRIMARY_SOURCES)].copy()
    output_rows = []
    for (city, target_date), group in primary.groupby(["city", "target_date"], sort=True):
        if group.empty:
            continue
        if weights:
            available_weights = group["source"].map(weights).fillna(0.0).astype(float)
            if available_weights.sum() <= 0:
                point = float(group["point_f"].astype(float).mean())
            else:
                point = float((group["point_f"].astype(float) * available_weights).sum() / available_weights.sum())
        else:
            point = float(group["point_f"].astype(float).mean())
        output_rows.append(
            {
                "city": city,
                "target_date": target_date,
                "source": "blend",
                "point_f": point,
                "actual_high_f": float(group["actual_high_f"].iloc[0]),
            }
        )
    return pd.DataFrame(output_rows)


def _recent_mae_weights(train_rows: pd.DataFrame, days: int = 90) -> dict[str, float]:
    if train_rows.empty:
        return {}
    max_date = train_rows["target_date"].max()
    recent = train_rows[
        (train_rows["target_date"] >= max_date - pd.Timedelta(days=days))
        & train_rows["source"].isin(PRIMARY_SOURCES)
    ].copy()
    if recent.empty:
        return {}
    recent["absolute_error_f"] = (
        recent["actual_high_f"].astype(float) - recent["point_f"].astype(float)
    ).abs()
    maes = recent.groupby("source")["absolute_error_f"].mean()
    raw = {source: 1.0 / max(float(mae), 0.01) for source, mae in maes.items()}
    total = sum(raw.values())
    return {source: value / total for source, value in raw.items()} if total else {}


def _evaluate_policy_window(
    *,
    train_policy: pd.DataFrame,
    test_policy: pd.DataFrame,
    offsets: tuple[float, ...],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    city: str,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bias = float((train_policy["actual_high_f"] - train_policy["point_f"]).mean())
    train_residuals = train_policy["actual_high_f"].astype(float) - (
        train_policy["point_f"].astype(float) + bias
    )
    lower = float(train_residuals.quantile(0.1))
    upper = float(train_residuals.quantile(0.9))
    prediction_rows = []
    event_rows = []
    for row in test_policy.itertuples(index=False):
        corrected = float(row.point_f) + bias
        actual = float(row.actual_high_f)
        error = actual - corrected
        interval_lower = corrected + lower
        interval_upper = corrected + upper
        prediction_rows.append(
            {
                "window_start": window_start.date().isoformat(),
                "window_end": window_end.date().isoformat(),
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "city": city,
                "source": source,
                "target_date": row.target_date.date().isoformat(),
                "point_f": float(row.point_f),
                "bias_correction_f": bias,
                "corrected_point_f": corrected,
                "actual_high_f": actual,
                "error_f": error,
                "absolute_error_f": abs(error),
                "interval_lower_f": interval_lower,
                "interval_upper_f": interval_upper,
                "interval_covered": bool(interval_lower <= actual <= interval_upper),
            }
        )
        for offset in offsets:
            threshold = round(corrected) + offset
            needed_residual = threshold - corrected
            probability = float((train_residuals >= needed_residual).mean())
            outcome = bool(actual >= threshold)
            event_rows.append(
                {
                    "window_start": window_start.date().isoformat(),
                    "window_end": window_end.date().isoformat(),
                    "city": city,
                    "source": source,
                    "target_date": row.target_date.date().isoformat(),
                    "threshold_f": threshold,
                    "offset_f": offset,
                    "predicted_probability": probability,
                    "recalibrated_probability": probability,
                    "outcome": outcome,
                    "actual_high_f": actual,
                    "corrected_point_f": corrected,
                }
            )
    return (
        pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS),
        pd.DataFrame(event_rows, columns=EVENT_COLUMNS),
    )


def _window_summary(predictions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for (window_start, source), group in predictions.groupby(["window_start", "source"], sort=True):
        event_group = events[(events["window_start"] == window_start) & (events["source"] == source)]
        rows.append({"window_start": window_start, "source": source, **_summary_metrics(group, event_group)})
    return pd.DataFrame(rows)


def _city_source_summary(predictions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows = []
    for (city, source), group in predictions.groupby(["city", "source"], sort=True):
        event_group = events[(events["city"] == city) & (events["source"] == source)]
        rows.append({"city": city, "source": source, **_summary_metrics(group, event_group)})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _threshold_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for (source, offset), group in events.groupby(["source", "offset_f"], sort=True):
        rows.append(
            {
                "source": source,
                "offset_f": offset,
                "n_events": int(len(group)),
                "brier_raw": _brier(group["predicted_probability"], group["outcome"]),
                "brier_recal": _brier(group["recalibrated_probability"], group["outcome"]),
                "observed_frequency": float(group["outcome"].astype(float).mean()),
                "mean_predicted_probability": float(group["predicted_probability"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _policy_leaderboard(city_source_summary: pd.DataFrame) -> pd.DataFrame:
    if city_source_summary.empty:
        return city_source_summary
    rows = []
    for source, group in city_source_summary.groupby("source", sort=True):
        rows.append(
            {
                "source": source,
                "n_cities": int(group["city"].nunique()),
                "n_predictions": int(group["n_predictions"].sum()),
                "n_events": int(group["n_events"].sum()),
                "mae": float(group["mae"].mean()),
                "bias": float(group["bias"].mean()),
                "brier_raw": float(group["brier_raw"].mean()),
                "brier_recal": float(group["brier_recal"].mean()),
                "ece_raw": float(group["ece_raw"].mean()),
                "ece_recal": float(group["ece_recal"].mean()),
                "logloss_raw": float(group["logloss_raw"].mean()),
                "logloss_recal": float(group["logloss_recal"].mean()),
                "worst_city": group.sort_values("mae", ascending=False).iloc[0]["city"],
                "worst_city_mae": float(group["mae"].max()),
                "stability_score": float(group["stability_score"].mean()),
                "leakage_safe": True,
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "brier_raw", "stability_score"])


def _summary_metrics(predictions: pd.DataFrame, events: pd.DataFrame) -> dict[str, object]:
    window_mae = predictions.groupby("window_start")["absolute_error_f"].mean()
    return {
        "n_predictions": int(len(predictions)),
        "n_events": int(len(events)),
        "mae": float(predictions["absolute_error_f"].mean()),
        "bias": float(predictions["error_f"].mean()),
        "coverage_80": float(predictions["interval_covered"].astype(float).mean()),
        "brier_raw": _brier(events["predicted_probability"], events["outcome"]) if not events.empty else pd.NA,
        "brier_recal": _brier(events["recalibrated_probability"], events["outcome"]) if not events.empty else pd.NA,
        "ece_raw": _ece(events["predicted_probability"], events["outcome"]) if not events.empty else pd.NA,
        "ece_recal": _ece(events["recalibrated_probability"], events["outcome"]) if not events.empty else pd.NA,
        "logloss_raw": _logloss(events["predicted_probability"], events["outcome"]) if not events.empty else pd.NA,
        "logloss_recal": _logloss(events["recalibrated_probability"], events["outcome"]) if not events.empty else pd.NA,
        "worst_window_mae": float(window_mae.max()),
        "best_window_mae": float(window_mae.min()),
        "stability_score": float(window_mae.std(ddof=0)) if len(window_mae) > 1 else 0.0,
    }


def _brier(probabilities: pd.Series, outcomes: pd.Series) -> float:
    return float(((probabilities.astype(float) - outcomes.astype(float)) ** 2).mean())


def _ece(probabilities: pd.Series, outcomes: pd.Series, buckets: int = 10) -> float:
    df = pd.DataFrame({"p": probabilities.astype(float), "y": outcomes.astype(float)})
    df["bucket"] = (df["p"] * buckets).clip(upper=buckets - 1e-9).astype(int)
    total = len(df)
    gap = 0.0
    for _, group in df.groupby("bucket"):
        gap += len(group) / total * abs(float(group["p"].mean()) - float(group["y"].mean()))
    return float(gap)


def _logloss(probabilities: pd.Series, outcomes: pd.Series) -> float:
    p = probabilities.astype(float).clip(1e-6, 1 - 1e-6)
    y = outcomes.astype(float)
    return float((-(y * p.map(math.log) + (1 - y) * (1 - p).map(math.log))).mean())


def _holdout_window(
    holdout_start: str | None,
    holdout_end: str | None,
    purge_days_before: int,
    purge_days_after: int,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if not holdout_start and not holdout_end:
        return None
    if not holdout_start or not holdout_end:
        raise ValueError("holdout-start and holdout-end must be provided together")
    start = pd.Timestamp(holdout_start) - pd.Timedelta(days=purge_days_before)
    end = pd.Timestamp(holdout_end) + pd.Timedelta(days=purge_days_after)
    return start, end


def _exclude_holdout(rows: pd.DataFrame, holdout: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    start, end = holdout
    return rows[(rows["target_date"] < start) | (rows["target_date"] > end)]


def _leaderboard_table(rows: pd.DataFrame) -> list[str]:
    if rows.empty:
        return ["No rows.", ""]
    wanted = [
        column
        for column in ["source", "city", "n_predictions", "mae", "bias", "brier_raw", "stability_score"]
        if column in rows.columns
    ]
    lines = ["| " + " | ".join(wanted) + " |", "|" + "|".join("---" for _ in wanted) + "|"]
    for row in rows.loc[:, wanted].itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    lines.append("")
    return lines


def _window_table(rows: pd.DataFrame) -> list[str]:
    if rows.empty:
        return ["No rows.", ""]
    wanted = [column for column in ["window_start", "source", "n_predictions", "mae", "bias"] if column in rows.columns]
    lines = ["| " + " | ".join(wanted) + " |", "|" + "|".join("---" for _ in wanted) + "|"]
    for row in rows.loc[:, wanted].itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    lines.append("")
    return lines


def _fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

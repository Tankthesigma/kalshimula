"""CLI: `python -m src.predict --city denver --date tomorrow`.

Milestone A: pulls every Open-Meteo source, pools members, renders an ASCII
histogram + point estimate + 80% CI. NWS comparison and bias correction come
in Milestone B.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import Station, get_station, load_stations
from src.fetchers.openmeteo import (
    SOURCES,
    ModelDailyHigh,
    fetch_source,
    members_dataframe,
)
from src.models.bias import apply_bias_correction, fit_bias_table
from src.models.ensemble import NaiveForecast, naive_forecast_from_members
from src.models.intervals import apply_empirical_intervals, fit_empirical_intervals
from src.models.threshold_calibration import GLOBAL_RECALIBRATION_KEY

PREDICTION_JSON_SCHEMA_VERSION = "1.0"
MULTI_SOURCE_MODES = ("single", "blend_equal", "blend_mae_90d")
MULTI_SOURCE_ARTIFACT_SOURCE = "openmeteo_naive"
MULTI_SOURCE_PRIMARY_SOURCES = (
    "gfs_ens",
    "ecmwf_ens",
    "icon_ens",
    "gem_ens",
    "aifs",
)

# Windows default console is cp1252 and chokes on box-drawing glyphs.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        with contextlib.suppress(Exception):
            _stream.reconfigure(encoding="utf-8")


def _parse_date(s: str) -> date:
    s = s.strip().lower()
    today = date.today()
    if s in {"today", "t"}:
        return today
    if s in {"tomorrow", "tmrw"}:
        return today + timedelta(days=1)
    if s in {"yesterday", "y"}:
        return today - timedelta(days=1)
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fetch_all_parallel(
    station: Station, target: date, *, use_historical: bool
) -> list[ModelDailyHigh]:
    """Hit every Open-Meteo source concurrently. Failures degrade to empty."""
    results: list[ModelDailyHigh] = []
    with ThreadPoolExecutor(max_workers=len(SOURCES)) as ex:
        futures = {
            ex.submit(
                fetch_source,
                slug,
                lat=station.lat,
                lon=station.lon,
                target=target,
                use_historical=use_historical,
            ): slug
            for slug, *_ in SOURCES
        }
        for fut in as_completed(futures):
            slug = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"  ! {slug} failed: {e}", file=sys.stderr)
                results.append(
                    ModelDailyHigh(source=slug, target_date=target, members_f=[])
                )
    # Stable order by SOURCES definition for nicer output.
    order = [s[0] for s in SOURCES]
    results.sort(key=lambda r: order.index(r.source))
    return results


def _load_selected_source(path: Path, city: str) -> str | None:
    """Return the selected source for a city from source_selection output."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {"city", "selected_source"} - fieldnames
        if missing:
            raise ValueError(f"selected sources CSV missing columns: {sorted(missing)}")

        city_key = city.strip().lower()
        rows = list(reader)
        global_sources = {
            row.get("selected_source", "").strip()
            for row in rows
            if row.get("selected_source", "").strip()
        }
        for row in rows:
            if row.get("city", "").strip().lower() == city_key:
                selected = row.get("selected_source", "").strip()
                return selected or None
        if (
            "recommended_policy" in fieldnames
            and len(global_sources) == 1
            and any(
                row.get("recommended_policy", "").strip()
                == "best_global_validation_source"
                for row in rows
            )
        ):
            return next(iter(global_sources))
    return None


def _members_for_selected_source(
    members: pd.DataFrame, selected_source: str | None
) -> tuple[pd.DataFrame, bool]:
    """Filter members to one selected source when possible.

    ``openmeteo_naive`` is the historical pooled baseline, so live prediction
    represents it by keeping all member rows.
    """
    if not selected_source or selected_source == "openmeteo_naive":
        return members, False

    selected = members[members["source"] == selected_source]
    if selected.empty:
        return members, False
    return selected, True


def _prediction_source(selected_source: str | None, *, selected_applied: bool) -> str:
    if selected_applied and selected_source:
        return selected_source
    return "openmeteo_naive"


def _members_for_primary_multi_source(members: pd.DataFrame) -> pd.DataFrame:
    """Keep the core global sources used for multi-source live blends."""
    selected = members[members["source"].isin(MULTI_SOURCE_PRIMARY_SOURCES)].copy()
    return selected if not selected.empty else members.copy()


def _weighted_percentile(values: pd.Series, weights: pd.Series, q: float) -> float:
    ordered = (
        pd.DataFrame({"value": values.astype(float), "weight": weights.astype(float)})
        .sort_values("value")
        .reset_index(drop=True)
    )
    total = float(ordered["weight"].sum())
    if total <= 0:
        raise ValueError("multi-source weights must sum to a positive value")
    cutoff = q * total
    cumulative = ordered["weight"].cumsum()
    index = int(cumulative.searchsorted(cutoff, side="left"))
    index = min(max(index, 0), len(ordered) - 1)
    return float(ordered.iloc[index]["value"])


def _forecast_from_weighted_members(
    members: pd.DataFrame, weights: pd.Series, *, bin_min_prob: float = 0.005
) -> NaiveForecast:
    """Build a NaiveForecast from per-member weights."""
    if members.empty:
        raise ValueError("No forecast members returned — every source failed?")
    if len(members) != len(weights):
        raise ValueError("member and weight lengths differ")

    temps = members["temp_f"].astype(float).reset_index(drop=True)
    weights = weights.astype(float).reset_index(drop=True)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("multi-source weights must sum to a positive value")
    weights = weights / total_weight

    point = float((temps * weights).sum())
    p10 = _weighted_percentile(temps, weights, 0.10)
    p50 = _weighted_percentile(temps, weights, 0.50)
    p90 = _weighted_percentile(temps, weights, 0.90)

    bins = temps.map(lambda value: int(math.floor(float(value) + 0.5)))
    raw_probs = weights.groupby(bins).sum().to_dict()
    kept = {int(bin_f): float(prob) for bin_f, prob in raw_probs.items() if prob >= bin_min_prob}
    if kept:
        kept_total = sum(kept.values())
        if kept_total > 0:
            kept = {bin_f: prob / kept_total for bin_f, prob in kept.items()}
    else:
        kept = {int(bin_f): float(prob) for bin_f, prob in raw_probs.items()}

    return NaiveForecast(
        n_members=len(members),
        point_f=point,
        p10_f=p10,
        p50_f=p50,
        p90_f=p90,
        bin_probs=dict(sorted(kept.items())),
        per_source_counts=members.groupby("source").size().to_dict(),
    )


def _equal_source_member_weights(members: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    counts = members.groupby("source").size()
    sources = sorted(counts.index.astype(str).tolist())
    if not sources:
        raise ValueError("No forecast members returned — every source failed?")
    source_weight = 1.0 / len(sources)
    weights = members["source"].map(
        {source: source_weight / float(counts[source]) for source in sources}
    )
    return weights.astype(float), {source: source_weight for source in sources}


def _recent_source_mae_weights(
    *,
    rows_path: Path,
    city: str,
    target: date,
    sources: set[str],
    lookback_days: int = 90,
) -> tuple[dict[str, float], dict[str, float]]:
    rows = pd.read_csv(
        rows_path,
        usecols=["city", "source", "target_date", "point_f", "actual_high_f"],
    )
    rows["target_date"] = pd.to_datetime(rows["target_date"], errors="coerce").dt.date
    rows["point_f"] = pd.to_numeric(rows["point_f"], errors="coerce")
    rows["actual_high_f"] = pd.to_numeric(rows["actual_high_f"], errors="coerce")
    start = target - timedelta(days=lookback_days)
    filtered = rows[
        (rows["city"].astype(str).str.lower() == city.lower())
        & (rows["source"].astype(str).isin(sources))
        & (rows["target_date"] >= start)
        & (rows["target_date"] < target)
    ].dropna(subset=["target_date", "point_f", "actual_high_f"])
    if filtered.empty:
        return {}, {}

    filtered = filtered.assign(
        absolute_error_f=(filtered["point_f"] - filtered["actual_high_f"]).abs()
    )
    mae_by_source = filtered.groupby("source")["absolute_error_f"].mean().to_dict()
    if not mae_by_source:
        return {}, {}
    fallback_mae = float(pd.Series(mae_by_source, dtype=float).median())
    inverse_scores = {
        source: 1.0 / max(float(mae_by_source.get(source, fallback_mae)), 0.1)
        for source in sources
    }
    total = sum(inverse_scores.values())
    if total <= 0:
        return {}, mae_by_source
    return {
        source: score / total for source, score in inverse_scores.items()
    }, {source: float(value) for source, value in mae_by_source.items()}


def _historical_multi_source_rows(
    *,
    rows_path: Path,
    city: str,
    target: date,
    source_weights: dict[str, float],
    lookback_days: int = 90,
) -> pd.DataFrame:
    if not source_weights:
        return pd.DataFrame(
            columns=["city", "source", "target_date", "point_f", "actual_high_f"]
        )
    rows = pd.read_csv(
        rows_path,
        usecols=["city", "source", "target_date", "point_f", "actual_high_f"],
    )
    rows["target_date"] = pd.to_datetime(rows["target_date"], errors="coerce").dt.date
    rows["point_f"] = pd.to_numeric(rows["point_f"], errors="coerce")
    rows["actual_high_f"] = pd.to_numeric(rows["actual_high_f"], errors="coerce")
    start = target - timedelta(days=lookback_days)
    filtered = rows[
        (rows["city"].astype(str).str.lower() == city.lower())
        & (rows["source"].astype(str).isin(source_weights))
        & (rows["target_date"] >= start)
        & (rows["target_date"] < target)
    ].dropna(subset=["target_date", "point_f", "actual_high_f"])
    if filtered.empty:
        return pd.DataFrame(
            columns=["city", "source", "target_date", "point_f", "actual_high_f"]
        )

    records = []
    for target_date, group in filtered.groupby("target_date", sort=True):
        available_weights = {
            str(row.source): float(source_weights[str(row.source)])
            for row in group.itertuples(index=False)
            if str(row.source) in source_weights
        }
        total_weight = sum(available_weights.values())
        if total_weight <= 0:
            continue
        point = 0.0
        for row in group.itertuples(index=False):
            source = str(row.source)
            if source in available_weights:
                point += float(row.point_f) * available_weights[source] / total_weight
        records.append(
            {
                "city": city,
                "source": MULTI_SOURCE_ARTIFACT_SOURCE,
                "target_date": target_date,
                "point_f": point,
                "actual_high_f": float(group["actual_high_f"].mean()),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["city", "source", "target_date", "point_f", "actual_high_f"],
    )


def _multi_source_forecast(
    *,
    members: pd.DataFrame,
    mode: str,
    city: str,
    target: date,
    model_run_dir: Path | None = None,
) -> tuple[NaiveForecast, dict, list[str]]:
    """Build an additive multi-source live blend forecast."""
    if mode not in MULTI_SOURCE_MODES or mode == "single":
        raise ValueError(f"unsupported multi-source mode: {mode}")

    blend_members = _members_for_primary_multi_source(members)
    warnings: list[str] = []
    sources = set(blend_members["source"].astype(str).unique())

    if mode == "blend_equal":
        weights, source_weights = _equal_source_member_weights(blend_members)
        mae_by_source: dict[str, float] = {}
    else:
        source_weights = {}
        mae_by_source = {}
        if model_run_dir is not None:
            rows_path = model_run_dir / "rows.csv"
            if rows_path.exists():
                source_weights, mae_by_source = _recent_source_mae_weights(
                    rows_path=rows_path,
                    city=city,
                    target=target,
                    sources=sources,
                )
        if not source_weights:
            warnings.append(
                "no recent 90d source MAE rows available; using equal source weights"
            )
            weights, source_weights = _equal_source_member_weights(blend_members)
        else:
            source_counts = blend_members.groupby("source").size()
            weights = blend_members["source"].map(
                {
                    source: source_weights[source] / float(source_counts[source])
                    for source in source_weights
                }
            ).astype(float)

    forecast = _forecast_from_weighted_members(blend_members, weights)
    metadata = {
        "mode": mode,
        "artifact_source": MULTI_SOURCE_ARTIFACT_SOURCE,
        "source_weights": {key: float(source_weights[key]) for key in sorted(source_weights)},
        "recent_90d_mae_f": {key: float(mae_by_source[key]) for key in sorted(mae_by_source)},
    }
    return forecast, metadata, warnings


def _has_city_source(table: pd.DataFrame, *, city: str, source: str) -> bool:
    required = {"city", "source"}
    if required - set(table.columns):
        return False
    city_values = table["city"].astype(str).str.lower()
    source_values = table["source"].astype(str)
    return bool(((city_values == city.lower()) & (source_values == source)).any())


def _apply_prediction_artifacts(
    *,
    city: str,
    source: str,
    target: date,
    point_f: float,
    bias_table_path: Path | None = None,
    interval_table_path: Path | None = None,
) -> tuple[pd.Series, list[str]]:
    """Apply optional trained bias and interval artifacts to one prediction."""
    row = pd.DataFrame(
        [
            {
                "city": city,
                "source": source,
                "target_date": target.isoformat(),
                "point_f": point_f,
            }
        ]
    )
    warnings: list[str] = []

    if bias_table_path is not None:
        bias_table = pd.read_csv(bias_table_path)
        if _has_city_source(bias_table, city=city, source=source):
            row = apply_bias_correction(row, bias_table)
        else:
            warnings.append(
                f"no bias row for {city}/{source}; leaving point uncorrected"
            )

    if interval_table_path is not None:
        interval_table = pd.read_csv(interval_table_path)
        if _has_city_source(interval_table, city=city, source=source):
            row = apply_empirical_intervals(row, interval_table)
        else:
            warnings.append(
                f"no interval row for {city}/{source}; omitting calibrated interval"
            )

    return row.iloc[0], warnings


def _load_threshold_residuals(
    path: Path, *, city: str, source: str
) -> pd.Series:
    residuals = pd.read_csv(path)
    required = {"city", "source", "residual_f"}
    missing = required - set(residuals.columns)
    if missing:
        raise ValueError(f"threshold residuals CSV missing columns: {sorted(missing)}")
    selected = residuals[
        (residuals["city"].astype(str).str.lower() == city.lower())
        & (residuals["source"].astype(str) == source)
    ]
    return selected["residual_f"].astype(float)


def _load_threshold_recalibration_table(
    path: Path, *, city: str, source: str
) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "city",
        "source",
        "bucket_start",
        "bucket_end",
        "recalibrated_probability",
        "used",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"threshold recalibration CSV missing columns: {sorted(missing)}")
    exact = (table["city"].astype(str).str.lower() == city.lower()) & (
        table["source"].astype(str) == source
    )
    global_fallback = (table["city"].astype(str) == GLOBAL_RECALIBRATION_KEY) & (
        table["source"].astype(str) == GLOBAL_RECALIBRATION_KEY
    )
    selected = table[
        (exact | global_fallback) & (table["used"].astype(str).str.lower() == "true")
    ]
    return selected.copy()


def _threshold_probability_rows(
    *,
    calibration: pd.Series,
    residuals: pd.Series,
    offsets: tuple[int, ...],
    recalibration_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate threshold event probabilities from empirical residuals."""
    if residuals.empty:
        return pd.DataFrame(
            columns=["threshold_f", "offset_f", "predicted_probability"]
        )
    center = calibration.get("corrected_point_f", calibration.get("point_f"))
    if center is None or pd.isna(center):
        return pd.DataFrame(
            columns=["threshold_f", "offset_f", "predicted_probability"]
        )
    center_f = float(center)
    rounded_center = int(math.floor(center_f + 0.5))
    records = []
    for offset in offsets:
        threshold = rounded_center + int(offset)
        needed_residual = threshold - center_f
        probability = float((residuals >= needed_residual).mean())
        records.append(
            {
                "threshold_f": threshold,
                "offset_f": int(offset),
                "predicted_probability": probability,
            }
        )
    rows = pd.DataFrame(records)
    if recalibration_table is not None and not recalibration_table.empty:
        rows = _apply_threshold_probability_recalibration(rows, recalibration_table)
    return rows


def _apply_threshold_probability_recalibration(
    rows: pd.DataFrame, recalibration_table: pd.DataFrame
) -> pd.DataFrame:
    out = rows.copy()
    out["raw_predicted_probability"] = out["predicted_probability"].astype(float)
    out["recalibration_used"] = False
    out["recalibration_scope"] = "none"
    out["recalibration_n"] = pd.NA
    for index, row in out.iterrows():
        probability = float(row["raw_predicted_probability"])
        matches = recalibration_table[
            (recalibration_table["bucket_start"].astype(float) <= probability)
            & (
                (probability < recalibration_table["bucket_end"].astype(float))
                | (recalibration_table["bucket_end"].astype(float) >= 1.0)
            )
        ]
        if matches.empty:
            continue
        matches = matches.assign(
            _global_rank=(
                matches["city"].astype(str) == GLOBAL_RECALIBRATION_KEY
            ).astype(int)
        ).sort_values("_global_rank")
        match = matches.iloc[0]
        recalibrated = float(match["recalibrated_probability"])
        out.at[index, "predicted_probability"] = min(max(recalibrated, 0.0), 1.0)
        out.at[index, "recalibration_used"] = True
        out.at[index, "recalibration_scope"] = (
            "global"
            if str(match["city"]) == GLOBAL_RECALIBRATION_KEY
            else "city_source"
        )
        if "n" in match.index and pd.notna(match["n"]):
            out.at[index, "recalibration_n"] = int(match["n"])
    return out



def _existing_artifact(path: Path) -> Path | None:
    return path if path.exists() else None


def _resolve_model_artifacts(
    *,
    model_run_dir: Path | None,
    selected_sources: Path | None,
    bias_table: Path | None,
    interval_table: Path | None,
    threshold_residuals: Path | None,
    threshold_recalibration_table: Path | None,
) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None]:
    """Resolve explicit artifact paths, optionally defaulting from a run dir."""
    if model_run_dir is None:
        return (
            selected_sources,
            bias_table,
            interval_table,
            threshold_residuals,
            threshold_recalibration_table,
        )
    return (
        selected_sources
        or _existing_artifact(
            model_run_dir / "source_selection" / "recommended_sources.csv"
        )
        or _existing_artifact(model_run_dir / "source_selection" / "selected_sources.csv"),
        bias_table
        or _existing_artifact(model_run_dir / "model_policy" / "bias_table.csv")
        or _existing_artifact(model_run_dir / "train_eval" / "bias_table.csv"),
        interval_table
        or _existing_artifact(model_run_dir / "model_policy" / "interval_table.csv")
        or _existing_artifact(model_run_dir / "train_eval" / "interval_table.csv"),
        threshold_residuals
        or _existing_artifact(
            model_run_dir / "probability_calibration" / "threshold_residuals.csv"
        ),
        threshold_recalibration_table
        or _existing_artifact(
            model_run_dir
            / "probability_calibration"
            / "threshold_recalibration_table.csv"
        ),
    )


def _render_ascii_histogram(fc: NaiveForecast, width: int = 32) -> str:
    if not fc.bin_probs:
        return "  (no bins)"
    max_prob = max(fc.bin_probs.values())
    modal_bin = max(fc.bin_probs, key=fc.bin_probs.get)  # type: ignore[arg-type]
    lines: list[str] = []
    for b, p in fc.bin_probs.items():
        n_blocks = int(round((p / max_prob) * width))
        bar = "█" * n_blocks if n_blocks else "▏"
        marker = "  ← MODAL" if b == modal_bin else ""
        lines.append(f"  {b:>3}°F: {bar:<{width}} {p * 100:>4.1f}%{marker}")
    return "\n".join(lines)


def _render_calibration(row: pd.Series | None) -> str:
    if row is None:
        return ""

    parts = [f"Model source: {row['source']}"]
    corrected_point = row.get("corrected_point_f")
    if corrected_point is not None and pd.notna(corrected_point):
        parts.append(f"Corrected point: {float(corrected_point):.1f}°F")

    lower = row.get("interval_lower_f")
    upper = row.get("interval_upper_f")
    if lower is not None and upper is not None and pd.notna(lower) and pd.notna(upper):
        parts.append(f"Empirical interval: [{float(lower):.1f}°F, {float(upper):.1f}°F]")

    return "Calibration: " + "  |  ".join(parts) + "\n"


def _render_threshold_probabilities(thresholds: pd.DataFrame | None) -> str:
    if thresholds is None or thresholds.empty:
        return ""
    lines = ["Threshold probabilities:"]
    for row in thresholds.itertuples(index=False):
        suffix = ""
        if hasattr(row, "raw_predicted_probability") and getattr(
            row, "recalibration_used", False
        ):
            suffix = f" (raw {float(row.raw_predicted_probability) * 100:>4.1f}%)"
        lines.append(
            f"  P(high >= {int(row.threshold_f)}°F): "
            f"{float(row.predicted_probability) * 100:>5.1f}%{suffix}"
        )
    return "\n".join(lines) + "\n"


def _json_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _json_calibration(row: pd.Series | None) -> dict | None:
    if row is None:
        return None
    return {
        "city": row.get("city"),
        "source": row.get("source"),
        "target_date": row.get("target_date"),
        "point_f": _json_float(row.get("point_f")),
        "corrected_point_f": _json_float(row.get("corrected_point_f")),
        "bias_correction_f": _json_float(row.get("bias_correction_f")),
        "interval_lower_f": _json_float(row.get("interval_lower_f")),
        "interval_upper_f": _json_float(row.get("interval_upper_f")),
    }


def _json_forecast(fc: NaiveForecast) -> dict:
    return {
        "n_members": fc.n_members,
        "point_f": fc.point_f,
        "p10_f": fc.p10_f,
        "p50_f": fc.p50_f,
        "p90_f": fc.p90_f,
        "bin_probabilities": {str(key): value for key, value in fc.bin_probs.items()},
        "per_source_counts": fc.per_source_counts,
    }


def _json_thresholds(thresholds: pd.DataFrame | None) -> list[dict]:
    if thresholds is None or thresholds.empty:
        return []
    rows = []
    for row in thresholds.itertuples(index=False):
        item = {
            "threshold_f": int(row.threshold_f),
            "offset_f": int(row.offset_f),
            "predicted_probability": float(row.predicted_probability),
        }
        if hasattr(row, "raw_predicted_probability"):
            item["raw_predicted_probability"] = float(row.raw_predicted_probability)
        if hasattr(row, "recalibration_used"):
            item["recalibration_used"] = bool(row.recalibration_used)
        if hasattr(row, "recalibration_scope"):
            item["recalibration_scope"] = str(row.recalibration_scope)
        if hasattr(row, "recalibration_n") and pd.notna(row.recalibration_n):
            item["recalibration_n"] = int(row.recalibration_n)
        rows.append(item)
    return rows


def _json_multi_source_prediction(
    *,
    city: str,
    target: date,
    members: pd.DataFrame,
    mode: str,
    model_run_dir: Path | None,
    bias_table_path: Path | None,
    interval_table_path: Path | None,
    threshold_residuals_path: Path | None,
    threshold_recalibration_table_path: Path | None,
    threshold_offsets: tuple[int, ...] | None,
) -> dict:
    forecast, metadata, warnings = _multi_source_forecast(
        members=members,
        mode=mode,
        city=city,
        target=target,
        model_run_dir=model_run_dir,
    )
    source = MULTI_SOURCE_ARTIFACT_SOURCE
    calibration = None
    historical_rows = pd.DataFrame()
    rows_path = model_run_dir / "rows.csv" if model_run_dir is not None else None
    if rows_path is not None and rows_path.exists():
        historical_rows = _historical_multi_source_rows(
            rows_path=rows_path,
            city=city,
            target=target,
            source_weights=metadata["source_weights"],
        )

    if not historical_rows.empty:
        current = pd.DataFrame(
            [
                {
                    "city": city,
                    "source": source,
                    "target_date": target.isoformat(),
                    "point_f": forecast.point_f,
                }
            ]
        )
        bias_table = fit_bias_table(historical_rows)
        calibration = apply_bias_correction(current, bias_table).iloc[0]
        interval_table = fit_empirical_intervals(historical_rows)
        calibration = apply_empirical_intervals(
            pd.DataFrame([calibration.to_dict()]),
            interval_table,
        ).iloc[0]
    elif bias_table_path or interval_table_path:
        calibration, artifact_warnings = _apply_prediction_artifacts(
            city=city,
            source=source,
            target=target,
            point_f=forecast.point_f,
            bias_table_path=bias_table_path,
            interval_table_path=interval_table_path,
        )
        warnings.extend(artifact_warnings)

    threshold_probabilities = None
    if threshold_offsets is not None:
        if calibration is None:
            calibration = pd.Series(
                {
                    "city": city,
                    "source": source,
                    "target_date": target.isoformat(),
                    "point_f": forecast.point_f,
                }
            )
        if not historical_rows.empty:
            corrected_history = apply_bias_correction(
                historical_rows,
                fit_bias_table(historical_rows),
            )
            residuals = (
                corrected_history["actual_high_f"].astype(float)
                - corrected_history["corrected_point_f"].astype(float)
            )
        elif threshold_residuals_path is None:
            warnings.append(
                "threshold offsets requested but no threshold residuals artifact found"
            )
            residuals = pd.Series(dtype=float)
        else:
            residuals = _load_threshold_residuals(
                threshold_residuals_path, city=city, source=source
            )
        if not residuals.empty:
            recalibration_table = None
            if threshold_recalibration_table_path is not None:
                recalibration_table = _load_threshold_recalibration_table(
                    threshold_recalibration_table_path,
                    city=city,
                    source=source,
                )
            threshold_probabilities = _threshold_probability_rows(
                calibration=calibration,
                residuals=residuals,
                offsets=threshold_offsets,
                recalibration_table=recalibration_table,
            )
            if threshold_probabilities.empty:
                warnings.append(f"no threshold residual rows for {city}/{source}")

    return {
        **metadata,
        "forecast": _json_forecast(forecast),
        "calibration": _json_calibration(calibration),
        "threshold_probabilities": _json_thresholds(threshold_probabilities),
        "warnings": warnings,
    }


def _json_payload(
    station: Station,
    target: date,
    fc: NaiveForecast,
    *,
    selected_source: str | None,
    selected_applied: bool,
    calibration: pd.Series | None,
    threshold_probabilities: pd.DataFrame | None,
    generated_at: datetime | None = None,
    model_run_dir: Path | None = None,
    selected_sources_path: Path | None = None,
    bias_table_path: Path | None = None,
    interval_table_path: Path | None = None,
    threshold_residuals_path: Path | None = None,
    threshold_recalibration_table_path: Path | None = None,
) -> dict:
    artifact_paths = {
        "model_run_dir": str(model_run_dir) if model_run_dir is not None else None,
        "selected_sources": str(selected_sources_path) if selected_sources_path is not None else None,
        "bias_table": str(bias_table_path) if bias_table_path is not None else None,
        "interval_table": str(interval_table_path) if interval_table_path is not None else None,
        "threshold_residuals": (
            str(threshold_residuals_path) if threshold_residuals_path is not None else None
        ),
        "threshold_recalibration_table": (
            str(threshold_recalibration_table_path)
            if threshold_recalibration_table_path is not None
            else None
        ),
    }
    return {
        "schema_version": PREDICTION_JSON_SCHEMA_VERSION,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "city": station.slug,
        "station": {
            "name": station.name,
            "nws_station": station.nws_station,
            "lst_offset_hours": station.lst_offset_hours,
        },
        "target_date": target.isoformat(),
        "selected_source": selected_source,
        "selected_source_applied": selected_applied,
        "forecast": _json_forecast(fc),
        "artifact_paths": artifact_paths,
        "calibration": _json_calibration(calibration),
        "threshold_probabilities": _json_thresholds(threshold_probabilities),
    }


def _render(
    station: Station,
    target: date,
    fc: NaiveForecast,
    *,
    calibration: pd.Series | None = None,
    threshold_probabilities: pd.DataFrame | None = None,
) -> str:
    header = (
        f"{station.name} | {target.strftime('%a %b %d, %Y')} "
        f"| Settlement: {station.nws_station} (LST UTC{station.lst_offset_hours:+d})"
    )
    rule = "═" * len(header)
    src_parts = [f"{s}({n})" for s, n in fc.per_source_counts.items() if n]
    src_line = "+".join(src_parts) + f" → {fc.n_members} members"
    body = _render_ascii_histogram(fc)
    return (
        f"\n{header}\n{rule}\n{body}\n\n"
        f"Point estimate: {fc.point_f:.1f}°F  "
        f"|  80% CI: [{fc.p10_f:.1f}°F, {fc.p90_f:.1f}°F]  "
        f"|  Median: {fc.p50_f:.1f}°F\n"
        f"{_render_calibration(calibration)}"
        f"{_render_threshold_probabilities(threshold_probabilities)}"
        f"Sources: {src_line}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="predict",
        description="Probabilistic daily-high prediction (Milestone A naive ensemble).",
    )
    parser.add_argument(
        "--city",
        required=True,
        help=f"City slug. One of: {sorted(load_stations().keys())}",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="YYYY-MM-DD, or `today`/`tomorrow`/`yesterday`.",
    )
    parser.add_argument(
        "--model-run-dir",
        type=Path,
        help=(
            "Optional historical run directory. Defaults artifact paths to "
            "<run>/source_selection/recommended_sources.csv when present "
            "(falling back to selected_sources.csv) and model_policy tables "
            "when present (falling back to train_eval tables)."
        ),
    )
    parser.add_argument(
        "--selected-sources",
        type=Path,
        help=(
            "Optional source_selection/selected_sources.csv. When the city "
            "has an individual selected Open-Meteo source, predict from that "
            "source's members; openmeteo_naive keeps the pooled baseline."
        ),
    )
    parser.add_argument(
        "--bias-table",
        type=Path,
        help="Optional train_eval/bias_table.csv to apply a point correction.",
    )
    parser.add_argument(
        "--interval-table",
        type=Path,
        help="Optional train_eval/interval_table.csv to apply empirical intervals.",
    )
    parser.add_argument(
        "--threshold-residuals",
        type=Path,
        help="Optional probability_calibration/threshold_residuals.csv.",
    )
    parser.add_argument(
        "--threshold-recalibration-table",
        type=Path,
        help="Optional probability_calibration/threshold_recalibration_table.csv.",
    )
    parser.add_argument(
        "--threshold-offsets",
        help=(
            "Optional comma-separated integer offsets around rounded corrected "
            "point for threshold probabilities, e.g. -4,-2,0,2,4."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON to stdout instead of the text report.",
    )
    args = parser.parse_args(argv)
    (
        selected_sources_path,
        bias_table_path,
        interval_table_path,
        threshold_residuals_path,
        threshold_recalibration_table_path,
    ) = _resolve_model_artifacts(
        model_run_dir=args.model_run_dir,
        selected_sources=args.selected_sources,
        bias_table=args.bias_table,
        interval_table=args.interval_table,
        threshold_residuals=args.threshold_residuals,
        threshold_recalibration_table=args.threshold_recalibration_table,
    )

    station = get_station(args.city)
    target = _parse_date(args.date)
    use_historical = target < date.today() - timedelta(days=2)

    print(f"\nFetching {len(SOURCES)} Open-Meteo sources for "
          f"{station.name} on {target}...", file=sys.stderr)
    sources = _fetch_all_parallel(station, target, use_historical=use_historical)
    members = members_dataframe(sources)

    if members.empty:
        print("ERROR: every Open-Meteo source returned empty.", file=sys.stderr)
        return 1

    selected_source = None
    selected_applied = False
    if selected_sources_path:
        try:
            selected_source = _load_selected_source(selected_sources_path, args.city)
        except (OSError, ValueError) as error:
            print(f"ERROR: could not read selected sources: {error}", file=sys.stderr)
            return 1

        members, selected_applied = _members_for_selected_source(
            members, selected_source
        )
        if selected_source is None:
            print(
                f"  ! no selected source for {args.city}; using pooled members",
                file=sys.stderr,
            )
        elif selected_applied:
            print(f"  = using selected source: {selected_source}", file=sys.stderr)
        elif selected_source == "openmeteo_naive":
            print(
                "  = selected source is openmeteo_naive; using pooled members",
                file=sys.stderr,
            )
        else:
            print(
                f"  ! selected source {selected_source!r} returned no members; "
                "using pooled members",
                file=sys.stderr,
            )

    fc = naive_forecast_from_members(members)
    calibration = None
    threshold_probabilities = None
    if bias_table_path or interval_table_path:
        source = _prediction_source(selected_source, selected_applied=selected_applied)
        try:
            calibration, warnings = _apply_prediction_artifacts(
                city=args.city,
                source=source,
                target=target,
                point_f=fc.point_f,
                bias_table_path=bias_table_path,
                interval_table_path=interval_table_path,
            )
        except (OSError, ValueError) as error:
            print(f"ERROR: could not apply model artifacts: {error}", file=sys.stderr)
            return 1
        for warning in warnings:
            print(f"  ! {warning}", file=sys.stderr)

    if args.threshold_offsets:
        source = _prediction_source(selected_source, selected_applied=selected_applied)
        if calibration is None:
            calibration = pd.Series(
                {
                    "city": args.city,
                    "source": source,
                    "target_date": target.isoformat(),
                    "point_f": fc.point_f,
                }
            )
        if threshold_residuals_path is None:
            print(
                "  ! threshold offsets requested but no threshold residuals artifact found",
                file=sys.stderr,
            )
        else:
            try:
                residuals = _load_threshold_residuals(
                    threshold_residuals_path, city=args.city, source=source
                )
                recalibration_table = None
                if threshold_recalibration_table_path is not None:
                    recalibration_table = _load_threshold_recalibration_table(
                        threshold_recalibration_table_path,
                        city=args.city,
                        source=source,
                    )
                threshold_probabilities = _threshold_probability_rows(
                    calibration=calibration,
                    residuals=residuals,
                    offsets=_parse_int_list(args.threshold_offsets, name="threshold-offsets"),
                    recalibration_table=recalibration_table,
                )
                if threshold_probabilities.empty:
                    print(
                        f"  ! no threshold residual rows for {args.city}/{source}",
                        file=sys.stderr,
                    )
            except (OSError, ValueError) as error:
                print(f"ERROR: could not apply threshold residuals: {error}", file=sys.stderr)
                return 1

    if args.json:
        print(
            json.dumps(
                _json_payload(
                    station,
                    target,
                    fc,
                    selected_source=selected_source,
                    selected_applied=selected_applied,
                    calibration=calibration,
                    threshold_probabilities=threshold_probabilities,
                    model_run_dir=args.model_run_dir,
                    selected_sources_path=selected_sources_path,
                    bias_table_path=bias_table_path,
                    interval_table_path=interval_table_path,
                    threshold_residuals_path=threshold_residuals_path,
                    threshold_recalibration_table_path=threshold_recalibration_table_path,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            _render(
                station,
                target,
                fc,
                calibration=calibration,
                threshold_probabilities=threshold_probabilities,
            )
        )
    return 0


def _parse_int_list(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise SystemExit(f"--{name} must be a comma-separated integer list") from exc
    if not parsed:
        raise SystemExit(f"--{name} must contain at least one value")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

"""Market-free probability calibration for NBM candidate packets.

The calibrator changes only the NBM degree PMF shape. It does not consume
market prices, order books, private PnL labels, or trading instructions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.nbm_candidate import _quantiles
from src.models.nowcast_predictions import NOWCAST_PREDICTION_COLUMNS, QUANTILES

NBM_CALIBRATED_SCHEMA_VERSION = "1.0"
NBM_CALIBRATED_MODEL_VERSION = "nbm-text-calibrated-v1"
DEFAULT_TARGET_COVERAGE = 0.80


@dataclass(frozen=True)
class NbmCalibrationParams:
    temperature: float
    objective: str
    train_start: str
    train_end: str
    target_coverage: float
    n_train: int
    train_nll: float
    train_degree_brier: float
    train_q10_q90_coverage: float
    generated_at: str
    schema_version: str = NBM_CALIBRATED_SCHEMA_VERSION
    model_version: str = NBM_CALIBRATED_MODEL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "generated_at": self.generated_at,
            "temperature": self.temperature,
            "objective": self.objective,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "target_coverage": self.target_coverage,
            "n_train": self.n_train,
            "train_nll": self.train_nll,
            "train_degree_brier": self.train_degree_brier,
            "train_q10_q90_coverage": self.train_q10_q90_coverage,
            "notes": [
                "Market-free NBM PMF calibration. No market prices, order books, private PnL labels, or trade instructions.",
                "Temperature >1 flattens an overconfident PMF while preserving the NBM degree support and rank order.",
            ],
        }


def load_calibration_params(path: Path) -> NbmCalibrationParams:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return NbmCalibrationParams(
        temperature=float(payload["temperature"]),
        objective=str(payload.get("objective") or "nll"),
        train_start=str(payload.get("train_start") or ""),
        train_end=str(payload.get("train_end") or ""),
        target_coverage=float(payload.get("target_coverage", DEFAULT_TARGET_COVERAGE)),
        n_train=int(payload.get("n_train", 0)),
        train_nll=float(payload.get("train_nll", math.nan)),
        train_degree_brier=float(payload.get("train_degree_brier", math.nan)),
        train_q10_q90_coverage=float(payload.get("train_q10_q90_coverage", math.nan)),
        generated_at=str(payload.get("generated_at") or ""),
        schema_version=str(payload.get("schema_version") or NBM_CALIBRATED_SCHEMA_VERSION),
        model_version=str(payload.get("model_version") or NBM_CALIBRATED_MODEL_VERSION),
    )


def write_calibration_params(params: NbmCalibrationParams, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fit_temperature_scale(
    scored_rows: pd.DataFrame,
    *,
    prediction_root: Path,
    train_start: str,
    train_end: str,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
    objective: str = "nll",
    min_temperature: float = 1.0,
    max_temperature: float = 4.0,
    step: float = 0.05,
) -> tuple[NbmCalibrationParams, pd.DataFrame]:
    """Fit one global PMF temperature on the training window only."""
    training = _training_items(scored_rows, prediction_root, train_start=train_start, train_end=train_end)
    if not training:
        raise ValueError("no training rows with PMFs and actuals")
    candidates = []
    count = int(round((max_temperature - min_temperature) / step))
    for index in range(count + 1):
        temperature = round(min_temperature + index * step, 10)
        metrics = _metrics_for_temperature(training, temperature)
        candidates.append(
            {
                "temperature": temperature,
                "n": metrics["n"],
                "nll": metrics["nll"],
                "degree_brier": metrics["degree_brier"],
                "q10_q90_coverage": metrics["q10_q90_coverage"],
                "coverage_gap": abs(metrics["q10_q90_coverage"] - target_coverage),
            }
        )
    grid = pd.DataFrame(candidates)
    if objective == "coverage_then_nll":
        selected = grid.sort_values(["coverage_gap", "nll", "temperature"]).iloc[0]
    elif objective == "nll":
        selected = grid.sort_values(["nll", "coverage_gap", "temperature"]).iloc[0]
    else:
        raise ValueError(f"unsupported objective: {objective}")
    params = NbmCalibrationParams(
        temperature=float(selected["temperature"]),
        objective=objective,
        train_start=train_start,
        train_end=train_end,
        target_coverage=target_coverage,
        n_train=int(selected["n"]),
        train_nll=float(selected["nll"]),
        train_degree_brier=float(selected["degree_brier"]),
        train_q10_q90_coverage=float(selected["q10_q90_coverage"]),
        generated_at=datetime.now(UTC).isoformat(),
    )
    return params, grid


def write_nbm_calibrated_predictions(
    *,
    input_predictions_path: Path,
    output_dir: Path,
    calibration_params_path: Path,
    git_commit: str | None = None,
) -> pd.DataFrame:
    params = load_calibration_params(calibration_params_path)
    predictions = pd.read_csv(input_predictions_path, dtype={"decision_time_label": "string"})
    calibrated = build_nbm_calibrated_predictions(predictions=predictions, params=params)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(output_dir / "predictions_nowcast.csv", index=False)
    manifest = {
        "schema_version": NBM_CALIBRATED_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "model_version": params.model_version,
        "row_count": int(len(calibrated)),
        "input_rows": int(len(predictions)),
        "calibration_params_path": str(calibration_params_path),
        "calibration_params": params.to_dict(),
        "notes": [
            "Weather-only calibrated NBM candidate packet. No market prices, order books, private PnL labels, or trade instructions.",
            "Raw predictions_nowcast_nbm remains unchanged; this mode is separate and candidate-only.",
        ],
    }
    (output_dir / "predictions_nowcast_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return calibrated


def build_nbm_calibrated_predictions(
    *,
    predictions: pd.DataFrame,
    params: NbmCalibrationParams,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=NOWCAST_PREDICTION_COLUMNS)
    rows: list[dict[str, Any]] = []
    group_cols = [
        column
        for column in [
            "city",
            "platform",
            "market_type",
            "station_id",
            "target_date",
            "decision_time_label",
        ]
        if column in predictions.columns
    ]
    for _, group in predictions.groupby(group_cols, sort=False, dropna=False):
        first = group.iloc[0].to_dict()
        pmf = _parse_pmf(first.get("pmf_degree_json"))
        if not pmf:
            continue
        calibrated = temperature_scale_pmf(pmf, params.temperature)
        rows.extend(_rows_from_pmf(first, calibrated, params=params))
    return pd.DataFrame(rows, columns=NOWCAST_PREDICTION_COLUMNS)


def temperature_scale_pmf(pmf: dict[int, float], temperature: float) -> dict[int, float]:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    scaled = {degree: probability ** (1.0 / temperature) for degree, probability in pmf.items() if probability > 0}
    total = sum(scaled.values())
    if total <= 0:
        return {}
    return {degree: probability / total for degree, probability in sorted(scaled.items())}


def apply_calibration_to_root(
    *,
    prediction_root: Path,
    calibration_params_path: Path,
    apply_start: str,
    apply_end: str,
    git_commit: str | None = None,
) -> list[Path]:
    written = []
    for path in sorted(prediction_root.rglob("predictions_nowcast_nbm/predictions_nowcast.csv")):
        target_date = _target_date_from_file(path)
        if target_date is None or target_date < apply_start or target_date > apply_end:
            continue
        output_dir = path.parent.parent / "predictions_nowcast_nbm_calibrated"
        write_nbm_calibrated_predictions(
            input_predictions_path=path,
            output_dir=output_dir,
            calibration_params_path=calibration_params_path,
            git_commit=git_commit,
        )
        written.append(output_dir / "predictions_nowcast.csv")
    return written


def _training_items(
    scored_rows: pd.DataFrame,
    prediction_root: Path,
    *,
    train_start: str,
    train_end: str,
) -> list[tuple[dict[int, float], int]]:
    rows = scored_rows.copy()
    rows["target_date"] = rows["target_date"].astype(str).str.slice(0, 10)
    rows = rows[(rows["target_date"] >= train_start) & (rows["target_date"] <= train_end)]
    actuals = {
        (
            str(row["city"]).strip().lower(),
            str(row["target_date"])[:10],
            str(row["decision_time_label"]).lstrip("0") or "0",
        ): int(float(row["actual_degree_f"]))
        for row in rows.to_dict("records")
    }
    items: list[tuple[dict[int, float], int]] = []
    for path in sorted(prediction_root.rglob("predictions_nowcast_nbm/predictions_nowcast.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            frame = pd.read_csv(handle)
        if frame.empty:
            continue
        first = frame.iloc[0].to_dict()
        key = (
            str(first.get("city") or "").strip().lower(),
            str(first.get("target_date") or "")[:10],
            str(first.get("decision_time_label") or "").lstrip("0") or "0",
        )
        actual = actuals.get(key)
        if actual is None:
            continue
        pmf = _parse_pmf(first.get("pmf_degree_json"))
        if pmf:
            items.append((pmf, actual))
    return items


def _metrics_for_temperature(
    items: list[tuple[dict[int, float], int]],
    temperature: float,
) -> dict[str, float]:
    nll = 0.0
    brier = 0.0
    coverage = 0
    for pmf_raw, actual in items:
        pmf = temperature_scale_pmf(pmf_raw, temperature)
        actual_prob = pmf.get(actual, 0.0)
        nll += -math.log(max(actual_prob, 1e-12))
        support = set(pmf) | {actual}
        brier += sum((pmf.get(degree, 0.0) - (1.0 if degree == actual else 0.0)) ** 2 for degree in support)
        q10 = _pmf_quantile(pmf, 0.10)
        q90 = _pmf_quantile(pmf, 0.90)
        coverage += int(q10 <= actual <= q90)
    n = len(items)
    return {
        "n": float(n),
        "nll": nll / n,
        "degree_brier": brier / n,
        "q10_q90_coverage": coverage / n,
    }


def _rows_from_pmf(
    base: dict[str, Any],
    pmf: dict[int, float],
    *,
    params: NbmCalibrationParams,
) -> list[dict[str, Any]]:
    quantiles = _quantiles(pmf)
    point = sum(degree * probability for degree, probability in pmf.items())
    pmf_json = json.dumps({str(degree): probability for degree, probability in sorted(pmf.items())}, sort_keys=True)
    rows = []
    for degree, probability in sorted(pmf.items()):
        row = {column: base.get(column, "") for column in NOWCAST_PREDICTION_COLUMNS}
        row.update(
            {
                "model_version": params.model_version,
                "bin_lower_f": degree,
                "bin_upper_f": degree,
                "bin_label": str(degree),
                "model_probability": probability,
                "calibrated_probability": probability,
                "point_f": point,
                **{f"q{q:02d}_f": quantiles[q] for q in QUANTILES},
                "pmf_degree_json": pmf_json,
                "source_policy": "nbm_text_calibrated",
                "weather_reason_codes": _append_reason(base.get("weather_reason_codes"), "nbm_temperature_scaled"),
            }
        )
        rows.append(row)
    return rows


def _target_date_from_file(path: Path) -> str | None:
    try:
        frame = pd.read_csv(path, nrows=1, usecols=["target_date"])
    except ValueError:
        return None
    if frame.empty:
        return None
    return str(frame.iloc[0]["target_date"])[:10]


def _parse_pmf(value: Any) -> dict[int, float]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    pmf: dict[int, float] = {}
    if not isinstance(payload, dict):
        return {}
    for degree_raw, probability_raw in payload.items():
        try:
            degree = int(float(degree_raw))
            probability = float(probability_raw)
        except (TypeError, ValueError):
            continue
        if probability > 0 and math.isfinite(probability):
            pmf[degree] = pmf.get(degree, 0.0) + probability
    total = sum(pmf.values())
    if total <= 0:
        return {}
    return {degree: probability / total for degree, probability in sorted(pmf.items())}


def _pmf_quantile(pmf: dict[int, float], cutoff: float) -> int:
    cumulative = 0.0
    selected = max(pmf)
    for degree, probability in sorted(pmf.items()):
        cumulative += probability
        if cumulative >= cutoff:
            selected = degree
            break
    return selected


def _append_reason(existing: Any, reason: str) -> str:
    text = "" if existing is None or pd.isna(existing) else str(existing).strip()
    if not text:
        return reason
    parts = [part for part in text.split(";") if part]
    if reason not in parts:
        parts.append(reason)
    return ";".join(parts)

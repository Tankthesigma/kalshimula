import json
from pathlib import Path

import pandas as pd

from src.models.nbm_probability_calibration import (
    NbmCalibrationParams,
    build_nbm_calibrated_predictions,
    fit_temperature_scale,
    temperature_scale_pmf,
    write_calibration_params,
    write_nbm_calibrated_predictions,
)
from src.models.nowcast_predictions import NOWCAST_PREDICTION_COLUMNS


def test_temperature_scale_pmf_flattens_overconfident_distribution() -> None:
    pmf = {70: 0.8, 69: 0.1, 71: 0.1}

    scaled = temperature_scale_pmf(pmf, 2.0)

    assert abs(sum(scaled.values()) - 1.0) < 1e-9
    assert scaled[70] < pmf[70]
    assert scaled[69] > pmf[69]


def test_fit_temperature_scale_uses_training_window_only(tmp_path: Path) -> None:
    root = tmp_path / "packets"
    _write_packet(root, "2026-05-01", "nyc", "07", {70: 0.8, 69: 0.1, 71: 0.1})
    _write_packet(root, "2026-05-02", "nyc", "07", {70: 0.8, 69: 0.1, 71: 0.1})
    _write_packet(root, "2026-05-20", "nyc", "07", {70: 0.8, 69: 0.1, 71: 0.1})
    scored = pd.DataFrame(
        [
            {"city": "nyc", "target_date": "2026-05-01", "decision_time_label": "07", "actual_degree_f": 69},
            {"city": "nyc", "target_date": "2026-05-02", "decision_time_label": "07", "actual_degree_f": 71},
            {"city": "nyc", "target_date": "2026-05-20", "decision_time_label": "07", "actual_degree_f": 70},
        ]
    )

    params, grid = fit_temperature_scale(
        scored,
        prediction_root=root,
        train_start="2026-05-01",
        train_end="2026-05-02",
        max_temperature=2.0,
        step=0.5,
    )

    assert params.n_train == 2
    assert params.temperature > 1.0
    assert set(grid["temperature"]) == {1.0, 1.5, 2.0}


def test_build_nbm_calibrated_predictions_emits_separate_mode() -> None:
    predictions = _prediction_frame("2026-05-01", "nyc", "07", {70: 0.8, 69: 0.1, 71: 0.1})
    params = _params(temperature=2.0)

    out = build_nbm_calibrated_predictions(predictions=predictions, params=params)

    assert list(out.columns) == NOWCAST_PREDICTION_COLUMNS
    assert out["model_version"].unique().tolist() == ["nbm-text-calibrated-v1"]
    assert out["source_policy"].unique().tolist() == ["nbm_text_calibrated"]
    assert "nbm_temperature_scaled" in out.iloc[0]["weather_reason_codes"]
    assert abs(out["calibrated_probability"].sum() - 1.0) < 1e-9
    assert out[out["bin_lower_f"] == 70]["calibrated_probability"].iloc[0] < 0.8


def test_write_nbm_calibrated_predictions_writes_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "predictions_nowcast_nbm" / "predictions_nowcast.csv"
    input_path.parent.mkdir(parents=True)
    _prediction_frame("2026-05-01", "nyc", "07", {70: 1.0}).to_csv(input_path, index=False)
    params_path = tmp_path / "params.json"
    write_calibration_params(_params(temperature=1.5), params_path)

    out = write_nbm_calibrated_predictions(
        input_predictions_path=input_path,
        output_dir=tmp_path / "predictions_nowcast_nbm_calibrated",
        calibration_params_path=params_path,
        git_commit="abc123",
    )

    assert len(out) == 1
    manifest = json.loads(
        (tmp_path / "predictions_nowcast_nbm_calibrated" / "predictions_nowcast_manifest.json").read_text()
    )
    assert manifest["git_commit"] == "abc123"
    assert manifest["calibration_params"]["temperature"] == 1.5


def _write_packet(
    root: Path,
    target_date: str,
    city: str,
    label: str,
    pmf: dict[int, float],
) -> None:
    path = (
        root
        / target_date
        / f"{int(label):02d}_local"
        / city
        / "weather_desk"
        / "predictions_nowcast_nbm"
        / "predictions_nowcast.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _prediction_frame(target_date, city, label, pmf).to_csv(path, index=False)


def _prediction_frame(target_date: str, city: str, label: str, pmf: dict[int, float]) -> pd.DataFrame:
    rows = []
    pmf_json = json.dumps({str(k): v for k, v in sorted(pmf.items())})
    for degree, probability in sorted(pmf.items()):
        row = {column: "" for column in NOWCAST_PREDICTION_COLUMNS}
        row.update(
            {
                "model_version": "nbm-text-candidate-v1",
                "city": city,
                "platform": "kalshi",
                "market_type": "high",
                "station_id": "KNYC",
                "target_date": target_date,
                "decision_time_label": str(int(label)),
                "as_of_ts_utc": f"{target_date}T11:20:00+00:00",
                "bin_lower_f": degree,
                "bin_upper_f": degree,
                "bin_label": str(degree),
                "model_probability": probability,
                "calibrated_probability": probability,
                "point_f": sum(k * v for k, v in pmf.items()),
                "q10_f": min(pmf),
                "q90_f": max(pmf),
                "pmf_degree_json": pmf_json,
                "source_policy": "nbm_text",
                "weather_reason_codes": "nbm_guidance_candidate",
                "source_independence_score": 1.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=NOWCAST_PREDICTION_COLUMNS)


def _params(temperature: float) -> NbmCalibrationParams:
    return NbmCalibrationParams(
        temperature=temperature,
        objective="nll",
        train_start="2026-05-01",
        train_end="2026-05-12",
        target_coverage=0.8,
        n_train=2,
        train_nll=1.0,
        train_degree_brier=0.5,
        train_q10_q90_coverage=0.8,
        generated_at="2026-05-26T00:00:00+00:00",
    )

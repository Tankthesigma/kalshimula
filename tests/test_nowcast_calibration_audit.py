import json
from pathlib import Path

import pandas as pd

from src.models.nowcast_calibration_audit import (
    build_calibration_audit,
    discover_prediction_files,
    mode_from_prediction_file,
    read_actuals_csv,
    write_calibration_audit,
)
from src.models.nowcast_predictions import NOWCAST_PREDICTION_COLUMNS
from src.nowcast_calibration_audit_cli import main as calibration_cli_main


def test_build_calibration_audit_scores_modes_and_smoke_label(tmp_path: Path) -> None:
    raw_file = _write_prediction_file(
        tmp_path,
        mode="raw",
        pmf={70: 1.0},
        expected=70.0,
    )
    nbm_file = _write_prediction_file(
        tmp_path,
        mode="nbm",
        pmf={69: 0.5, 71: 0.5},
        expected=70.0,
    )
    actuals = pd.DataFrame(
        [
            {
                "city": "nyc",
                "target_date": "2026-05-25",
                "actual_high_f": 70.0,
                "actual_source": "fixture",
            }
        ]
    )

    result = build_calibration_audit(
        [raw_file, nbm_file],
        actuals=actuals,
        min_statistical_n=30,
    )

    summary = result.summary.set_index("mode")
    assert summary.loc["raw", "mean_degree_brier"] == 0.0
    assert summary.loc["raw", "mean_expected_high_mae_f"] == 0.0
    assert summary.loc["raw", "top1_hit_rate"] == 1.0
    assert summary.loc["nbm", "mean_degree_brier"] > summary.loc["raw", "mean_degree_brier"]
    assert summary.loc["nbm", "mean_log_loss"] > summary.loc["raw", "mean_log_loss"]
    assert summary.loc["raw", "evidence_label"] == "SMOKE / NOT STATISTICAL EVIDENCE"
    assert set(result.reliability["mode"]) == {"raw", "nbm"}


def test_missing_actual_is_excluded(tmp_path: Path) -> None:
    prediction_file = _write_prediction_file(
        tmp_path,
        mode="raw",
        pmf={70: 1.0},
        expected=70.0,
    )
    result = build_calibration_audit(
        [prediction_file],
        actuals=pd.DataFrame(columns=["city", "target_date", "actual_high_f", "actual_source"]),
    )

    assert result.scored_rows.empty
    assert result.exclusions.iloc[0]["reason"] == "missing_actual_high"


def test_actual_high_rounding_is_half_up(tmp_path: Path) -> None:
    prediction_file = _write_prediction_file(
        tmp_path,
        mode="raw",
        pmf={71: 1.0},
        expected=71.0,
    )
    actuals = pd.DataFrame(
        [{"city": "nyc", "target_date": "2026-05-25", "actual_high_f": 70.5}]
    )

    result = build_calibration_audit([prediction_file], actuals=actuals)

    assert result.scored_rows.iloc[0]["actual_degree_f"] == 71
    assert result.summary.iloc[0]["mean_degree_brier"] == 0.0


def test_discover_prediction_files_and_mode_name(tmp_path: Path) -> None:
    prediction_file = _write_prediction_file(
        tmp_path,
        mode="heat_corrected",
        pmf={70: 1.0},
        expected=70.0,
    )

    assert discover_prediction_files([tmp_path]) == [prediction_file]
    assert mode_from_prediction_file(prediction_file) == "heat_corrected"


def test_read_actuals_csv_and_cli_write_outputs(tmp_path: Path) -> None:
    prediction_file = _write_prediction_file(
        tmp_path,
        mode="raw",
        pmf={70: 1.0},
        expected=70.0,
    )
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date,actual_high_f\nNYC,2026-05-25,70\n",
        encoding="utf-8",
    )
    assert read_actuals_csv(actuals_path).iloc[0]["actual_source"] == "actuals_csv"
    out_dir = tmp_path / "audit"

    assert (
        calibration_cli_main(
            [
                "--prediction-file",
                str(prediction_file),
                "--actuals-csv",
                str(actuals_path),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert (out_dir / "probability_calibration_summary.csv").exists()
    assert (out_dir / "probability_calibration_report.md").exists()


def test_write_calibration_audit_outputs_all_artifacts(tmp_path: Path) -> None:
    prediction_file = _write_prediction_file(
        tmp_path,
        mode="raw",
        pmf={70: 1.0},
        expected=70.0,
    )
    actuals = pd.DataFrame(
        [{"city": "nyc", "target_date": "2026-05-25", "actual_high_f": 70.0}]
    )
    result = build_calibration_audit([prediction_file], actuals=actuals)
    out_dir = tmp_path / "model_quality"

    write_calibration_audit(result, out_dir)

    assert (out_dir / "probability_calibration_rows.csv").exists()
    assert (out_dir / "probability_calibration_reliability.csv").exists()
    assert (out_dir / "probability_calibration_exclusions.csv").exists()
    assert (out_dir / "probability_calibration_manifest.json").exists()


def _write_prediction_file(
    tmp_path: Path,
    *,
    mode: str,
    pmf: dict[int, float],
    expected: float,
) -> Path:
    packet_dir = tmp_path / f"predictions_nowcast_{mode}"
    packet_dir.mkdir()
    row = {column: "" for column in NOWCAST_PREDICTION_COLUMNS}
    row.update(
        {
            "model_version": f"{mode}-fixture",
            "city": "nyc",
            "platform": "kalshi",
            "market_type": "high",
            "station_id": "KNYC",
            "target_date": "2026-05-25",
            "prediction_ts_utc": "2026-05-25T12:00:00+00:00",
            "prediction_time_local": "2026-05-25T08:00:00-04:00",
            "decision_time_label": "07",
            "as_of_ts_utc": "2026-05-25T12:00:00+00:00",
            "bin_lower_f": min(pmf),
            "bin_upper_f": max(pmf),
            "bin_label": "fixture",
            "model_probability": 1.0,
            "calibrated_probability": 1.0,
            "point_f": expected,
            "q10_f": min(pmf),
            "q90_f": max(pmf),
            "pmf_degree_json": json.dumps({str(k): v for k, v in pmf.items()}),
            "source_policy": mode,
            "nowcast_veto_flag": "ok",
            "weather_reason_codes": mode,
            "station_rule_confidence": "high",
            "source_independence_score": 1.0,
        }
    )
    path = packet_dir / "predictions_nowcast.csv"
    pd.DataFrame([row], columns=NOWCAST_PREDICTION_COLUMNS).to_csv(path, index=False)
    return path

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.lone_outlier_correction import (
    apply_lone_outlier_correction,
    write_lone_outlier_correction,
)


def _predictions(point: float = 100.0) -> pd.DataFrame:
    rows = []
    for degree, probability in [(99, 0.25), (100, 0.5), (101, 0.25)]:
        rows.append(
            {
                "model_version": "mainline-nowcast-v1",
                "city": "phoenix",
                "platform": "kalshi",
                "market_type": "high",
                "station_id": "KPHX",
                "target_date": "2026-05-24",
                "prediction_ts_utc": "2026-05-24T12:00:00+00:00",
                "prediction_time_local": "2026-05-24T05:00:00-07:00",
                "decision_time_label": "morning",
                "as_of_ts_utc": "2026-05-24T12:00:00+00:00",
                "bin_lower_f": degree,
                "bin_upper_f": degree,
                "bin_label": str(degree),
                "model_probability": probability,
                "calibrated_probability": probability,
                "point_f": point,
                "q05_f": 99,
                "q10_f": 99,
                "q20_f": 99,
                "q25_f": 99,
                "q30_f": 100,
                "q40_f": 100,
                "q50_f": 100,
                "q60_f": 100,
                "q70_f": 100,
                "q75_f": 101,
                "q80_f": 101,
                "q90_f": 101,
                "q95_f": 101,
                "pmf_degree_json": json.dumps({"99": 0.25, "100": 0.5, "101": 0.25}),
                "source_policy": "gfs_ens",
                "nowcast_veto_flag": False,
                "weather_reason_codes": "",
                "station_rule_confidence": "high",
                "source_independence_score": 1.0,
                "feature_hash": "abc",
            }
        )
    return pd.DataFrame(rows)


def _payload(consensus_point: float = 96.0) -> dict:
    return {
        "predictions": [
            {
                "city": "phoenix",
                "target_date": "2026-05-24",
                "multi_source": {
                    "forecast": {"point_f": consensus_point},
                    "calibration": {"corrected_point_f": consensus_point},
                },
            }
        ]
    }


def _guidance(nws_point: float = 96.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": "phoenix",
                "source": "nws_forecast",
                "station_id": "KPHX",
                "market_type": "high",
                "target_date": "2026-05-24",
                "issue_ts_utc": "2026-05-24T11:00:00+00:00",
                "valid_ts_utc": "2026-05-25T00:00:00+00:00",
                "available_ts_utc": "2026-05-24T11:00:00+00:00",
                "guidance_point_f": nws_point,
                "guidance_q10_f": None,
                "guidance_q50_f": nws_point,
                "guidance_q90_f": None,
                "actual_high_f": None,
                "raw_payload_hash": "hash",
            }
        ]
    )


def test_lone_outlier_correction_blends_gfs_toward_consensus_and_nws() -> None:
    corrected, corrections = apply_lone_outlier_correction(
        _predictions(),
        prediction_payload=_payload(),
        guidance_rows=_guidance(),
        threshold_f=3.0,
        blend_weight=0.5,
    )

    assert len(corrections) == 1
    correction = corrections.iloc[0]
    assert correction["original_point_f"] == 100.0
    assert correction["target_point_f"] == 96.0
    assert correction["corrected_point_f"] == 98.0
    assert correction["delta_f"] == -2.0
    assert set(corrected["point_f"]) == {98.0}
    assert set(corrected["model_version"]) == {"mainline-nowcast-v1-lone-outlier-candidate"}
    assert "lone_outlier_corrected" in corrected.iloc[0]["weather_reason_codes"]
    assert corrected["calibrated_probability"].sum() == pytest.approx(1.0)
    assert corrected["bin_lower_f"].tolist() == [97, 98, 99]


def test_lone_outlier_correction_requires_same_side_outlier() -> None:
    corrected, corrections = apply_lone_outlier_correction(
        _predictions(),
        prediction_payload=_payload(consensus_point=96.0),
        guidance_rows=_guidance(nws_point=104.0),
    )

    assert corrections.empty
    assert corrected["bin_lower_f"].tolist() == [99, 100, 101]


def test_lone_outlier_correction_requires_threshold() -> None:
    corrected, corrections = apply_lone_outlier_correction(
        _predictions(),
        prediction_payload=_payload(consensus_point=98.0),
        guidance_rows=_guidance(nws_point=98.0),
    )

    assert corrections.empty
    assert corrected["point_f"].tolist() == [100.0, 100.0, 100.0]


def test_write_lone_outlier_correction_writes_packet(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    payload_path = tmp_path / "predictions.json"
    guidance_path = tmp_path / "guidance.csv"
    out_dir = tmp_path / "out"
    _predictions().to_csv(predictions_path, index=False)
    payload_path.write_text(json.dumps(_payload()), encoding="utf-8")
    _guidance().to_csv(guidance_path, index=False)

    result = write_lone_outlier_correction(
        predictions_path=predictions_path,
        prediction_json_path=payload_path,
        guidance_path=guidance_path,
        output_dir=out_dir,
    )

    assert result.manifest["correction_count"] == 1
    assert (out_dir / "predictions_nowcast.csv").exists()
    assert (out_dir / "lone_outlier_corrections.csv").exists()
    assert (out_dir / "lone_outlier_manifest.json").exists()

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.heat_regime_correction import (
    DEFAULT_HEAT_REGIME_CORRECTIONS,
    apply_heat_regime_correction,
    write_heat_regime_correction,
)


def _predictions(city: str = "phoenix", point: float = 100.0) -> pd.DataFrame:
    rows = []
    for degree, probability in [(99, 0.25), (100, 0.5), (101, 0.25)]:
        rows.append(
            {
                "model_version": "mainline-nowcast-v1",
                "city": city,
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


def test_heat_regime_correction_adds_city_hot_bias() -> None:
    corrected, corrections = apply_heat_regime_correction(_predictions())

    assert len(corrections) == 1
    correction = corrections.iloc[0]
    assert correction["warm_threshold_f"] == DEFAULT_HEAT_REGIME_CORRECTIONS["phoenix"][0]
    assert correction["correction_f"] == DEFAULT_HEAT_REGIME_CORRECTIONS["phoenix"][1]
    assert correction["corrected_point_f"] == pytest.approx(101.9)
    assert corrected["point_f"].nunique() == 1
    assert corrected["point_f"].iloc[0] == pytest.approx(101.9)
    assert set(corrected["model_version"]) == {"mainline-nowcast-v1-heat-candidate"}
    assert corrected["calibrated_probability"].sum() == pytest.approx(1.0)
    assert "heat_regime_corrected" in corrected.iloc[0]["weather_reason_codes"]


def test_heat_regime_correction_does_not_fire_below_city_threshold() -> None:
    corrected, corrections = apply_heat_regime_correction(_predictions(point=94.0))

    assert corrections.empty
    assert corrected["point_f"].tolist() == [94.0, 94.0, 94.0]


def test_heat_regime_correction_fires_for_phoenix_mild_hot_regime() -> None:
    corrected, corrections = apply_heat_regime_correction(_predictions(point=95.7))

    assert len(corrections) == 1
    assert corrections.iloc[0]["warm_threshold_f"] == 95.0
    assert corrected["point_f"].iloc[0] == pytest.approx(97.6)


def test_heat_regime_correction_fires_for_miami_warm_regime() -> None:
    corrected, corrections = apply_heat_regime_correction(
        _predictions(city="miami", point=86.3),
    )

    assert len(corrections) == 1
    assert corrections.iloc[0]["warm_threshold_f"] == 85.0
    assert corrected["point_f"].iloc[0] == pytest.approx(87.1)


def test_heat_regime_correction_can_apply_negative_city_bias() -> None:
    corrected, corrections = apply_heat_regime_correction(
        _predictions(city="nyc", point=81.0),
    )

    assert len(corrections) == 1
    correction = corrections.iloc[0]
    assert correction["correction_f"] == -1.1
    assert correction["corrected_point_f"] == pytest.approx(79.9)
    assert corrected["point_f"].nunique() == 1
    assert corrected["point_f"].iloc[0] == pytest.approx(79.9)


def test_write_heat_regime_correction_writes_packet(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    out_dir = tmp_path / "out"
    _predictions().to_csv(predictions_path, index=False)

    result = write_heat_regime_correction(
        predictions_path=predictions_path,
        output_dir=out_dir,
    )

    assert result.manifest["correction_count"] == 1
    assert (out_dir / "predictions_nowcast.csv").exists()
    assert (out_dir / "heat_corrections.csv").exists()
    assert (out_dir / "heat_regime_manifest.json").exists()

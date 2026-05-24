import pandas as pd

from src.models.nowcast_report import build_nowcast_report, summarize_nowcast_predictions


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": "boston",
                "platform": "kalshi",
                "market_type": "high",
                "station_id": "KBOS",
                "target_date": "2026-05-24",
                "decision_time_label": "10",
                "source_policy": "gfs_ens",
                "point_f": 56,
                "q10_f": 54,
                "q50_f": 56,
                "q90_f": 58,
                "bin_label": "55",
                "calibrated_probability": 0.55,
                "nowcast_veto_flag": False,
                "weather_reason_codes": "",
                "station_rule_confidence": "high",
                "source_independence_score": 1.0,
            },
            {
                "city": "boston",
                "platform": "kalshi",
                "market_type": "high",
                "station_id": "KBOS",
                "target_date": "2026-05-24",
                "decision_time_label": "10",
                "source_policy": "gfs_ens",
                "point_f": 56,
                "q10_f": 54,
                "q50_f": 56,
                "q90_f": 58,
                "bin_label": "56",
                "calibrated_probability": 0.25,
                "nowcast_veto_flag": False,
                "weather_reason_codes": "",
                "station_rule_confidence": "high",
                "source_independence_score": 1.0,
            },
        ]
    )


def test_summarize_nowcast_predictions_returns_city_summary() -> None:
    summary = summarize_nowcast_predictions(_rows())

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["city"] == "boston"
    assert row["top_bin_label"] == "55"
    assert row["second_bin_label"] == "56"
    assert row["priority"] == "high"


def test_summarize_nowcast_predictions_veto_wins_priority() -> None:
    rows = _rows()
    rows["nowcast_veto_flag"] = True
    rows["weather_reason_codes"] = "high_so_far_exceeds_model_point"

    summary = summarize_nowcast_predictions(rows)

    assert summary.iloc[0]["priority"] == "veto"
    assert summary.iloc[0]["weather_reason_codes"] == "high_so_far_exceeds_model_point"


def test_build_nowcast_report_marks_weather_only() -> None:
    result = build_nowcast_report(_rows())

    assert "Weather-Only Nowcast Report" in result.markdown
    assert "not a trading signal" in result.markdown
    assert result.manifest["row_counts"]["summary_rows"] == 1

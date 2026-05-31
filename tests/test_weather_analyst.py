import pandas as pd

from src.models.weather_analyst import (
    build_weather_analyst_packet,
    summarize_weather_analyst_rows,
)


def test_weather_analyst_marks_clean_row_when_weather_inputs_align() -> None:
    rows = summarize_weather_analyst_rows(_summary(), guidance_comparison=_guidance(1.0))

    assert rows.iloc[0]["desk_priority"] == "clean"
    assert rows.iloc[0]["risk_flags"] == ""
    assert "clean" in rows.iloc[0]["analyst_note"]


def test_weather_analyst_marks_nws_divergence_for_veto() -> None:
    rows = summarize_weather_analyst_rows(_summary(), guidance_comparison=_guidance(3.2))

    assert rows.iloc[0]["desk_priority"] == "veto"
    assert "nws_divergent" in rows.iloc[0]["risk_flags"]
    assert "divergence clears" in rows.iloc[0]["analyst_note"]


def test_weather_analyst_veto_overrides_other_flags() -> None:
    summary = _summary()
    summary.loc[0, "priority"] = "veto"
    summary.loc[0, "nowcast_veto_flag"] = True

    rows = summarize_weather_analyst_rows(summary, guidance_comparison=_guidance(0.0))

    assert rows.iloc[0]["desk_priority"] == "veto"
    assert "weather_veto" in rows.iloc[0]["risk_flags"]


def test_weather_analyst_packet_renders_weather_only_markdown() -> None:
    packet = build_weather_analyst_packet(_summary(), guidance_comparison=_guidance(2.5))

    assert "Weather-only risk triage" in packet.markdown
    assert "private audit" in packet.markdown
    assert packet.manifest["row_counts"]["analyst_rows"] == 1


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": "chicago",
                "platform": "kalshi",
                "market_type": "high",
                "station_id": "KMDW",
                "target_date": "2026-05-24",
                "decision_time_label": "10",
                "source_policy": "gfs_ens",
                "point_f": 75.0,
                "q10_f": 73.0,
                "q90_f": 77.0,
                "top_bin_label": "75",
                "top_bin_probability": 0.5,
                "second_bin_label": "74",
                "second_bin_probability": 0.3,
                "nowcast_veto_flag": False,
                "weather_reason_codes": "",
                "station_rule_confidence": "high",
                "source_independence_score": 1.0,
                "priority": "high",
            }
        ]
    )


def _guidance(delta: float) -> pd.DataFrame:
    agreement = "aligned"
    if delta >= 3:
        agreement = "divergent"
    elif delta > 2:
        agreement = "watch"
    return pd.DataFrame(
        [
            {
                "city": "chicago",
                "platform": "kalshi",
                "market_type": "high",
                "station_id": "KMDW",
                "target_date": "2026-05-24",
                "decision_time_label": "10",
                "model_point_f": 75.0,
                "nws_guidance_point_f": 75.0 - delta,
                "model_minus_nws_f": delta,
                "abs_model_minus_nws_f": abs(delta),
                "model_vs_nws_direction": "model_hotter" if delta > 0 else "aligned",
                "guidance_agreement": agreement,
                "model_q10_f": 73.0,
                "model_q90_f": 77.0,
                "nws_available_ts_utc": "2026-05-24T14:30:00+00:00",
                "nws_issue_ts_utc": "2026-05-24T14:30:00+00:00",
                "priority": "high",
            }
        ]
    )

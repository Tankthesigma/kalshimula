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
    assert "weather_analyst_clean_rows.csv" in packet.markdown
    assert "private audit" in packet.markdown
    assert packet.manifest["row_counts"]["analyst_rows"] == 1
    assert packet.manifest["row_counts"]["clean_rows"] == 1
    assert packet.manifest["priority_counts"] == {"clean": 1, "review": 0, "veto": 0}
    assert packet.manifest["clean_cities"] == ["chicago"]


def test_weather_analyst_packet_tracks_zero_clean_rows() -> None:
    packet = build_weather_analyst_packet(_summary(), guidance_comparison=_guidance(3.2))

    assert packet.manifest["row_counts"]["clean_rows"] == 0
    assert packet.manifest["priority_counts"] == {"clean": 0, "review": 0, "veto": 1}
    assert packet.manifest["clean_cities"] == []


def test_weather_analyst_marks_uncalibrated_source_policy_as_review() -> None:
    summary = _summary()
    summary.loc[0, "source_policy"] = "openmeteo_naive"

    rows = summarize_weather_analyst_rows(
        summary,
        guidance_comparison=_guidance(1.0),
        calibration_coverage={("chicago", "gfs_ens")},
    )

    assert rows.iloc[0]["desk_priority"] == "review"
    assert rows.iloc[0]["calibration_supported"] == "no"
    assert "uncalibrated_source_policy" in rows.iloc[0]["risk_flags"]
    assert "lacks bias/interval calibration coverage" in rows.iloc[0]["analyst_note"]


def test_weather_analyst_keeps_calibrated_source_policy_clean() -> None:
    rows = summarize_weather_analyst_rows(
        _summary(),
        guidance_comparison=_guidance(1.0),
        calibration_coverage={("chicago", "gfs_ens")},
    )

    assert rows.iloc[0]["desk_priority"] == "clean"
    assert rows.iloc[0]["calibration_supported"] == "yes"


def test_weather_analyst_keeps_diffuse_distribution_clean_with_caution() -> None:
    summary = _summary()
    summary.loc[0, "top_bin_probability"] = 0.3

    rows = summarize_weather_analyst_rows(
        summary,
        guidance_comparison=_guidance(1.0),
        calibration_coverage={("chicago", "gfs_ens")},
    )

    assert rows.iloc[0]["desk_priority"] == "clean"
    assert "diffuse_distribution" in rows.iloc[0]["risk_flags"]
    assert "distribution is broad" in rows.iloc[0]["analyst_note"]


def test_weather_analyst_packet_counts_uncalibrated_rows() -> None:
    summary = _summary()
    summary.loc[0, "source_policy"] = "openmeteo_naive"

    packet = build_weather_analyst_packet(
        summary,
        guidance_comparison=_guidance(1.0),
        calibration_coverage={("chicago", "gfs_ens")},
    )

    assert packet.manifest["row_counts"]["uncalibrated_rows"] == 1
    assert "source_policy" in packet.markdown
    assert "calibrated" in packet.markdown


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

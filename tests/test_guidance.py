import pandas as pd

from src.models.guidance import (
    build_guidance_diagnostics,
    latest_guidance_as_of,
    normalize_guidance_rows,
    summarize_guidance_accuracy,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": "NYC",
                "source": "NBM",
                "station_id": "KNYC",
                "market_type": "high",
                "target_date": "2026-05-24",
                "issue_ts_utc": "2026-05-24T06:00:00Z",
                "valid_ts_utc": "2026-05-25T00:00:00Z",
                "available_ts_utc": "2026-05-24T06:20:00Z",
                "guidance_point_f": 70,
                "guidance_q10_f": 68,
                "guidance_q50_f": 70,
                "guidance_q90_f": 72,
                "actual_high_f": 71,
            },
            {
                "city": "NYC",
                "source": "NBM",
                "station_id": "KNYC",
                "market_type": "high",
                "target_date": "2026-05-24",
                "issue_ts_utc": "2026-05-24T12:00:00Z",
                "valid_ts_utc": "2026-05-25T00:00:00Z",
                "available_ts_utc": "2026-05-24T12:20:00Z",
                "guidance_point_f": 72,
                "guidance_q10_f": 70,
                "guidance_q50_f": 72,
                "guidance_q90_f": 74,
                "actual_high_f": 71,
            },
            {
                "city": "NYC",
                "source": "NBM",
                "station_id": "KNYC",
                "market_type": "high",
                "target_date": "2026-05-24",
                "issue_ts_utc": "2026-05-24T18:00:00Z",
                "valid_ts_utc": "2026-05-25T00:00:00Z",
                "available_ts_utc": "2026-05-24T18:20:00Z",
                "guidance_point_f": 99,
                "actual_high_f": 71,
            },
        ]
    )


def test_normalize_guidance_rows_enforces_schema() -> None:
    rows = normalize_guidance_rows(_rows())

    assert rows["city"].unique().tolist() == ["nyc"]
    assert rows["source"].unique().tolist() == ["nbm"]
    assert rows["station_id"].unique().tolist() == ["KNYC"]
    assert rows.loc[0, "available_ts_utc"] == "2026-05-24T06:20:00+00:00"


def test_latest_guidance_as_of_excludes_future_available_rows() -> None:
    latest = latest_guidance_as_of(
        _rows(),
        as_of_ts="2026-05-24T13:00:00Z",
        target_date="2026-05-24",
    )

    assert len(latest) == 1
    assert latest.iloc[0]["guidance_point_f"] == 72
    assert latest.iloc[0]["available_ts_utc"] == "2026-05-24T12:20:00+00:00"


def test_summarize_guidance_accuracy_scores_settled_rows() -> None:
    summary = summarize_guidance_accuracy(_rows().iloc[:2])

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n"] == 2
    assert row["mae"] == 1.0
    assert row["bias"] == 0.0
    assert row["q10_q90_coverage"] == 1.0
    assert row["mean_interval_width_f"] == 4.0


def test_build_guidance_diagnostics_renders_weather_only_report() -> None:
    result = build_guidance_diagnostics(
        _rows(),
        as_of_ts="2026-05-24T13:00:00Z",
        target_date="2026-05-24",
    )

    assert "Professional Guidance Diagnostics" in result.report
    assert "No market prices" in result.report
    assert result.manifest["row_counts"]["latest_rows"] == 1

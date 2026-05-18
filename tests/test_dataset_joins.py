from datetime import date

from src.datasets.joins import ActualRecord, ForecastRecord, join_forecasts_to_actuals


def test_join_forecasts_to_actuals_creates_examples_for_matching_records() -> None:
    examples = join_forecasts_to_actuals(
        forecasts=[
            ForecastRecord("denver", date(2025, 1, 1), "nws", 70),
            ForecastRecord("denver", date(2025, 1, 2), "nws", 71),
        ],
        actuals=[ActualRecord("denver", date(2025, 1, 1), 68)],
    )

    assert len(examples) == 1
    assert examples[0].error_f == 2.0


def test_join_forecasts_to_actuals_skips_missing_values() -> None:
    examples = join_forecasts_to_actuals(
        forecasts=[ForecastRecord("denver", date(2025, 1, 1), "nws", None)],
        actuals=[ActualRecord("denver", date(2025, 1, 1), 68)],
    )

    assert examples == []


def test_join_forecasts_to_actuals_skips_missing_actuals() -> None:
    examples = join_forecasts_to_actuals(
        forecasts=[ForecastRecord("denver", date(2025, 1, 1), "nws", 70)],
        actuals=[ActualRecord("denver", date(2025, 1, 1), None)],
    )

    assert examples == []

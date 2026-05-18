import pandas as pd
import pytest

from src.models.ensemble import naive_forecast_from_members


def test_naive_forecast_summarizes_members_and_probabilities() -> None:
    members = pd.DataFrame(
        {
            "source": ["a", "a", "b", "b"],
            "temp_f": [70.0, 71.0, 71.0, 72.0],
        }
    )

    forecast = naive_forecast_from_members(members)

    assert forecast.n_members == 4
    assert forecast.point_f == pytest.approx(71.0)
    assert forecast.p50_f == pytest.approx(71.0)
    assert forecast.bin_probs == {70: 0.25, 71: 0.5, 72: 0.25}
    assert forecast.per_source_counts == {"a": 2, "b": 2}


def test_naive_forecast_rejects_empty_members() -> None:
    members = pd.DataFrame({"source": [], "temp_f": []})

    with pytest.raises(ValueError):
        naive_forecast_from_members(members)

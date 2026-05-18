from datetime import date

import pandas as pd

from src.datasets.features import add_error_features, training_examples_to_xy
from src.datasets.training import make_training_example


def test_training_examples_to_xy_returns_feature_matrix_and_target() -> None:
    example = make_training_example(
        city="denver",
        target_date=date(2025, 1, 1),
        source="nws",
        forecast_high_f=70,
        actual_high_f=68,
    )

    x, y = training_examples_to_xy([example])

    assert list(x.columns) == ["forecast_high_f"]
    assert x.iloc[0]["forecast_high_f"] == 70.0
    assert y.iloc[0] == 68.0


def test_add_error_features_does_not_mutate_input() -> None:
    df = pd.DataFrame({"forecast_high_f": [70.0], "source": ["nws"]})

    out = add_error_features(df)

    assert "forecast_high_f_squared" not in df.columns
    assert out.iloc[0]["forecast_high_f_squared"] == 4900.0
    assert out.iloc[0]["source_code"] == 0

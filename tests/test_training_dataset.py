from datetime import date

from src.datasets.training import examples_to_dataframe, make_training_example


def test_make_training_example_computes_signed_error() -> None:
    example = make_training_example(
        city="denver",
        target_date=date(2025, 1, 2),
        source="gfs_ens",
        forecast_high_f=70,
        actual_high_f=67.5,
    )

    assert example.city == "denver"
    assert example.target_date == date(2025, 1, 2)
    assert example.error_f == 2.5


def test_examples_to_dataframe_has_stable_columns() -> None:
    example = make_training_example(
        city="chicago",
        target_date=date(2025, 2, 3),
        source="nws",
        forecast_high_f=30,
        actual_high_f=34,
    )

    df = examples_to_dataframe([example])

    assert list(df.columns) == [
        "city",
        "target_date",
        "source",
        "forecast_high_f",
        "actual_high_f",
        "error_f",
    ]
    assert df.iloc[0].to_dict() == {
        "city": "chicago",
        "target_date": date(2025, 2, 3),
        "source": "nws",
        "forecast_high_f": 30.0,
        "actual_high_f": 34.0,
        "error_f": -4.0,
    }

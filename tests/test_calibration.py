import pytest

from src.models.calibration import calibration_table


def test_calibration_table_buckets_probabilities() -> None:
    table = calibration_table(
        predicted_probabilities=[0.05, 0.15, 0.85, 0.95],
        outcomes=[False, True, True, True],
        n_buckets=2,
    )

    assert list(table.columns) == [
        "bucket_start",
        "bucket_end",
        "n",
        "mean_predicted_probability",
        "observed_frequency",
    ]
    assert table.iloc[0]["n"] == 2
    assert table.iloc[0]["observed_frequency"] == pytest.approx(0.5)
    assert table.iloc[1]["n"] == 2
    assert table.iloc[1]["observed_frequency"] == pytest.approx(1.0)


def test_calibration_table_returns_empty_stable_shape() -> None:
    table = calibration_table([], [], n_buckets=5)

    assert table.empty
    assert list(table.columns) == [
        "bucket_start",
        "bucket_end",
        "n",
        "mean_predicted_probability",
        "observed_frequency",
    ]


def test_calibration_table_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        calibration_table([0.1], [True, False])
    with pytest.raises(ValueError):
        calibration_table([0.1], [True], n_buckets=0)

import pandas as pd
import pytest

from src.models.baseline_training import (
    EVALUATION_COLUMNS,
    evaluate_corrected_predictions,
    train_bias_baseline,
    write_baseline_training_outputs,
)
from src.models.bias import apply_bias_correction, fit_bias_table


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"city": "denver", "source": "openmeteo", "point_f": 70, "actual_high_f": 68},
            {"city": "denver", "source": "openmeteo", "point_f": 72, "actual_high_f": 71},
            {"city": "chicago", "source": "openmeteo", "point_f": 30, "actual_high_f": 34},
        ]
    )


def test_train_bias_baseline_returns_bias_and_evaluation() -> None:
    result = train_bias_baseline(_rows())

    assert len(result.bias_table) == 2
    assert list(result.evaluation.columns) == EVALUATION_COLUMNS
    denver = result.evaluation[result.evaluation["city"] == "denver"].iloc[0]
    assert denver["mae_corrected"] <= denver["mae_raw"]


def test_evaluate_corrected_predictions_groups_metrics() -> None:
    bias_table = fit_bias_table(_rows())
    corrected = apply_bias_correction(_rows(), bias_table)

    evaluation = evaluate_corrected_predictions(corrected)

    chicago = evaluation[evaluation["city"] == "chicago"].iloc[0]
    assert chicago["n"] == 1
    assert chicago["bias_corrected"] == pytest.approx(0.0)


def test_evaluate_corrected_predictions_returns_empty_stable_shape() -> None:
    rows = pd.DataFrame(columns=["city", "source", "actual_high_f", "point_f", "corrected_point_f"])

    evaluation = evaluate_corrected_predictions(rows)

    assert evaluation.empty
    assert list(evaluation.columns) == EVALUATION_COLUMNS


def test_evaluate_corrected_predictions_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        evaluate_corrected_predictions(pd.DataFrame({"city": ["denver"]}))


def test_write_baseline_training_outputs(tmp_path) -> None:
    input_path = tmp_path / "rows.csv"
    bias_out = tmp_path / "model" / "bias.csv"
    evaluation_out = tmp_path / "model" / "evaluation.csv"
    _rows().to_csv(input_path, index=False)

    result = write_baseline_training_outputs(
        input_path=input_path,
        bias_out=bias_out,
        evaluation_out=evaluation_out,
    )

    assert len(result.bias_table) == 2
    assert bias_out.exists()
    assert evaluation_out.exists()

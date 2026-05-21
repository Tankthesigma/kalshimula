import json

from src import prediction_review_cli


def _payload(*, gate_passed=True, errors=None):
    errors = errors or []
    return {
        "schema_version": "1.0",
        "generated_at": "2026-05-21T12:00:00+00:00",
        "target_date": "2026-04-30",
        "model_gate": {
            "required": True,
            "passed": gate_passed,
            "checks": [
                {
                    "name": "test_mae_corrected",
                    "value": 0.99 if gate_passed else 1.4,
                    "threshold": 1.05,
                    "passed": gate_passed,
                    "detail": "recommended bias policy held-out MAE",
                }
            ],
        },
        "n_predictions": 1,
        "n_errors": len(errors),
        "errors": errors,
        "predictions": [
            {
                "city": "denver",
                "selected_source": "gfs_ens",
                "forecast": {
                    "point_f": 46.0,
                    "n_members": 31,
                },
                "calibration": {
                    "source": "gfs_ens",
                    "corrected_point_f": 46.94,
                    "interval_lower_f": 46.08,
                    "interval_upper_f": 49.48,
                },
                "threshold_probabilities": [
                    {"threshold_f": 45, "predicted_probability": 0.986},
                    {"threshold_f": 47, "predicted_probability": 0.6},
                ],
            }
        ],
    }


def test_build_prediction_review_renders_gate_and_prediction_table() -> None:
    report = prediction_review_cli.build_prediction_review(_payload())

    assert "Prediction review" in report
    assert "Gate: PASS (required=true, checks=1)" in report
    assert "denver" in report
    assert "gfs_ens" in report
    assert "46.9" in report
    assert "P>=45F" in report
    assert "98.6%" in report


def test_prediction_review_cli_writes_report(tmp_path, capsys) -> None:
    input_path = tmp_path / "predictions.json"
    output_path = tmp_path / "review.txt"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")

    code = prediction_review_cli.main(
        ["--input", str(input_path), "--out", str(output_path)]
    )

    assert code == 0
    assert capsys.readouterr().out == ""
    assert "Gate: PASS" in output_path.read_text(encoding="utf-8")


def test_prediction_review_cli_fails_failed_gate(tmp_path, capsys) -> None:
    input_path = tmp_path / "predictions.json"
    input_path.write_text(json.dumps(_payload(gate_passed=False)), encoding="utf-8")

    code = prediction_review_cli.main(["--input", str(input_path)])

    output = capsys.readouterr().out
    assert code == 1
    assert "Gate: FAIL" in output
    assert "FAIL test_mae_corrected" in output


def test_prediction_review_cli_allow_errors_overrides_exit_code(tmp_path, capsys) -> None:
    input_path = tmp_path / "predictions.json"
    input_path.write_text(
        json.dumps(
            _payload(
                errors=[{"city": "boston", "error": "every Open-Meteo source returned empty"}]
            )
        ),
        encoding="utf-8",
    )

    code = prediction_review_cli.main(["--input", str(input_path), "--allow-errors"])

    output = capsys.readouterr().out
    assert code == 0
    assert "Errors: 1" in output
    assert "boston: every Open-Meteo source returned empty" in output

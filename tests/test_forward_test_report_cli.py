import csv
import json
from datetime import UTC, datetime

import pytest

from src import forward_test_report_cli


def _write_history(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "settled_at",
                "packet_path",
                "target_date",
                "city",
                "actual_source",
                "observed_high_f",
                "predicted_point_f",
                "predicted_corrected_point_f",
                "error_f",
                "absolute_error_f",
                "offset_f",
                "threshold_f",
                "predicted_probability",
                "raw_predicted_probability",
                "recalibration_used",
                "recalibration_scope",
                "recalibration_n",
                "outcome",
                "brier",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _row(
    *,
    target_date="2026-05-22",
    city="denver",
    source="ncei",
    error="-1.0",
    absolute_error="1.0",
    offset="0",
    brier="0.04",
):
    return {
        "settled_at": "2026-05-23T00:00:00+00:00",
        "packet_path": "packet.json",
        "target_date": target_date,
        "city": city,
        "actual_source": source,
        "observed_high_f": "73.0",
        "predicted_point_f": "70.0",
        "predicted_corrected_point_f": "72.0",
        "error_f": error,
        "absolute_error_f": absolute_error,
        "offset_f": offset,
        "threshold_f": "72",
        "predicted_probability": "0.8",
        "raw_predicted_probability": "0.75",
        "recalibration_used": "True",
        "recalibration_scope": "city_source",
        "recalibration_n": "123",
        "outcome": "True",
        "brier": brier,
    }


def test_build_forward_test_report_separates_predictions_from_thresholds(tmp_path):
    history_path = tmp_path / "history.csv"
    _write_history(
        history_path,
        [
            _row(city="denver", offset="-2", brier="0.04"),
            _row(city="denver", offset="0", brier="0.09"),
            _row(city="austin", error="2.0", absolute_error="2.0", offset="-2"),
            _row(city="austin", error="2.0", absolute_error="2.0", offset="0"),
        ],
    )

    payload = forward_test_report_cli.build_forward_test_report(
        history_path,
        generated_at=datetime(2026, 5, 23, tzinfo=UTC),
    )

    assert payload["generated_at"] == "2026-05-23T00:00:00+00:00"
    assert payload["summary"]["n_history_rows"] == 4
    assert payload["summary"]["n_predictions"] == 2
    assert payload["summary"]["n_threshold_events"] == 4
    assert payload["summary"]["n_cities"] == 2
    assert payload["summary"]["mae_corrected_f"] == 1.5
    assert payload["summary"]["bias_corrected_f"] == 0.5
    assert payload["summary"]["threshold_brier_score"] == pytest.approx(
        (0.04 + 0.09 + 0.04 + 0.04) / 4
    )
    assert payload["summary"]["actual_sources"] == {"ncei": 2}
    assert payload["daily"][0]["target_date"] == "2026-05-22"
    assert payload["daily"][0]["n_predictions"] == 2


def test_build_forward_test_report_uses_latest_duplicate_rows(tmp_path):
    history_path = tmp_path / "history.csv"
    _write_history(
        history_path,
        [
            _row(source="asos", error="5.0", absolute_error="5.0", offset="0"),
            _row(source="ncei", error="1.0", absolute_error="1.0", offset="0"),
        ],
    )

    payload = forward_test_report_cli.build_forward_test_report(history_path)

    assert payload["summary"]["n_history_rows"] == 2
    assert payload["summary"]["n_predictions"] == 1
    assert payload["summary"]["n_threshold_events"] == 1
    assert payload["summary"]["mae_corrected_f"] == 1.0
    assert payload["summary"]["actual_sources"] == {"ncei": 1}


def test_forward_test_report_cli_writes_json(tmp_path):
    history_path = tmp_path / "history.csv"
    out_path = tmp_path / "report.json"
    _write_history(history_path, [_row()])

    code = forward_test_report_cli.main(
        ["--history", str(history_path), "--out", str(out_path)]
    )

    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["summary"]["n_predictions"] == 1


def test_forward_test_report_cli_returns_failure_for_missing_history(capsys, tmp_path):
    code = forward_test_report_cli.main(["--history", str(tmp_path / "missing.csv")])

    assert code == 1
    assert "FAIL artifact_error" in capsys.readouterr().out

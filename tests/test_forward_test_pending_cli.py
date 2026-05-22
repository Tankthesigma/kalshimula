import csv
import json
from datetime import date

from src import forward_test_pending_cli


def _packet() -> dict:
    return {
        "schema_version": "1.0",
        "generated_at": "2026-05-21T12:00:00+00:00",
        "target_date": "2026-05-22",
        "predictions": [
            {
                "city": "denver",
                "threshold_probabilities": [
                    {"offset_f": -2},
                    {"offset_f": 0},
                    {"offset_f": 2},
                ],
            },
            {
                "city": "boston",
                "threshold_probabilities": [
                    {"offset_f": -2},
                    {"offset_f": 0},
                    {"offset_f": 2},
                ],
            },
        ],
    }


def _write_history(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target_date", "city", "offset_f"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_pending_status_reports_unsettled_ready_packet(tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    payload = forward_test_pending_cli.build_pending_status(
        packet_path=packet_path,
        as_of_date=date(2026, 5, 23),
    )

    assert payload["target_date"] == "2026-05-22"
    assert payload["settlement_status"] == "unsettled"
    assert payload["ready_to_settle"] is True
    assert payload["next_action"] == "run_forward_test_settle"
    assert payload["n_expected_threshold_rows"] == 6
    assert payload["n_settled_threshold_rows"] == 0
    assert payload["missing_city_offsets"] == {
        "boston": ["-2", "0", "2"],
        "denver": ["-2", "0", "2"],
    }


def test_build_pending_status_waits_for_target_date_to_pass(tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    payload = forward_test_pending_cli.build_pending_status(
        packet_path=packet_path,
        as_of_date=date(2026, 5, 22),
    )

    assert payload["settlement_status"] == "unsettled"
    assert payload["ready_to_settle"] is False
    assert payload["next_action"] == "wait_for_target_date_to_pass"


def test_build_pending_status_reports_partial_history(tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    history_path = tmp_path / "history.csv"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")
    _write_history(
        history_path,
        [
            {"target_date": "2026-05-22", "city": "denver", "offset_f": "-2"},
            {"target_date": "2026-05-22", "city": "denver", "offset_f": "0"},
            {"target_date": "2026-05-21", "city": "boston", "offset_f": "-2"},
        ],
    )

    payload = forward_test_pending_cli.build_pending_status(
        packet_path=packet_path,
        history_path=history_path,
        as_of_date=date(2026, 5, 23),
    )

    assert payload["settlement_status"] == "partial"
    assert payload["ready_to_settle"] is True
    assert payload["n_settled_threshold_rows"] == 2
    assert payload["missing_city_offsets"] == {
        "boston": ["-2", "0", "2"],
        "denver": ["2"],
    }


def test_build_pending_status_reports_fully_settled_packet(tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    history_path = tmp_path / "history.csv"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")
    _write_history(
        history_path,
        [
            {"target_date": "2026-05-22", "city": city, "offset_f": offset}
            for city in ["denver", "boston"]
            for offset in ["-2", "0", "2"]
        ],
    )

    payload = forward_test_pending_cli.build_pending_status(
        packet_path=packet_path,
        history_path=history_path,
        as_of_date=date(2026, 5, 23),
    )

    assert payload["settlement_status"] == "settled"
    assert payload["ready_to_settle"] is False
    assert payload["next_action"] == "already_settled"
    assert payload["missing_city_offsets"] == {}


def test_forward_test_pending_cli_prints_status(tmp_path, capsys) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    code = forward_test_pending_cli.main(
        [
            "--packet",
            str(packet_path),
            "--as-of-date",
            "2026-05-23",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "Ready to settle: true" in output
    assert "Next action: run_forward_test_settle" in output


def test_forward_test_pending_cli_prints_json(tmp_path, capsys) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    code = forward_test_pending_cli.main(
        [
            "--packet",
            str(packet_path),
            "--as-of-date",
            "2026-05-22",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready_to_settle"] is False

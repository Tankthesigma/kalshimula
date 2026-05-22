import json

import pytest

from src import forward_test_actuals_check_cli


def _packet() -> dict:
    return {
        "schema_version": "1.0",
        "target_date": "2026-05-22",
        "predictions": [
            {"city": "Denver"},
            {"city": "Boston"},
        ],
    }


def _write_packet(tmp_path):
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")
    return packet_path


def test_build_actuals_check_passes_complete_csv(tmp_path) -> None:
    packet_path = _write_packet(tmp_path)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date,actual_high_f,actual_source\n"
        "denver,2026-05-22,73,ncei\n"
        "boston,2026-05-22,68,ncei\n",
        encoding="utf-8",
    )

    payload = forward_test_actuals_check_cli.build_actuals_check(
        packet_path=packet_path,
        actuals_csv=actuals_path,
    )

    assert payload["passed"] is True
    assert payload["n_valid_actuals"] == 2
    assert payload["errors"] == []


def test_build_actuals_check_fails_missing_city(tmp_path) -> None:
    packet_path = _write_packet(tmp_path)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date,actual_high_f\n"
        "denver,2026-05-22,73\n",
        encoding="utf-8",
    )

    payload = forward_test_actuals_check_cli.build_actuals_check(
        packet_path=packet_path,
        actuals_csv=actuals_path,
    )

    assert payload["passed"] is False
    assert {"city": "boston", "error": "missing city for packet target_date"} in payload[
        "errors"
    ]


def test_build_actuals_check_fails_blank_and_non_numeric_actuals(tmp_path) -> None:
    packet_path = _write_packet(tmp_path)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date,actual_high_f\n"
        "denver,2026-05-22,\n"
        "boston,2026-05-22,warm\n",
        encoding="utf-8",
    )

    payload = forward_test_actuals_check_cli.build_actuals_check(
        packet_path=packet_path,
        actuals_csv=actuals_path,
    )

    assert payload["passed"] is False
    assert {"city": "denver", "error": "missing actual_high_f"} in payload["errors"]
    assert {"city": "boston", "error": "actual_high_f is not numeric"} in payload[
        "errors"
    ]


def test_build_actuals_check_fails_duplicate_city(tmp_path) -> None:
    packet_path = _write_packet(tmp_path)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date,actual_high_f\n"
        "denver,2026-05-22,73\n"
        "denver,2026-05-22,74\n"
        "boston,2026-05-22,68\n",
        encoding="utf-8",
    )

    payload = forward_test_actuals_check_cli.build_actuals_check(
        packet_path=packet_path,
        actuals_csv=actuals_path,
    )

    assert payload["passed"] is False
    assert {"city": "denver", "error": "duplicate city for packet target_date"} in payload[
        "errors"
    ]


def test_build_actuals_check_fails_extra_city_by_default(tmp_path) -> None:
    packet_path = _write_packet(tmp_path)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date,actual_high_f\n"
        "denver,2026-05-22,73\n"
        "boston,2026-05-22,68\n"
        "miami,2026-05-22,88\n",
        encoding="utf-8",
    )

    payload = forward_test_actuals_check_cli.build_actuals_check(
        packet_path=packet_path,
        actuals_csv=actuals_path,
    )

    assert payload["passed"] is False
    assert {"city": "miami", "error": "extra city for packet target_date"} in payload[
        "errors"
    ]


def test_build_actuals_check_can_allow_extra_city(tmp_path) -> None:
    packet_path = _write_packet(tmp_path)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date,actual_high_f\n"
        "denver,2026-05-22,73\n"
        "boston,2026-05-22,68\n"
        "miami,2026-05-22,88\n",
        encoding="utf-8",
    )

    payload = forward_test_actuals_check_cli.build_actuals_check(
        packet_path=packet_path,
        actuals_csv=actuals_path,
        allow_extra=True,
    )

    assert payload["passed"] is True
    assert payload["extra_cities"] == ["miami"]


def test_build_actuals_check_ignores_wrong_date_rows(tmp_path) -> None:
    packet_path = _write_packet(tmp_path)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date,actual_high_f\n"
        "denver,2026-05-21,73\n"
        "boston,2026-05-22,68\n",
        encoding="utf-8",
    )

    payload = forward_test_actuals_check_cli.build_actuals_check(
        packet_path=packet_path,
        actuals_csv=actuals_path,
    )

    assert payload["passed"] is False
    assert {"city": "denver", "error": "missing city for packet target_date"} in payload[
        "errors"
    ]


def test_build_actuals_check_rejects_missing_required_columns(tmp_path) -> None:
    packet_path = _write_packet(tmp_path)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date\n"
        "denver,2026-05-22\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="actual_high_f"):
        forward_test_actuals_check_cli.build_actuals_check(
            packet_path=packet_path,
            actuals_csv=actuals_path,
        )


def test_forward_test_actuals_check_cli_prints_json(tmp_path, capsys) -> None:
    packet_path = _write_packet(tmp_path)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date,actual_high_f\n"
        "denver,2026-05-22,73\n"
        "boston,2026-05-22,68\n",
        encoding="utf-8",
    )

    code = forward_test_actuals_check_cli.main(
        ["--packet", str(packet_path), "--actuals-csv", str(actuals_path), "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["passed"] is True
    assert payload["n_valid_actuals"] == 2


def test_forward_test_actuals_check_cli_returns_nonzero_for_incomplete_csv(
    tmp_path, capsys
) -> None:
    packet_path = _write_packet(tmp_path)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date,actual_high_f\n"
        "denver,2026-05-22,\n",
        encoding="utf-8",
    )

    code = forward_test_actuals_check_cli.main(
        ["--packet", str(packet_path), "--actuals-csv", str(actuals_path)]
    )

    output = capsys.readouterr().out
    assert code == 1
    assert "Actuals CSV: FAIL" in output
    assert "denver: missing actual_high_f" in output
    assert "boston: missing city for packet target_date" in output

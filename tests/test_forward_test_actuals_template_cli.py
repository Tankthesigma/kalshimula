import csv
import json

import pytest

from src import forward_test_actuals_template_cli


def _packet() -> dict:
    return {
        "schema_version": "1.0",
        "target_date": "2026-05-22",
        "predictions": [
            {"city": "Denver"},
            {"city": "boston"},
            {"city": "denver"},
            {"city": ""},
        ],
    }


def test_build_actuals_template_rows_dedupes_cities() -> None:
    rows = forward_test_actuals_template_cli.build_actuals_template_rows(_packet())

    assert rows == [
        {
            "city": "denver",
            "target_date": "2026-05-22",
            "actual_high_f": "",
            "actual_source": "",
        },
        {
            "city": "boston",
            "target_date": "2026-05-22",
            "actual_high_f": "",
            "actual_source": "",
        },
    ]


def test_build_actuals_template_rows_rejects_missing_target_date() -> None:
    with pytest.raises(ValueError, match="target_date"):
        forward_test_actuals_template_cli.build_actuals_template_rows(
            {"predictions": []}
        )


def test_write_actuals_template_writes_csv(tmp_path) -> None:
    out_path = tmp_path / "actuals.csv"

    forward_test_actuals_template_cli.write_actuals_template(
        forward_test_actuals_template_cli.build_actuals_template_rows(_packet()),
        out_path,
    )

    with out_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "city": "denver",
            "target_date": "2026-05-22",
            "actual_high_f": "",
            "actual_source": "",
        },
        {
            "city": "boston",
            "target_date": "2026-05-22",
            "actual_high_f": "",
            "actual_source": "",
        },
    ]


def test_forward_test_actuals_template_cli_writes_output(tmp_path, capsys) -> None:
    packet_path = tmp_path / "packet.json"
    out_path = tmp_path / "actuals.csv"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    code = forward_test_actuals_template_cli.main(
        ["--packet", str(packet_path), "--out", str(out_path)]
    )

    assert code == 0
    assert "2 cities" in capsys.readouterr().out
    assert out_path.exists()

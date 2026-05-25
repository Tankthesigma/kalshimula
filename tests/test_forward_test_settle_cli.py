import csv
import json
from datetime import UTC, date, datetime

from src import forward_test_settle_cli
from src.fetchers.ncei import NceiDailyHigh


def _packet(target_date: str = "2026-05-22") -> dict:
    return {
        "schema_version": "1.0",
        "generated_at": "2026-05-21T12:00:00+00:00",
        "target_date": target_date,
        "predictions": [
            {
                "city": "denver",
                "target_date": target_date,
                "forecast": {"point_f": 70.0, "p10_f": 68.0, "p90_f": 74.0},
                "calibration": {
                    "corrected_point_f": 72.0,
                    "interval_lower_f": 69.0,
                    "interval_upper_f": 76.0,
                },
                "threshold_probabilities": [
                    {
                        "offset_f": -2,
                        "threshold_f": 70,
                        "predicted_probability": 0.80,
                        "raw_predicted_probability": 0.75,
                        "recalibration_used": True,
                        "recalibration_scope": "city_source",
                        "recalibration_n": 123,
                    },
                    {
                        "offset_f": 2,
                        "threshold_f": 74,
                        "predicted_probability": 0.25,
                    },
                ],
            }
        ],
    }


def test_build_settlement_payload_prefers_ncei(monkeypatch, tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    def fake_ncei(station, target):
        return NceiDailyHigh(
            station=station.ghcnd_bare,
            target_date=target,
            high_f=75.0,
        )

    monkeypatch.setattr("src.forward_test_settle_cli.ncei.fetch_daily_high", fake_ncei)
    monkeypatch.setattr(
        "src.forward_test_settle_cli.asos.fetch_asos_observation_csv",
        lambda station, start, end: (_ for _ in ()).throw(AssertionError("ASOS unused")),
    )

    payload, code = forward_test_settle_cli.build_settlement_payload(
        packet_path=packet_path,
        target=date(2026, 5, 22),
        generated_at=datetime(2026, 5, 23, 1, 2, 3, tzinfo=UTC),
    )

    assert code == 0
    assert payload["generated_at"] == "2026-05-23T01:02:03+00:00"
    assert payload["n_rows"] == 1
    assert payload["summary"]["n_settled"] == 1
    assert payload["summary"]["n_errors"] == 0
    assert payload["summary"]["mae_corrected_f"] == 3.0
    assert payload["summary"]["interval_coverage"] == 1.0
    assert payload["summary"]["actual_sources"] == ["ncei"]
    row = payload["rows"][0]
    assert row["actual_source"] == "ncei"
    assert row["observed_high_f"] == 75.0
    assert row["predicted_point_f"] == 70.0
    assert row["predicted_corrected_point_f"] == 72.0
    assert row["p10_f"] == 68.0
    assert row["p90_f"] == 74.0
    assert row["interval_lower_f"] == 69.0
    assert row["interval_upper_f"] == 76.0
    assert row["error_f"] == -3.0
    assert row["absolute_error_f"] == 3.0
    assert row["threshold_outcomes"][0]["outcome"] is True
    assert row["threshold_outcomes"][1]["outcome"] is True

    history_rows = forward_test_settle_cli.settlement_history_rows(payload)
    assert len(history_rows) == 2
    assert history_rows[0]["city"] == "denver"
    assert history_rows[0]["predicted_probability"] == 0.8
    assert history_rows[0]["outcome"] is True
    assert history_rows[0]["brier"] == (0.8 - 1.0) ** 2


def test_build_settlement_payload_falls_back_to_asos(monkeypatch, tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    def fake_ncei(station, target):
        return NceiDailyHigh(
            station=station.ghcnd_bare,
            target_date=target,
            high_f=None,
        )

    monkeypatch.setattr("src.forward_test_settle_cli.ncei.fetch_daily_high", fake_ncei)
    def fake_asos(station, start, end):
        assert start == date(2026, 5, 22)
        assert end == date(2026, 5, 23)
        return (
            "station,valid,tmpf\n"
            "DEN,2026-05-22 12:00,71.0\n"
            "DEN,2026-05-22 15:00,73.5\n"
        )

    monkeypatch.setattr(
        "src.forward_test_settle_cli.asos.fetch_asos_observation_csv",
        fake_asos,
    )

    payload, code = forward_test_settle_cli.build_settlement_payload(
        packet_path=packet_path,
        target=date(2026, 5, 22),
    )

    assert code == 0
    assert payload["rows"][0]["actual_source"] == "asos"
    assert payload["rows"][0]["observed_high_f"] == 73.5


def test_build_settlement_payload_uses_offline_actuals(tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    payload, code = forward_test_settle_cli.build_settlement_payload(
        packet_path=packet_path,
        target=date(2026, 5, 22),
        actuals={
            "denver": forward_test_settle_cli.ObservedHigh(
                high_f=73.0,
                source="manual_csv",
            )
        },
    )

    assert code == 0
    assert payload["summary"]["actual_sources"] == ["manual_csv"]
    assert payload["summary"]["mae_corrected_f"] == 1.0
    assert payload["summary"]["threshold_brier_score"] == ((0.8 - 1.0) ** 2 + 0.25**2) / 2
    assert payload["rows"][0]["actual_source"] == "manual_csv"
    assert payload["rows"][0]["observed_high_f"] == 73.0


def test_read_actuals_csv_filters_to_target_date(tmp_path) -> None:
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "city,target_date,actual_high_f,actual_source\n"
        "denver,2026-05-21,99,ncei\n"
        "denver,2026-05-22,73,manual\n",
        encoding="utf-8",
    )

    actuals = forward_test_settle_cli._read_actuals_csv(
        actuals_path,
        date(2026, 5, 22),
    )

    assert actuals == {
        "denver": forward_test_settle_cli.ObservedHigh(
            high_f=73.0,
            source="manual",
        )
    }


def test_forward_test_settle_cli_writes_json_and_history(monkeypatch, tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    monkeypatch.setattr(
        "src.forward_test_settle_cli._fetch_observed_high",
        lambda station, target: forward_test_settle_cli.ObservedHigh(
            high_f=71.0,
            source="ncei",
        ),
    )

    code = forward_test_settle_cli.main(
        [
            "--packet",
            str(packet_path),
            "--target-date",
            "2026-05-22",
            "--out-dir",
            str(tmp_path / "forward"),
        ]
    )

    out_path = tmp_path / "forward" / "2026-05-22_settlement.json"
    history_path = tmp_path / "forward" / "history.csv"
    report_path = tmp_path / "forward" / "report.json"
    assert code == 0
    assert json.loads(out_path.read_text(encoding="utf-8"))["n_rows"] == 1
    history = history_path.read_text(encoding="utf-8")
    assert "city,target_date" not in history
    assert "denver" in history
    assert "69.0" in history
    assert "76.0" in history
    assert "0.8" in history
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["n_predictions"] == 1
    assert report["summary"]["n_threshold_events"] == 2


def test_history_columns_are_unique() -> None:
    assert len(forward_test_settle_cli.HISTORY_COLUMNS) == len(
        set(forward_test_settle_cli.HISTORY_COLUMNS)
    )


def test_append_history_atomic_replaces_existing_city_date_offset(tmp_path) -> None:
    history_path = tmp_path / "history.csv"
    old_row = {key: None for key in forward_test_settle_cli.HISTORY_COLUMNS}
    old_row.update(
        {
            "settled_at": "2026-05-23T00:00:00+00:00",
            "target_date": "2026-05-22",
            "city": "denver",
            "actual_source": "asos",
            "offset_f": "0",
            "threshold_f": "72",
            "predicted_probability": "0.5",
            "outcome": "True",
            "brier": "0.25",
        }
    )
    new_row = {key: None for key in forward_test_settle_cli.HISTORY_COLUMNS}
    new_row.update(
        {
            "settled_at": "2026-05-24T00:00:00+00:00",
            "target_date": "2026-05-22",
            "city": "denver",
            "actual_source": "ncei",
            "offset_f": "0",
            "threshold_f": "72",
            "predicted_probability": "0.5",
            "outcome": "False",
            "brier": "0.25",
        }
    )
    other_offset = {
        **new_row,
        "actual_source": "asos",
        "offset_f": "2",
        "threshold_f": "74",
    }

    forward_test_settle_cli.append_history_atomic(history_path, [old_row, other_offset])
    forward_test_settle_cli.append_history_atomic(history_path, [new_row])

    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {(row["city"], row["offset_f"]) for row in rows} == {
        ("denver", "0"),
        ("denver", "2"),
    }
    replaced = next(row for row in rows if row["offset_f"] == "0")
    assert replaced["actual_source"] == "ncei"
    assert replaced["settled_at"] == "2026-05-24T00:00:00+00:00"


def test_forward_test_settle_cli_can_skip_report(monkeypatch, tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")
    monkeypatch.setattr(
        "src.forward_test_settle_cli._fetch_observed_high",
        lambda station, target: forward_test_settle_cli.ObservedHigh(
            high_f=71.0,
            source="ncei",
        ),
    )

    code = forward_test_settle_cli.main(
        [
            "--packet",
            str(packet_path),
            "--target-date",
            "2026-05-22",
            "--out-dir",
            str(tmp_path / "forward"),
            "--no-report",
        ]
    )

    assert code == 0
    assert not (tmp_path / "forward" / "report.json").exists()


def test_forward_test_settle_cli_writes_custom_report_path(monkeypatch, tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    report_path = tmp_path / "reports" / "forward_report.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")
    monkeypatch.setattr(
        "src.forward_test_settle_cli._fetch_observed_high",
        lambda station, target: forward_test_settle_cli.ObservedHigh(
            high_f=71.0,
            source="ncei",
        ),
    )

    code = forward_test_settle_cli.main(
        [
            "--packet",
            str(packet_path),
            "--target-date",
            "2026-05-22",
            "--out-dir",
            str(tmp_path / "forward"),
            "--report-out",
            str(report_path),
        ]
    )

    assert code == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["summary"][
        "n_predictions"
    ] == 1


def test_forward_test_settle_cli_report_failure_does_not_override_settle_success(
    monkeypatch, tmp_path, capsys
) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")
    monkeypatch.setattr(
        "src.forward_test_settle_cli._fetch_observed_high",
        lambda station, target: forward_test_settle_cli.ObservedHigh(
            high_f=71.0,
            source="ncei",
        ),
    )
    monkeypatch.setattr(
        "src.forward_test_settle_cli.write_forward_test_report",
        lambda history_path, report_path: (_ for _ in ()).throw(
            ValueError("report failed")
        ),
    )

    code = forward_test_settle_cli.main(
        [
            "--packet",
            str(packet_path),
            "--target-date",
            "2026-05-22",
            "--out-dir",
            str(tmp_path / "forward"),
        ]
    )

    assert code == 0
    assert "Skipped forward test report: report failed" in capsys.readouterr().out


def test_forward_test_settle_cli_can_use_actuals_csv_without_fetching(
    monkeypatch, tmp_path
) -> None:
    packet_path = tmp_path / "packet.json"
    actuals_path = tmp_path / "actuals.csv"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")
    actuals_path.write_text(
        "city,target_date,actual_high_f,actual_source\n"
        "denver,2026-05-22,73,manual\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.forward_test_settle_cli._fetch_observed_high",
        lambda station, target: (_ for _ in ()).throw(AssertionError("fetch unused")),
    )

    code = forward_test_settle_cli.main(
        [
            "--packet",
            str(packet_path),
            "--target-date",
            "2026-05-22",
            "--actuals-csv",
            str(actuals_path),
            "--out-dir",
            str(tmp_path / "forward"),
        ]
    )

    out_path = tmp_path / "forward" / "2026-05-22_settlement.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["rows"][0]["actual_source"] == "manual"
    assert payload["summary"]["n_settled"] == 1


def test_forward_test_settle_cli_preflights_actuals_csv_before_writing(
    monkeypatch, tmp_path, capsys
) -> None:
    packet_path = tmp_path / "packet.json"
    actuals_path = tmp_path / "actuals.csv"
    out_dir = tmp_path / "forward"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")
    actuals_path.write_text(
        "city,target_date,actual_high_f,actual_source\n"
        "denver,2026-05-22,,manual\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.forward_test_settle_cli._fetch_observed_high",
        lambda station, target: (_ for _ in ()).throw(AssertionError("fetch unused")),
    )

    code = forward_test_settle_cli.main(
        [
            "--packet",
            str(packet_path),
            "--target-date",
            "2026-05-22",
            "--actuals-csv",
            str(actuals_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    output = capsys.readouterr().out
    assert code == 1
    assert "Actuals CSV: FAIL" in output
    assert "Settlement not written" in output
    assert not (out_dir / "2026-05-22_settlement.json").exists()
    assert not (out_dir / "history.csv").exists()


def test_build_settlement_payload_returns_error_for_city_fetch_failure(
    monkeypatch, tmp_path
) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    def fail_fetch(station, target):
        raise ValueError("no observed high")

    monkeypatch.setattr(
        "src.forward_test_settle_cli._fetch_observed_high",
        fail_fetch,
    )

    payload, code = forward_test_settle_cli.build_settlement_payload(
        packet_path=packet_path,
        target=date(2026, 5, 22),
    )

    assert code == 1
    assert payload["n_rows"] == 0
    assert payload["errors"] == [{"city": "denver", "error": "no observed high"}]

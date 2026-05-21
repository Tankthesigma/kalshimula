import json

from src import predict_batch_cli
from src.fetchers.openmeteo import ModelDailyHigh


def test_predict_batch_cli_writes_json(monkeypatch, tmp_path, capsys) -> None:
    selected_sources = tmp_path / "selected_sources.csv"
    selected_sources.write_text(
        "city,selected_source\n"
        "denver,gfs_ens\n"
        "boston,gfs_ens\n",
        encoding="utf-8",
    )
    bias_table = tmp_path / "bias_table.csv"
    bias_table.write_text(
        "city,source,n,mean_error_f,bias_correction_f\n"
        "denver,gfs_ens,10,-2.0,2.0\n"
        "boston,gfs_ens,10,1.0,-1.0\n",
        encoding="utf-8",
    )
    threshold_residuals = tmp_path / "threshold_residuals.csv"
    threshold_residuals.write_text(
        "city,source,residual_f\n"
        "denver,gfs_ens,-2\n"
        "denver,gfs_ens,0\n"
        "denver,gfs_ens,2\n"
        "boston,gfs_ens,-1\n"
        "boston,gfs_ens,1\n",
        encoding="utf-8",
    )

    def fake_fetch_all_parallel(station, target, *, use_historical):
        base = 70.0 if station.slug == "denver" else 40.0
        return [
            ModelDailyHigh(
                source="gfs_ens",
                target_date=target,
                members_f=[base, base + 2.0],
            )
        ]

    monkeypatch.setattr(
        "src.predict._fetch_all_parallel",
        fake_fetch_all_parallel,
    )

    code = predict_batch_cli.main(
        [
            "--cities",
            "denver,boston",
            "--date",
            "2025-01-01",
            "--selected-sources",
            str(selected_sources),
            "--bias-table",
            str(bias_table),
            "--threshold-residuals",
            str(threshold_residuals),
            "--threshold-offsets",
            "0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["n_predictions"] == 2
    assert payload["n_errors"] == 0
    assert payload["predictions"][0]["city"] == "denver"
    assert payload["predictions"][0]["threshold_probabilities"][0]["threshold_f"] == 73
    assert payload["predictions"][1]["calibration"]["corrected_point_f"] == 40.0


def test_predict_batch_cli_continues_after_city_error(monkeypatch, tmp_path, capsys) -> None:
    selected_sources = tmp_path / "selected_sources.csv"
    selected_sources.write_text(
        "city,selected_source\n"
        "denver,gfs_ens\n"
        "boston,gfs_ens\n",
        encoding="utf-8",
    )

    def fake_fetch_all_parallel(station, target, *, use_historical):
        if station.slug == "boston":
            return []
        return [
            ModelDailyHigh(
                source="gfs_ens",
                target_date=target,
                members_f=[70.0],
            )
        ]

    monkeypatch.setattr(
        "src.predict._fetch_all_parallel",
        fake_fetch_all_parallel,
    )

    code = predict_batch_cli.main(
        [
            "--cities",
            "denver,boston",
            "--date",
            "2025-01-01",
            "--selected-sources",
            str(selected_sources),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["n_predictions"] == 1
    assert payload["n_errors"] == 1
    assert payload["errors"] == [
        {"city": "boston", "error": "every Open-Meteo source returned empty"}
    ]


def test_build_batch_payload_can_write_file(monkeypatch, tmp_path) -> None:
    out_path = tmp_path / "predictions.json"

    monkeypatch.setattr(
        "src.predict._fetch_all_parallel",
        lambda station, target, *, use_historical: [
            ModelDailyHigh(source="gfs_ens", target_date=target, members_f=[70.0])
        ],
    )

    code = predict_batch_cli.main(
        [
            "--cities",
            "denver",
            "--date",
            "2025-01-01",
            "--out",
            str(out_path),
        ]
    )

    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["predictions"][0]["forecast"]["point_f"] == 70.0

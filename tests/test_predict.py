from datetime import date

import pandas as pd

from src import predict
from src.fetchers.openmeteo import ModelDailyHigh


def test_load_selected_source_matches_city_case_insensitively(tmp_path) -> None:
    selected_sources = tmp_path / "selected_sources.csv"
    selected_sources.write_text(
        "city,selected_source\nDenver,gfs_ens\nnyc,openmeteo_naive\n",
        encoding="utf-8",
    )

    assert predict._load_selected_source(selected_sources, "denver") == "gfs_ens"
    assert predict._load_selected_source(selected_sources, "NYC") == "openmeteo_naive"
    assert predict._load_selected_source(selected_sources, "chicago") is None


def test_members_for_selected_source_filters_individual_source() -> None:
    members = pd.DataFrame(
        {
            "source": ["gfs_ens", "gfs_ens", "ecmwf_ens"],
            "temp_f": [70.0, 72.0, 75.0],
        }
    )

    selected, applied = predict._members_for_selected_source(members, "gfs_ens")

    assert applied
    assert selected["source"].tolist() == ["gfs_ens", "gfs_ens"]


def test_members_for_selected_source_keeps_pool_for_openmeteo_naive() -> None:
    members = pd.DataFrame(
        {
            "source": ["gfs_ens", "ecmwf_ens"],
            "temp_f": [70.0, 75.0],
        }
    )

    selected, applied = predict._members_for_selected_source(
        members, "openmeteo_naive"
    )

    assert not applied
    assert selected.equals(members)


def test_apply_prediction_artifacts_corrects_point_and_interval(tmp_path) -> None:
    bias_table = tmp_path / "bias_table.csv"
    bias_table.write_text(
        "city,source,n,mean_error_f,bias_correction_f\n"
        "denver,gfs_ens,10,-2.0,2.0\n",
        encoding="utf-8",
    )
    interval_table = tmp_path / "interval_table.csv"
    interval_table.write_text(
        "city,source,n,lower_error_f,upper_error_f,alpha\n"
        "denver,gfs_ens,10,-1.0,3.0,0.2\n",
        encoding="utf-8",
    )

    row, warnings = predict._apply_prediction_artifacts(
        city="denver",
        source="gfs_ens",
        target=date(2025, 1, 1),
        point_f=71.0,
        bias_table_path=bias_table,
        interval_table_path=interval_table,
    )

    assert warnings == []
    assert row["corrected_point_f"] == 73.0
    assert row["interval_lower_f"] == 70.0
    assert row["interval_upper_f"] == 74.0


def test_apply_prediction_artifacts_warns_on_missing_source(tmp_path) -> None:
    bias_table = tmp_path / "bias_table.csv"
    bias_table.write_text(
        "city,source,n,mean_error_f,bias_correction_f\n"
        "denver,openmeteo_naive,10,-2.0,2.0\n",
        encoding="utf-8",
    )

    row, warnings = predict._apply_prediction_artifacts(
        city="denver",
        source="gfs_ens",
        target=date(2025, 1, 1),
        point_f=71.0,
        bias_table_path=bias_table,
    )

    assert row["point_f"] == 71.0
    assert "corrected_point_f" not in row
    assert warnings == ["no bias row for denver/gfs_ens; leaving point uncorrected"]


def test_resolve_model_artifacts_defaults_from_run_dir(tmp_path) -> None:
    run_dir = tmp_path / "run"
    recommended_sources = run_dir / "source_selection" / "recommended_sources.csv"
    selected_sources = run_dir / "source_selection" / "selected_sources.csv"
    bias_table = run_dir / "train_eval" / "bias_table.csv"
    recommended_sources.parent.mkdir(parents=True)
    bias_table.parent.mkdir(parents=True)
    recommended_sources.write_text("city,selected_source\n", encoding="utf-8")
    selected_sources.write_text("city,selected_source\n", encoding="utf-8")
    bias_table.write_text("city,source,bias_correction_f\n", encoding="utf-8")

    resolved = predict._resolve_model_artifacts(
        model_run_dir=run_dir,
        selected_sources=None,
        bias_table=None,
        interval_table=None,
    )

    assert resolved == (recommended_sources, bias_table, None)


def test_resolve_model_artifacts_falls_back_to_selected_sources(tmp_path) -> None:
    run_dir = tmp_path / "run"
    selected_sources = run_dir / "source_selection" / "selected_sources.csv"
    selected_sources.parent.mkdir(parents=True)
    selected_sources.write_text("city,selected_source\n", encoding="utf-8")

    resolved = predict._resolve_model_artifacts(
        model_run_dir=run_dir,
        selected_sources=None,
        bias_table=None,
        interval_table=None,
    )

    assert resolved == (selected_sources, None, None)


def test_resolve_model_artifacts_allows_explicit_overrides(tmp_path) -> None:
    run_dir = tmp_path / "run"
    explicit_selected = tmp_path / "selected.csv"
    explicit_bias = tmp_path / "bias.csv"
    explicit_interval = tmp_path / "interval.csv"

    resolved = predict._resolve_model_artifacts(
        model_run_dir=run_dir,
        selected_sources=explicit_selected,
        bias_table=explicit_bias,
        interval_table=explicit_interval,
    )

    assert resolved == (explicit_selected, explicit_bias, explicit_interval)


def test_predict_cli_uses_selected_source(monkeypatch, tmp_path, capsys) -> None:
    selected_sources = tmp_path / "selected_sources.csv"
    selected_sources.write_text(
        "city,selected_source\n"
        "denver,gfs_ens\n"
        "chicago,openmeteo_naive\n",
        encoding="utf-8",
    )

    def fake_fetch_all_parallel(station, target, *, use_historical):
        assert station.name == "Denver"
        assert target == date(2025, 1, 1)
        assert use_historical
        return [
            ModelDailyHigh(
                source="gfs_ens",
                target_date=target,
                members_f=[70.0, 72.0],
            ),
            ModelDailyHigh(
                source="ecmwf_ens",
                target_date=target,
                members_f=[80.0, 82.0],
            ),
        ]

    monkeypatch.setattr(predict, "_fetch_all_parallel", fake_fetch_all_parallel)

    exit_code = predict.main(
        [
            "--city",
            "denver",
            "--date",
            "2025-01-01",
            "--selected-sources",
            str(selected_sources),
        ]
    )
    output = capsys.readouterr()

    assert exit_code == 0
    assert "using selected source: gfs_ens" in output.err
    assert "Point estimate: 71.0" in output.out
    assert "Sources: gfs_ens(2)" in output.out
    assert "ecmwf_ens" not in output.out


def test_predict_cli_applies_model_artifacts(monkeypatch, tmp_path, capsys) -> None:
    selected_sources = tmp_path / "selected_sources.csv"
    selected_sources.write_text("city,selected_source\ndenver,gfs_ens\n", encoding="utf-8")
    bias_table = tmp_path / "bias_table.csv"
    bias_table.write_text(
        "city,source,n,mean_error_f,bias_correction_f\n"
        "denver,gfs_ens,10,-2.0,2.0\n",
        encoding="utf-8",
    )
    interval_table = tmp_path / "interval_table.csv"
    interval_table.write_text(
        "city,source,n,lower_error_f,upper_error_f,alpha\n"
        "denver,gfs_ens,10,-1.0,3.0,0.2\n",
        encoding="utf-8",
    )

    def fake_fetch_all_parallel(station, target, *, use_historical):
        return [
            ModelDailyHigh(
                source="gfs_ens",
                target_date=target,
                members_f=[70.0, 72.0],
            ),
            ModelDailyHigh(
                source="ecmwf_ens",
                target_date=target,
                members_f=[80.0, 82.0],
            ),
        ]

    monkeypatch.setattr(predict, "_fetch_all_parallel", fake_fetch_all_parallel)

    exit_code = predict.main(
        [
            "--city",
            "denver",
            "--date",
            "2025-01-01",
            "--selected-sources",
            str(selected_sources),
            "--bias-table",
            str(bias_table),
            "--interval-table",
            str(interval_table),
        ]
    )
    output = capsys.readouterr()

    assert exit_code == 0
    assert "Calibration: Model source: gfs_ens" in output.out
    assert "Corrected point: 73.0" in output.out
    assert "Empirical interval: [70.0" in output.out


def test_predict_cli_uses_model_run_dir(monkeypatch, tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    recommended_sources = run_dir / "source_selection" / "recommended_sources.csv"
    selected_sources = run_dir / "source_selection" / "selected_sources.csv"
    bias_table = run_dir / "train_eval" / "bias_table.csv"
    interval_table = run_dir / "train_eval" / "interval_table.csv"
    recommended_sources.parent.mkdir(parents=True)
    bias_table.parent.mkdir(parents=True)
    recommended_sources.write_text(
        "city,selected_source\ndenver,gfs_ens\n", encoding="utf-8"
    )
    selected_sources.write_text(
        "city,selected_source\ndenver,openmeteo_naive\n", encoding="utf-8"
    )
    bias_table.write_text(
        "city,source,n,mean_error_f,bias_correction_f\n"
        "denver,gfs_ens,10,-2.0,2.0\n",
        encoding="utf-8",
    )
    interval_table.write_text(
        "city,source,n,lower_error_f,upper_error_f,alpha\n"
        "denver,gfs_ens,10,-1.0,3.0,0.2\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        predict,
        "_fetch_all_parallel",
        lambda station, target, *, use_historical: [
            ModelDailyHigh(
                source="gfs_ens",
                target_date=target,
                members_f=[70.0, 72.0],
            ),
            ModelDailyHigh(
                source="ecmwf_ens",
                target_date=target,
                members_f=[80.0, 82.0],
            ),
        ],
    )

    exit_code = predict.main(
        [
            "--city",
            "denver",
            "--date",
            "2025-01-01",
            "--model-run-dir",
            str(run_dir),
        ]
    )
    output = capsys.readouterr()

    assert exit_code == 0
    assert "using selected source: gfs_ens" in output.err
    assert "Corrected point: 73.0" in output.out

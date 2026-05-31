import json
from datetime import date

import pandas as pd
import pytest

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


def test_load_selected_source_uses_global_recommended_policy_for_missing_city(tmp_path) -> None:
    selected_sources = tmp_path / "recommended_sources.csv"
    selected_sources.write_text(
        "city,selected_source,recommended_policy\n"
        "austin,gfs_ens,best_global_validation_source\n"
        "boston,gfs_ens,best_global_validation_source\n",
        encoding="utf-8",
    )

    assert predict._load_selected_source(selected_sources, "chicago") == "gfs_ens"


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


def test_multi_source_equal_blend_weights_sources_equally() -> None:
    members = pd.DataFrame(
        {
            "source": ["gfs_ens", "gfs_ens", "ecmwf_ens"],
            "temp_f": [60.0, 80.0, 100.0],
        }
    )

    forecast, metadata, warnings = predict._multi_source_forecast(
        members=members,
        mode="blend_equal",
        city="denver",
        target=date(2026, 5, 22),
    )

    assert warnings == []
    assert forecast.point_f == pytest.approx(85.0)
    assert forecast.per_source_counts == {"ecmwf_ens": 1, "gfs_ens": 2}
    assert metadata["source_weights"] == {"ecmwf_ens": 0.5, "gfs_ens": 0.5}
    assert metadata["artifact_source"] == "openmeteo_naive"


def test_multi_source_mae_blend_uses_recent_city_source_mae(tmp_path) -> None:
    members = pd.DataFrame(
        {
            "source": ["gfs_ens", "ecmwf_ens"],
            "temp_f": [70.0, 90.0],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "target_date": "2026-05-01",
                "point_f": 70.0,
                "actual_high_f": 71.0,
            },
            {
                "city": "denver",
                "source": "ecmwf_ens",
                "target_date": "2026-05-01",
                "point_f": 70.0,
                "actual_high_f": 75.0,
            },
        ]
    ).to_csv(run_dir / "rows.csv", index=False)

    forecast, metadata, warnings = predict._multi_source_forecast(
        members=members,
        mode="blend_mae_90d",
        city="denver",
        target=date(2026, 5, 22),
        model_run_dir=run_dir,
    )

    assert warnings == []
    assert metadata["recent_90d_mae_f"] == {"ecmwf_ens": 5.0, "gfs_ens": 1.0}
    assert metadata["source_weights"]["gfs_ens"] > metadata["source_weights"]["ecmwf_ens"]
    assert forecast.point_f == pytest.approx(73.3333333333)


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


def test_threshold_probability_rows_use_empirical_residuals() -> None:
    calibration = pd.Series(
        {
            "city": "denver",
            "source": "gfs_ens",
            "point_f": 70.0,
            "corrected_point_f": 72.0,
        }
    )
    residuals = pd.Series([-2.0, 0.0, 2.0, 4.0])

    rows = predict._threshold_probability_rows(
        calibration=calibration,
        residuals=residuals,
        offsets=(0, 2),
    )

    assert rows["threshold_f"].tolist() == [72, 74]
    assert rows["predicted_probability"].tolist() == [0.75, 0.5]


def test_threshold_probability_rows_apply_recalibration_table() -> None:
    calibration = pd.Series(
        {
            "city": "denver",
            "source": "gfs_ens",
            "point_f": 70.0,
            "corrected_point_f": 72.0,
        }
    )
    residuals = pd.Series([-2.0, 0.0, 2.0, 4.0])
    recalibration_table = pd.DataFrame(
        [
            {
                "city": "__global__",
                "source": "__global__",
                "bucket_start": 0.7,
                "bucket_end": 0.8,
                "recalibrated_probability": 0.65,
                "n": 40,
                "used": True,
            },
            {
                "city": "denver",
                "source": "gfs_ens",
                "bucket_start": 0.7,
                "bucket_end": 0.8,
                "recalibrated_probability": 0.6,
                "n": 25,
                "used": True,
            }
        ]
    )

    rows = predict._threshold_probability_rows(
        calibration=calibration,
        residuals=residuals,
        offsets=(0, 2),
        recalibration_table=recalibration_table,
    )

    assert rows["predicted_probability"].tolist() == [0.6, 0.5]
    assert rows["raw_predicted_probability"].tolist() == [0.75, 0.5]
    assert rows["recalibration_used"].tolist() == [True, False]
    assert rows["recalibration_scope"].tolist() == ["city_source", "none"]
    assert int(rows.iloc[0]["recalibration_n"]) == 25
    assert pd.isna(rows.iloc[1]["recalibration_n"])


def test_threshold_probability_rows_use_global_recalibration_fallback() -> None:
    calibration = pd.Series(
        {
            "city": "denver",
            "source": "gfs_ens",
            "point_f": 70.0,
            "corrected_point_f": 72.0,
        }
    )
    residuals = pd.Series([-2.0, 0.0, 2.0, 4.0])
    recalibration_table = pd.DataFrame(
        [
            {
                "city": "__global__",
                "source": "__global__",
                "bucket_start": 0.7,
                "bucket_end": 0.8,
                "recalibrated_probability": 0.65,
                "n": 40,
                "used": True,
            }
        ]
    )

    rows = predict._threshold_probability_rows(
        calibration=calibration,
        residuals=residuals,
        offsets=(0, 2),
        recalibration_table=recalibration_table,
    )

    assert rows["predicted_probability"].tolist() == [0.65, 0.5]
    assert rows["raw_predicted_probability"].tolist() == [0.75, 0.5]
    assert rows["recalibration_used"].tolist() == [True, False]
    assert rows["recalibration_scope"].tolist() == ["global", "none"]
    assert int(rows.iloc[0]["recalibration_n"]) == 40
    assert pd.isna(rows.iloc[1]["recalibration_n"])


def test_load_threshold_recalibration_table_keeps_global_fallback(tmp_path) -> None:
    path = tmp_path / "threshold_recalibration_table.csv"
    path.write_text(
        "city,source,bucket_start,bucket_end,recalibrated_probability,used\n"
        "denver,gfs_ens,0.7,0.8,0.6,True\n"
        "__global__,__global__,0.5,0.6,0.55,True\n"
        "boston,gfs_ens,0.7,0.8,0.7,True\n"
        "denver,gfs_ens,0.2,0.3,0.25,False\n",
        encoding="utf-8",
    )

    table = predict._load_threshold_recalibration_table(
        path,
        city="denver",
        source="gfs_ens",
    )

    assert table["city"].tolist() == ["denver", "__global__"]
    assert table["recalibrated_probability"].tolist() == [0.6, 0.55]


def test_resolve_model_artifacts_defaults_from_run_dir(tmp_path) -> None:
    run_dir = tmp_path / "run"
    recommended_sources = run_dir / "source_selection" / "recommended_sources.csv"
    selected_sources = run_dir / "source_selection" / "selected_sources.csv"
    policy_bias_table = run_dir / "model_policy" / "bias_table.csv"
    policy_interval_table = run_dir / "model_policy" / "interval_table.csv"
    threshold_residuals = run_dir / "probability_calibration" / "threshold_residuals.csv"
    threshold_recalibration = (
        run_dir / "probability_calibration" / "threshold_recalibration_table.csv"
    )
    train_eval_bias_table = run_dir / "train_eval" / "bias_table.csv"
    recommended_sources.parent.mkdir(parents=True)
    policy_bias_table.parent.mkdir(parents=True)
    threshold_residuals.parent.mkdir(parents=True)
    train_eval_bias_table.parent.mkdir(parents=True)
    recommended_sources.write_text("city,selected_source\n", encoding="utf-8")
    selected_sources.write_text("city,selected_source\n", encoding="utf-8")
    policy_bias_table.write_text("city,source,bias_correction_f\n", encoding="utf-8")
    policy_interval_table.write_text("city,source,lower_error_f,upper_error_f\n", encoding="utf-8")
    threshold_residuals.write_text("city,source,residual_f\n", encoding="utf-8")
    threshold_recalibration.write_text(
        "city,source,bucket_start,bucket_end,recalibrated_probability,used\n",
        encoding="utf-8",
    )
    train_eval_bias_table.write_text("city,source,bias_correction_f\n", encoding="utf-8")

    resolved = predict._resolve_model_artifacts(
        model_run_dir=run_dir,
        selected_sources=None,
        bias_table=None,
        interval_table=None,
        threshold_residuals=None,
        threshold_recalibration_table=None,
    )

    assert resolved == (
        recommended_sources,
        policy_bias_table,
        policy_interval_table,
        threshold_residuals,
        threshold_recalibration,
    )


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
        threshold_residuals=None,
        threshold_recalibration_table=None,
    )

    assert resolved == (selected_sources, None, None, None, None)


def test_resolve_model_artifacts_allows_explicit_overrides(tmp_path) -> None:
    run_dir = tmp_path / "run"
    explicit_selected = tmp_path / "selected.csv"
    explicit_bias = tmp_path / "bias.csv"
    explicit_interval = tmp_path / "interval.csv"
    explicit_threshold = tmp_path / "threshold_residuals.csv"
    explicit_recalibration = tmp_path / "threshold_recalibration_table.csv"

    resolved = predict._resolve_model_artifacts(
        model_run_dir=run_dir,
        selected_sources=explicit_selected,
        bias_table=explicit_bias,
        interval_table=explicit_interval,
        threshold_residuals=explicit_threshold,
        threshold_recalibration_table=explicit_recalibration,
    )

    assert resolved == (
        explicit_selected,
        explicit_bias,
        explicit_interval,
        explicit_threshold,
        explicit_recalibration,
    )


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


def test_predict_cli_prints_threshold_probabilities(monkeypatch, tmp_path, capsys) -> None:
    selected_sources = tmp_path / "selected_sources.csv"
    selected_sources.write_text("city,selected_source\ndenver,gfs_ens\n", encoding="utf-8")
    bias_table = tmp_path / "bias_table.csv"
    bias_table.write_text(
        "city,source,n,mean_error_f,bias_correction_f\n"
        "denver,gfs_ens,10,-2.0,2.0\n",
        encoding="utf-8",
    )
    threshold_residuals = tmp_path / "threshold_residuals.csv"
    threshold_residuals.write_text(
        "city,source,residual_f\n"
        "denver,gfs_ens,-2\n"
        "denver,gfs_ens,0\n"
        "denver,gfs_ens,2\n"
        "denver,gfs_ens,4\n",
        encoding="utf-8",
    )
    threshold_recalibration = tmp_path / "threshold_recalibration_table.csv"
    threshold_recalibration.write_text(
        "city,source,bucket_start,bucket_end,recalibrated_probability,used\n"
        "denver,gfs_ens,0.7,0.8,0.6,True\n"
        "denver,gfs_ens,0.5,0.6,0.4,True\n",
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
            )
        ],
    )

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
            "--threshold-residuals",
            str(threshold_residuals),
            "--threshold-recalibration-table",
            str(threshold_recalibration),
            "--threshold-offsets",
            "0,2",
        ]
    )
    output = capsys.readouterr()

    assert exit_code == 0
    assert "Threshold probabilities:" in output.out
    assert "P(high >= 73°F):  60.0% (raw 75.0%)" in output.out
    assert "P(high >= 75°F):  40.0% (raw 50.0%)" in output.out


def test_predict_cli_prints_json(monkeypatch, tmp_path, capsys) -> None:
    selected_sources = tmp_path / "selected_sources.csv"
    selected_sources.write_text("city,selected_source\ndenver,gfs_ens\n", encoding="utf-8")
    bias_table = tmp_path / "bias_table.csv"
    bias_table.write_text(
        "city,source,n,mean_error_f,bias_correction_f\n"
        "denver,gfs_ens,10,-2.0,2.0\n",
        encoding="utf-8",
    )
    threshold_residuals = tmp_path / "threshold_residuals.csv"
    threshold_residuals.write_text(
        "city,source,residual_f\n"
        "denver,gfs_ens,-2\n"
        "denver,gfs_ens,0\n"
        "denver,gfs_ens,2\n"
        "denver,gfs_ens,4\n",
        encoding="utf-8",
    )
    threshold_recalibration = tmp_path / "threshold_recalibration_table.csv"
    threshold_recalibration.write_text(
        "city,source,bucket_start,bucket_end,n,recalibrated_probability,used\n"
        "denver,gfs_ens,0.7,0.8,25,0.6,True\n",
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
            )
        ],
    )

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
            "--threshold-residuals",
            str(threshold_residuals),
            "--threshold-recalibration-table",
            str(threshold_recalibration),
            "--threshold-offsets",
            "0",
            "--json",
        ]
    )
    output = capsys.readouterr()
    payload = json.loads(output.out)

    assert exit_code == 0
    assert payload["schema_version"] == "1.0"
    assert payload["generated_at"].endswith("+00:00")
    assert payload["city"] == "denver"
    assert payload["selected_source"] == "gfs_ens"
    assert payload["artifact_paths"]["selected_sources"] == str(selected_sources)
    assert payload["artifact_paths"]["bias_table"] == str(bias_table)
    assert payload["artifact_paths"]["threshold_residuals"] == str(threshold_residuals)
    assert payload["artifact_paths"]["threshold_recalibration_table"] == str(
        threshold_recalibration
    )
    assert payload["forecast"]["point_f"] == 71.0
    assert payload["calibration"]["corrected_point_f"] == 73.0
    assert payload["threshold_probabilities"] == [
        {
            "offset_f": 0,
            "predicted_probability": 0.6,
            "raw_predicted_probability": 0.75,
            "recalibration_n": 25,
            "recalibration_scope": "city_source",
            "recalibration_used": True,
            "threshold_f": 73,
        }
    ]


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

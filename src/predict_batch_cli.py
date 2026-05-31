"""Batch JSON prediction CLI for dashboards and review scripts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src import model_gate_cli, predict
from src.config import get_station, load_stations
from src.fetchers.openmeteo import ModelDailyHigh, members_dataframe
from src.models.ensemble import naive_forecast_from_members


def _parse_cities(value: str) -> list[str]:
    cities = [city.strip() for city in value.split(",") if city.strip()]
    if not cities:
        raise argparse.ArgumentTypeError("at least one city is required")
    return cities


def _prediction_for_city(
    *,
    city: str,
    target,
    model_run_dir: Path | None,
    selected_sources_path: Path | None,
    bias_table_path: Path | None,
    interval_table_path: Path | None,
    threshold_residuals_path: Path | None,
    threshold_recalibration_table_path: Path | None,
    threshold_offsets: tuple[int, ...] | None,
    multi_source_mode: str,
    generated_at: datetime,
) -> dict:
    station = get_station(city)
    use_historical = target < date.today() - timedelta(days=2)
    selected_source = None
    if selected_sources_path:
        selected_source = predict._load_selected_source(selected_sources_path, city)
    print(
        f"Fetching {station.name} on {target}...",
        file=sys.stderr,
    )
    sources = _forecast_sources_for_city(
        station=station,
        target=target,
        use_historical=use_historical,
        selected_source=selected_source,
        multi_source_mode=multi_source_mode,
    )
    all_members = members_dataframe(sources)
    if all_members.empty:
        raise ValueError("every Open-Meteo source returned empty")
    members = all_members

    selected_applied = False
    if selected_source is not None:
        members, selected_applied = predict._members_for_selected_source(
            members, selected_source
        )

    forecast = naive_forecast_from_members(members)
    calibration = None
    if bias_table_path or interval_table_path:
        source = predict._prediction_source(
            selected_source, selected_applied=selected_applied
        )
        calibration, warnings = predict._apply_prediction_artifacts(
            city=city,
            source=source,
            target=target,
            point_f=forecast.point_f,
            bias_table_path=bias_table_path,
            interval_table_path=interval_table_path,
        )
        for warning in warnings:
            print(f"  ! {city}: {warning}", file=sys.stderr)

    threshold_probabilities = None
    if threshold_offsets is not None:
        source = predict._prediction_source(
            selected_source, selected_applied=selected_applied
        )
        if calibration is None:
            calibration = pd.Series(
                {
                    "city": city,
                    "source": source,
                    "target_date": target.isoformat(),
                    "point_f": forecast.point_f,
                }
            )
        if threshold_residuals_path is None:
            print(
                f"  ! {city}: threshold offsets requested but no residual artifact found",
                file=sys.stderr,
            )
        else:
            residuals = predict._load_threshold_residuals(
                threshold_residuals_path, city=city, source=source
            )
            recalibration_table = None
            if threshold_recalibration_table_path is not None:
                recalibration_table = predict._load_threshold_recalibration_table(
                    threshold_recalibration_table_path,
                    city=city,
                    source=source,
                )
            threshold_probabilities = predict._threshold_probability_rows(
                calibration=calibration,
                residuals=residuals,
                offsets=threshold_offsets,
                recalibration_table=recalibration_table,
            )

    payload = predict._json_payload(
        station,
        target,
        forecast,
        selected_source=selected_source,
        selected_applied=selected_applied,
        calibration=calibration,
        threshold_probabilities=threshold_probabilities,
        generated_at=generated_at,
        model_run_dir=model_run_dir,
        selected_sources_path=selected_sources_path,
        bias_table_path=bias_table_path,
        interval_table_path=interval_table_path,
        threshold_residuals_path=threshold_residuals_path,
        threshold_recalibration_table_path=threshold_recalibration_table_path,
    )
    if multi_source_mode != "single":
        payload["multi_source"] = predict._json_multi_source_prediction(
            city=city,
            target=target,
            members=all_members,
            mode=multi_source_mode,
            model_run_dir=model_run_dir,
            bias_table_path=bias_table_path,
            interval_table_path=interval_table_path,
            threshold_residuals_path=threshold_residuals_path,
            threshold_recalibration_table_path=threshold_recalibration_table_path,
            threshold_offsets=threshold_offsets,
        )
    return payload


def _forecast_sources_for_city(
    *,
    station,
    target,
    use_historical: bool,
    selected_source: str | None,
    multi_source_mode: str,
) -> list[ModelDailyHigh]:
    """Fetch the minimum source set needed for the requested prediction mode."""
    if (
        multi_source_mode == "single"
        and selected_source
        and selected_source != "openmeteo_naive"
    ):
        try:
            return [
                predict.fetch_source(
                    selected_source,
                    lat=station.lat,
                    lon=station.lon,
                    target=target,
                    use_historical=use_historical,
                )
            ]
        except Exception as error:  # noqa: BLE001
            print(
                f"  ! {station.slug}: selected source {selected_source} failed, falling back to full source sweep: {error}",
                file=sys.stderr,
            )
    return predict._fetch_all_parallel(
        station, target, use_historical=use_historical
    )


def _artifact_paths_payload(
    *,
    model_run_dir: Path | None,
    selected_sources_path: Path | None,
    bias_table_path: Path | None,
    interval_table_path: Path | None,
    threshold_residuals_path: Path | None,
    threshold_recalibration_table_path: Path | None,
) -> dict:
    return {
        "model_run_dir": str(model_run_dir) if model_run_dir is not None else None,
        "selected_sources": (
            str(selected_sources_path) if selected_sources_path is not None else None
        ),
        "bias_table": str(bias_table_path) if bias_table_path is not None else None,
        "interval_table": (
            str(interval_table_path) if interval_table_path is not None else None
        ),
        "threshold_residuals": (
            str(threshold_residuals_path) if threshold_residuals_path is not None else None
        ),
        "threshold_recalibration_table": (
            str(threshold_recalibration_table_path)
            if threshold_recalibration_table_path is not None
            else None
        ),
    }


def _gate_check_payload(check: model_gate_cli.GateCheck) -> dict:
    return {
        "name": check.name,
        "value": check.value,
        "threshold": check.threshold,
        "passed": check.passed,
        "detail": check.detail,
    }


def build_gate_payload(run_dir: Path) -> tuple[dict, bool]:
    """Build the default readiness gate payload for a model run."""
    checks = model_gate_cli.build_gate_checks(
        run_dir=run_dir,
        min_rows=model_gate_cli.DEFAULT_MIN_ROWS,
        min_cities=model_gate_cli.DEFAULT_MIN_CITIES,
        min_sources=model_gate_cli.DEFAULT_MIN_SOURCES,
        min_target_dates=model_gate_cli.DEFAULT_MIN_TARGET_DATES,
        max_test_mae=model_gate_cli.DEFAULT_MAX_TEST_MAE,
        min_interval_coverage=model_gate_cli.DEFAULT_MIN_INTERVAL_COVERAGE,
        max_interval_width=model_gate_cli.DEFAULT_MAX_INTERVAL_WIDTH,
        max_recalibrated_brier=model_gate_cli.DEFAULT_MAX_RECALIBRATED_BRIER,
        max_recalibrated_ece=model_gate_cli.DEFAULT_MAX_RECALIBRATED_ECE,
        min_brier_improvement=model_gate_cli.DEFAULT_MIN_BRIER_IMPROVEMENT,
        min_ece_improvement=model_gate_cli.DEFAULT_MIN_ECE_IMPROVEMENT,
        expected_source=model_gate_cli.DEFAULT_EXPECTED_SOURCE,
    )
    passed = all(check.passed for check in checks)
    return {
        "required": True,
        "passed": passed,
        "checks": [_gate_check_payload(check) for check in checks],
    }, passed


def _gate_failure_payload(
    *,
    target,
    created_at: datetime,
    model_run_dir: Path | None,
    selected_sources_path: Path | None,
    bias_table_path: Path | None,
    interval_table_path: Path | None,
    threshold_residuals_path: Path | None,
    threshold_recalibration_table_path: Path | None,
    model_gate: dict,
    error: str,
) -> dict:
    return {
        "schema_version": predict.PREDICTION_JSON_SCHEMA_VERSION,
        "generated_at": created_at.isoformat(),
        "target_date": target.isoformat(),
        "artifact_paths": _artifact_paths_payload(
            model_run_dir=model_run_dir,
            selected_sources_path=selected_sources_path,
            bias_table_path=bias_table_path,
            interval_table_path=interval_table_path,
            threshold_residuals_path=threshold_residuals_path,
            threshold_recalibration_table_path=threshold_recalibration_table_path,
        ),
        "model_gate": model_gate,
        "n_predictions": 0,
        "n_errors": 1,
        "predictions": [],
        "errors": [{"city": "__model_gate__", "error": error}],
    }


def build_batch_payload(
    *,
    cities: list[str],
    target,
    model_run_dir: Path | None,
    selected_sources_path: Path | None,
    bias_table_path: Path | None,
    interval_table_path: Path | None,
    threshold_residuals_path: Path | None,
    threshold_recalibration_table_path: Path | None,
    threshold_offsets: tuple[int, ...] | None,
    multi_source_mode: str = "single",
    model_gate: dict | None = None,
    generated_at: datetime | None = None,
) -> tuple[dict, int]:
    predictions = []
    errors = []
    created_at = generated_at or datetime.now(UTC)
    for city in cities:
        try:
            predictions.append(
                _prediction_for_city(
                    city=city,
                    target=target,
                    model_run_dir=model_run_dir,
                    selected_sources_path=selected_sources_path,
                    bias_table_path=bias_table_path,
                    interval_table_path=interval_table_path,
                    threshold_residuals_path=threshold_residuals_path,
                    threshold_recalibration_table_path=threshold_recalibration_table_path,
                    threshold_offsets=threshold_offsets,
                    multi_source_mode=multi_source_mode,
                    generated_at=created_at,
                )
            )
        except Exception as error:  # noqa: BLE001
            errors.append({"city": city, "error": str(error)})
            print(f"  ! {city}: {error}", file=sys.stderr)

    payload = {
        "schema_version": predict.PREDICTION_JSON_SCHEMA_VERSION,
        "generated_at": created_at.isoformat(),
        "target_date": target.isoformat(),
        "artifact_paths": _artifact_paths_payload(
            model_run_dir=model_run_dir,
            selected_sources_path=selected_sources_path,
            bias_table_path=bias_table_path,
            interval_table_path=interval_table_path,
            threshold_residuals_path=threshold_residuals_path,
            threshold_recalibration_table_path=threshold_recalibration_table_path,
        ),
        "model_gate": model_gate or {"required": False, "passed": None, "checks": []},
        "n_predictions": len(predictions),
        "n_errors": len(errors),
        "predictions": predictions,
        "errors": errors,
    }
    return payload, 0 if not errors else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="predict_batch",
        description="Emit batch calibrated predictions as JSON.",
    )
    parser.add_argument(
        "--cities",
        type=_parse_cities,
        default=",".join(load_stations().keys()),
        help="Comma-separated city slugs. Defaults to all configured cities.",
    )
    parser.add_argument("--date", required=True)
    parser.add_argument("--model-run-dir", type=Path)
    parser.add_argument("--selected-sources", type=Path)
    parser.add_argument("--bias-table", type=Path)
    parser.add_argument("--interval-table", type=Path)
    parser.add_argument("--threshold-residuals", type=Path)
    parser.add_argument("--threshold-recalibration-table", type=Path)
    parser.add_argument("--threshold-offsets")
    parser.add_argument(
        "--multi-source-mode",
        choices=predict.MULTI_SOURCE_MODES,
        default="single",
        help=(
            "Add an alternate multi_source forecast to each prediction. "
            "Default single preserves the existing selected-source output."
        ),
    )
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help="Fail without predictions unless --model-run-dir passes the readiness gate.",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    (
        selected_sources_path,
        bias_table_path,
        interval_table_path,
        threshold_residuals_path,
        threshold_recalibration_table_path,
    ) = predict._resolve_model_artifacts(
        model_run_dir=args.model_run_dir,
        selected_sources=args.selected_sources,
        bias_table=args.bias_table,
        interval_table=args.interval_table,
        threshold_residuals=args.threshold_residuals,
        threshold_recalibration_table=args.threshold_recalibration_table,
    )
    target = predict._parse_date(args.date)
    offsets = (
        predict._parse_int_list(args.threshold_offsets, name="threshold-offsets")
        if args.threshold_offsets
        else None
    )
    created_at = datetime.now(UTC)
    gate_payload = {"required": False, "passed": None, "checks": []}
    if args.require_gate:
        if args.model_run_dir is None:
            gate_payload = {"required": True, "passed": False, "checks": []}
            payload = _gate_failure_payload(
                target=target,
                created_at=created_at,
                model_run_dir=args.model_run_dir,
                selected_sources_path=selected_sources_path,
                bias_table_path=bias_table_path,
                interval_table_path=interval_table_path,
                threshold_residuals_path=threshold_residuals_path,
                threshold_recalibration_table_path=threshold_recalibration_table_path,
                model_gate=gate_payload,
                error="--require-gate requires --model-run-dir",
            )
            text = json.dumps(payload, indent=2, sort_keys=True)
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(text + "\n", encoding="utf-8")
            else:
                print(text)
            return 1

        try:
            gate_payload, gate_passed = build_gate_payload(args.model_run_dir)
        except ValueError as error:
            gate_payload = {"required": True, "passed": False, "checks": []}
            payload = _gate_failure_payload(
                target=target,
                created_at=created_at,
                model_run_dir=args.model_run_dir,
                selected_sources_path=selected_sources_path,
                bias_table_path=bias_table_path,
                interval_table_path=interval_table_path,
                threshold_residuals_path=threshold_residuals_path,
                threshold_recalibration_table_path=threshold_recalibration_table_path,
                model_gate=gate_payload,
                error=f"model readiness gate failed: {error}",
            )
            text = json.dumps(payload, indent=2, sort_keys=True)
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(text + "\n", encoding="utf-8")
            else:
                print(text)
            return 1

        if not gate_passed:
            payload = _gate_failure_payload(
                target=target,
                created_at=created_at,
                model_run_dir=args.model_run_dir,
                selected_sources_path=selected_sources_path,
                bias_table_path=bias_table_path,
                interval_table_path=interval_table_path,
                threshold_residuals_path=threshold_residuals_path,
                threshold_recalibration_table_path=threshold_recalibration_table_path,
                model_gate=gate_payload,
                error="model readiness gate did not pass",
            )
            text = json.dumps(payload, indent=2, sort_keys=True)
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(text + "\n", encoding="utf-8")
            else:
                print(text)
            return 1

    payload, exit_code = build_batch_payload(
        cities=args.cities,
        target=target,
        model_run_dir=args.model_run_dir,
        selected_sources_path=selected_sources_path,
        bias_table_path=bias_table_path,
        interval_table_path=interval_table_path,
        threshold_residuals_path=threshold_residuals_path,
        threshold_recalibration_table_path=threshold_recalibration_table_path,
        threshold_offsets=offsets,
        multi_source_mode=args.multi_source_mode,
        model_gate=gate_payload,
        generated_at=created_at,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

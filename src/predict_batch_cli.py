"""Batch JSON prediction CLI for dashboards and review scripts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src import predict
from src.config import get_station, load_stations
from src.fetchers.openmeteo import members_dataframe
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
    generated_at: datetime,
) -> dict:
    station = get_station(city)
    use_historical = target < date.today() - timedelta(days=2)
    print(
        f"Fetching {station.name} on {target}...",
        file=sys.stderr,
    )
    sources = predict._fetch_all_parallel(
        station, target, use_historical=use_historical
    )
    members = members_dataframe(sources)
    if members.empty:
        raise ValueError("every Open-Meteo source returned empty")

    selected_source = None
    selected_applied = False
    if selected_sources_path:
        selected_source = predict._load_selected_source(selected_sources_path, city)
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
    return payload


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
        "artifact_paths": {
            "model_run_dir": str(model_run_dir) if model_run_dir is not None else None,
            "selected_sources": (
                str(selected_sources_path) if selected_sources_path is not None else None
            ),
            "bias_table": str(bias_table_path) if bias_table_path is not None else None,
            "interval_table": (
                str(interval_table_path) if interval_table_path is not None else None
            ),
            "threshold_residuals": (
                str(threshold_residuals_path)
                if threshold_residuals_path is not None
                else None
            ),
            "threshold_recalibration_table": (
                str(threshold_recalibration_table_path)
                if threshold_recalibration_table_path is not None
                else None
            ),
        },
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

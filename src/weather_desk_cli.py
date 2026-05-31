"""Mainline weather-desk pipeline.

This command stitches together the weather-only nowcast stack:
features -> frozen prediction export -> weather-adjusted export -> report.
It does not fetch or use market data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from src.models.forward_packet import write_forward_packet_payload
from src.models.guidance import write_guidance_diagnostics
from src.models.heat_regime_correction import write_heat_regime_correction
from src.models.lone_outlier_correction import write_lone_outlier_correction
from src.models.lst_vulnerability import write_lst_vulnerability_scan
from src.models.nbm_candidate import write_nbm_candidate_predictions
from src.models.nbm_guidance import NOMADS_BLEND_BASE_URL, write_nbm_guidance_rows
from src.models.nbm_probability_calibration import write_nbm_calibrated_predictions
from src.models.nowcast_adjustment import write_nowcast_adjusted_predictions
from src.models.nowcast_features import write_nowcast_features
from src.models.nowcast_predictions import write_nowcast_predictions
from src.models.nowcast_report import write_nowcast_report
from src.models.nws_guidance import write_nws_guidance_rows
from src.models.station_rules import DEFAULT_STATION_RULES_PATH
from src.models.weather_analyst import write_weather_analyst_packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--as-of", required=True, help="UTC ISO timestamp")
    parser.add_argument("--decision-time-label", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--station-rules", type=Path, default=DEFAULT_STATION_RULES_PATH)
    parser.add_argument(
        "--cities",
        help="Comma-separated city slugs to build weather-desk rows for. Defaults to all station rules.",
    )
    parser.add_argument("--market-type", choices=["high"], default="high")
    parser.add_argument("--observations-csv", type=Path)
    parser.add_argument("--observation-store", type=Path)
    parser.add_argument("--update-observation-store", action="store_true")
    parser.add_argument("--fetch-live", action="store_true")
    parser.add_argument(
        "--include-nws-guidance",
        action="store_true",
        help="Fetch public NWS forecast guidance and compare it to the model packet.",
    )
    parser.add_argument(
        "--include-nbm-guidance",
        action="store_true",
        help="Fetch public NBM text guidance and emit a candidate NBM packet.",
    )
    parser.add_argument(
        "--nbm-base-url",
        default=NOMADS_BLEND_BASE_URL,
        help="NBM text product base URL. Use NOAA AWS S3 for historical archive probes.",
    )
    parser.add_argument(
        "--nbm-calibration-params",
        type=Path,
        help="Market-free NBM PMF calibration params JSON; emits predictions_nowcast_nbm_calibrated.",
    )
    parser.add_argument("--model-version", default="mainline-nowcast-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir
    git_commit = _git_commit()
    prediction_payload = json.loads(args.predictions_json.read_text(encoding="utf-8"))
    bias_table_path = _artifact_path(prediction_payload, "bias_table")
    interval_table_path = _artifact_path(prediction_payload, "interval_table")
    observations = (
        pd.read_csv(args.observations_csv)
        if args.observations_csv is not None
        else None
    )
    cities = _split_csv(args.cities)
    feature_result = write_nowcast_features(
        output_dir=out_dir / "nowcast_features",
        target_date=date.fromisoformat(args.target_date),
        as_of_ts=_parse_as_of(args.as_of),
        decision_time_label=args.decision_time_label,
        observations=observations,
        observation_store_path=args.observation_store,
        update_observation_store=args.update_observation_store,
        station_rules_path=args.station_rules,
        market_types=[args.market_type],
        cities=cities,
        fetch_live=args.fetch_live,
        git_commit=git_commit,
    )
    raw_result = write_nowcast_predictions(
        predictions_json_path=args.predictions_json,
        output_dir=out_dir / "predictions_nowcast_raw",
        decision_time_label=args.decision_time_label,
        nowcast_features_path=out_dir / "nowcast_features" / "nowcast_features.csv",
        station_rules_path=args.station_rules,
        as_of_ts_utc=args.as_of,
        market_type=args.market_type,
        model_version=args.model_version,
        git_commit=git_commit,
    )
    raw_packet_v2 = _write_forward_packet_v2(
        out_dir / "predictions_nowcast_raw" / "predictions_nowcast.csv",
        station_rules_path=args.station_rules,
        git_commit=git_commit,
    )
    vulnerability_result = write_lst_vulnerability_scan(
        observations_path=out_dir / "nowcast_features" / "asos_observations.csv",
        output_dir=out_dir / "lst_vulnerability",
        start_date=date.fromisoformat(args.target_date),
        end_date=date.fromisoformat(args.target_date),
        station_rules_path=args.station_rules,
        cities=cities,
        market_types=[args.market_type],
        git_commit=git_commit,
    )
    heat_result = write_heat_regime_correction(
        predictions_path=out_dir / "predictions_nowcast_raw" / "predictions_nowcast.csv",
        output_dir=out_dir / "predictions_nowcast_heat_corrected",
        git_commit=git_commit,
    )
    heat_packet_v2 = _write_forward_packet_v2(
        out_dir / "predictions_nowcast_heat_corrected" / "predictions_nowcast.csv",
        station_rules_path=args.station_rules,
        git_commit=git_commit,
    )
    adjusted_result = write_nowcast_adjusted_predictions(
        predictions_path=out_dir / "predictions_nowcast_raw" / "predictions_nowcast.csv",
        features_path=out_dir / "nowcast_features" / "nowcast_features.csv",
        output_dir=out_dir / "predictions_nowcast_adjusted",
        git_commit=git_commit,
    )
    adjusted_packet_v2 = _write_forward_packet_v2(
        out_dir / "predictions_nowcast_adjusted" / "predictions_nowcast.csv",
        station_rules_path=args.station_rules,
        git_commit=git_commit,
    )
    guidance_rows = pd.DataFrame()
    guidance_latest = pd.DataFrame()
    guidance_comparison = pd.DataFrame()
    lone_outlier_corrections = pd.DataFrame()
    lone_outlier_packet_v2_count = 0
    nbm_guidance_rows = pd.DataFrame()
    nbm_latest = pd.DataFrame()
    nbm_result = None
    nbm_packet_v2_count = 0
    nbm_calibrated_rows = pd.DataFrame()
    nbm_calibrated_packet_v2_count = 0
    nbm_guidance_error: str | None = None
    guarded_rows = pd.DataFrame()
    guarded_packet_v2_count = 0
    guarded_fallbacks = pd.DataFrame()
    if args.include_nws_guidance:
        guidance_path = out_dir / "guidance" / "nws_guidance_rows.csv"
        guidance_path.parent.mkdir(parents=True, exist_ok=True)
        guidance_rows = write_nws_guidance_rows(
            output_path=guidance_path,
            target=date.fromisoformat(args.target_date),
            cities=cities,
            market_types=[args.market_type],
            fetched_at=_parse_as_of(args.as_of),
        )
        guidance_result = write_guidance_diagnostics(
            input_path=guidance_path,
            output_dir=out_dir / "guidance_diagnostics",
            as_of_ts=args.as_of,
            target_date=args.target_date,
            git_commit=git_commit,
        )
        guidance_latest = guidance_result.latest
        guarded_rows, guarded_fallbacks = _write_nws_guarded_predictions(
            raw_predictions_path=out_dir / "predictions_nowcast_raw" / "predictions_nowcast.csv",
            adjusted_predictions_path=(
                out_dir / "predictions_nowcast_adjusted" / "predictions_nowcast.csv"
            ),
            guidance_latest=guidance_latest,
            output_dir=out_dir / "predictions_nowcast_nws_guarded",
        )
        guarded_packet_v2_count = len(
            _write_forward_packet_v2(
                out_dir / "predictions_nowcast_nws_guarded" / "predictions_nowcast.csv",
                station_rules_path=args.station_rules,
                git_commit=git_commit,
            ).packets
        )
        report_result = write_nowcast_report(
            predictions_path=(
                out_dir / "predictions_nowcast_nws_guarded" / "predictions_nowcast.csv"
            ),
            output_dir=out_dir / "nowcast_report",
            git_commit=git_commit,
        )
        guidance_comparison = _guidance_comparison(
            report_result.summary,
            guidance_latest,
        )
        guidance_comparison.to_csv(
            out_dir / "guidance" / "model_vs_nws_guidance.csv",
            index=False,
        )
        (out_dir / "guidance" / "model_vs_nws_guidance.md").write_text(
            _render_guidance_comparison(guidance_comparison),
            encoding="utf-8",
        )
        lone_outlier_result = write_lone_outlier_correction(
            predictions_path=out_dir / "predictions_nowcast_raw" / "predictions_nowcast.csv",
            prediction_json_path=args.predictions_json,
            guidance_path=guidance_path,
            output_dir=out_dir / "predictions_nowcast_lone_outlier",
            git_commit=git_commit,
        )
        lone_outlier_corrections = lone_outlier_result.corrections
        lone_outlier_packet_v2_count = len(
            _write_forward_packet_v2(
                out_dir / "predictions_nowcast_lone_outlier" / "predictions_nowcast.csv",
                station_rules_path=args.station_rules,
                git_commit=git_commit,
            ).packets
        )
    if args.include_nbm_guidance:
        nbm_guidance_path = out_dir / "guidance" / "nbm_guidance_rows.csv"
        nbm_guidance_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            nbm_guidance_rows = write_nbm_guidance_rows(
                output_path=nbm_guidance_path,
                target=date.fromisoformat(args.target_date),
                as_of_ts=_parse_as_of(args.as_of),
                station_rules_path=args.station_rules,
                cities=cities,
                market_types=[args.market_type],
                base_url=args.nbm_base_url,
            )
            nbm_guidance_result = write_guidance_diagnostics(
                input_path=nbm_guidance_path,
                output_dir=out_dir / "nbm_guidance_diagnostics",
                as_of_ts=args.as_of,
                target_date=args.target_date,
                git_commit=git_commit,
            )
            nbm_latest = nbm_guidance_result.latest
            nbm_result = write_nbm_candidate_predictions(
                raw_predictions_path=out_dir / "predictions_nowcast_raw" / "predictions_nowcast.csv",
                guidance_path=nbm_guidance_path,
                output_dir=out_dir / "predictions_nowcast_nbm",
                as_of_ts=args.as_of,
                git_commit=git_commit,
            )
            nbm_packet_v2_count = len(
                _write_forward_packet_v2(
                    out_dir / "predictions_nowcast_nbm" / "predictions_nowcast.csv",
                    station_rules_path=args.station_rules,
                    git_commit=git_commit,
                ).packets
            )
            if args.nbm_calibration_params is not None:
                nbm_calibrated_rows = write_nbm_calibrated_predictions(
                    input_predictions_path=(
                        out_dir / "predictions_nowcast_nbm" / "predictions_nowcast.csv"
                    ),
                    output_dir=out_dir / "predictions_nowcast_nbm_calibrated",
                    calibration_params_path=args.nbm_calibration_params,
                    git_commit=git_commit,
                )
                nbm_calibrated_packet_v2_count = len(
                    _write_forward_packet_v2(
                        out_dir
                        / "predictions_nowcast_nbm_calibrated"
                        / "predictions_nowcast.csv",
                        station_rules_path=args.station_rules,
                        git_commit=git_commit,
                    ).packets
                )
        except FileNotFoundError as exc:
            nbm_guidance_error = str(exc)
            print(f"NBM guidance unavailable; skipping NBM candidate mode: {exc}")
    report_result = write_nowcast_report(
        predictions_path=(
            out_dir / "predictions_nowcast_adjusted" / "predictions_nowcast.csv"
        ),
        output_dir=out_dir / "nowcast_report",
        git_commit=git_commit,
    )
    analyst_result = write_weather_analyst_packet(
        nowcast_summary_path=out_dir / "nowcast_report" / "nowcast_report_summary.csv",
        guidance_comparison_path=(
            out_dir / "guidance" / "model_vs_nws_guidance.csv"
            if args.include_nws_guidance
            else None
        ),
        bias_table_path=bias_table_path,
        interval_table_path=interval_table_path,
        output_dir=out_dir / "weather_analyst",
        git_commit=git_commit,
    )
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "target_date": args.target_date,
        "cities": cities,
        "market_type": args.market_type,
        "as_of_ts_utc": _parse_as_of(args.as_of).isoformat(),
        "decision_time_label": args.decision_time_label,
        "nbm_base_url": args.nbm_base_url if args.include_nbm_guidance else None,
        "artifacts": {
            "nowcast_features": "nowcast_features/nowcast_features.csv",
            "predictions_nowcast_raw": "predictions_nowcast_raw/predictions_nowcast.csv",
            "forward_packet_v2_raw": "predictions_nowcast_raw/forward_packet_v2.json",
            "predictions_nowcast_adjusted": (
                "predictions_nowcast_adjusted/predictions_nowcast.csv"
            ),
            "forward_packet_v2_adjusted": (
                "predictions_nowcast_adjusted/forward_packet_v2.json"
            ),
            "predictions_nowcast_heat_corrected": (
                "predictions_nowcast_heat_corrected/predictions_nowcast.csv"
            ),
            "forward_packet_v2_heat_corrected": (
                "predictions_nowcast_heat_corrected/forward_packet_v2.json"
            ),
            "heat_corrections": "predictions_nowcast_heat_corrected/heat_corrections.csv",
            "nowcast_report": "nowcast_report/nowcast_report.md",
            "weather_analyst_packet": "weather_analyst/weather_analyst_packet.md",
            "lst_vulnerability": (
                "lst_vulnerability/settlement_vulnerability_days.csv"
            ),
            "lst_vulnerability_report": (
                "lst_vulnerability/settlement_vulnerability_days.md"
            ),
            **(
                {
                    "nws_guidance": "guidance/nws_guidance_rows.csv",
                    "nws_guidance_diagnostics": "guidance_diagnostics/guidance_report.md",
                    "model_vs_nws_guidance": "guidance/model_vs_nws_guidance.csv",
                    "model_vs_nws_guidance_report": "guidance/model_vs_nws_guidance.md",
                    "predictions_nowcast_lone_outlier": (
                        "predictions_nowcast_lone_outlier/predictions_nowcast.csv"
                    ),
                    "forward_packet_v2_lone_outlier": (
                        "predictions_nowcast_lone_outlier/forward_packet_v2.json"
                    ),
                    "lone_outlier_corrections": (
                        "predictions_nowcast_lone_outlier/lone_outlier_corrections.csv"
                    ),
                    "predictions_nowcast_nws_guarded": (
                        "predictions_nowcast_nws_guarded/predictions_nowcast.csv"
                    ),
                    "forward_packet_v2_nws_guarded": (
                        "predictions_nowcast_nws_guarded/forward_packet_v2.json"
                    ),
                    "nws_guarded_fallbacks": (
                        "predictions_nowcast_nws_guarded/nws_guarded_fallbacks.csv"
                    ),
                }
                if args.include_nws_guidance
                else {}
            ),
            **(
                {
                    "nbm_guidance": "guidance/nbm_guidance_rows.csv",
                    "nbm_guidance_diagnostics": "nbm_guidance_diagnostics/guidance_report.md",
                    "predictions_nowcast_nbm": (
                        "predictions_nowcast_nbm/predictions_nowcast.csv"
                    ),
                    "forward_packet_v2_nbm": (
                        "predictions_nowcast_nbm/forward_packet_v2.json"
                    ),
                    **(
                        {
                            "predictions_nowcast_nbm_calibrated": (
                                "predictions_nowcast_nbm_calibrated/predictions_nowcast.csv"
                            ),
                            "forward_packet_v2_nbm_calibrated": (
                                "predictions_nowcast_nbm_calibrated/forward_packet_v2.json"
                            ),
                        }
                        if args.nbm_calibration_params is not None
                        else {}
                    ),
                }
                if args.include_nbm_guidance
                else {}
            ),
        },
        "row_counts": {
            "observations": int(len(feature_result.observations)),
            "features": int(len(feature_result.features)),
            "raw_prediction_rows": int(len(raw_result.predictions)),
            "raw_forward_packet_v2": int(len(raw_packet_v2.packets)),
            "lst_vulnerability_rows": int(len(vulnerability_result.rows)),
            "adjusted_prediction_rows": int(len(adjusted_result.predictions)),
            "adjusted_forward_packet_v2": int(len(adjusted_packet_v2.packets)),
            "heat_corrected_prediction_rows": int(len(heat_result.predictions)),
            "heat_corrected_forward_packet_v2": int(len(heat_packet_v2.packets)),
            "heat_corrections": int(len(heat_result.corrections)),
            "report_rows": int(len(report_result.summary)),
            "nws_guidance_rows": int(len(guidance_rows)),
            "nws_latest_rows": int(len(guidance_latest)),
            "model_vs_nws_guidance_rows": int(len(guidance_comparison)),
            "lone_outlier_corrections": int(len(lone_outlier_corrections)),
            "lone_outlier_forward_packet_v2": int(lone_outlier_packet_v2_count),
            "nws_guarded_prediction_rows": int(len(guarded_rows)),
            "nws_guarded_forward_packet_v2": int(guarded_packet_v2_count),
            "nws_guarded_fallback_rows": int(len(guarded_fallbacks)),
            "nbm_guidance_rows": int(len(nbm_guidance_rows)),
            "nbm_latest_rows": int(len(nbm_latest)),
            "nbm_prediction_rows": int(len(nbm_result.predictions)) if nbm_result else 0,
            "nbm_forward_packet_v2": int(nbm_packet_v2_count),
            "nbm_calibrated_prediction_rows": int(len(nbm_calibrated_rows)),
            "nbm_calibrated_forward_packet_v2": int(nbm_calibrated_packet_v2_count),
            "weather_analyst_rows": int(len(analyst_result.rows)),
        },
        "notes": [
            "Mainline weather-only pipeline. No market prices, order books, private "
            "PnL labels, or trade instructions.",
            "Raw and adjusted nowcast predictions are separate model modes; adjusted "
            "is a weather-aware candidate, not a promoted default.",
            "NWS-guarded predictions fall back to raw only when the adjustment layer "
            "clearly worsens same-day agreement with public NWS guidance.",
            "Lone-outlier correction is a candidate packet only; it is not a promoted default.",
            "Heat-regime correction is a candidate packet only; it is not a promoted default.",
            "NBM packet is a candidate mode only; it is not a promoted default.",
            "NBM calibrated packet is a candidate mode only; it is not a promoted default.",
            "forward_packet_v2 JSON artifacts are market-free weather packets joined "
            "privately on city/station/date/type/as_of.",
            "Bobby/private audit may consume forward_packet_v2 artifacts for "
            "market-side validation before any operational promotion.",
            *(
                [f"NBM guidance unavailable for this run; candidate NBM outputs were skipped: {nbm_guidance_error}"]
                if nbm_guidance_error
                else []
            ),
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weather_desk_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote weather desk packet to {out_dir}: "
        f"{len(adjusted_result.predictions)} adjusted prediction rows, "
        f"{len(report_result.summary)} report rows"
    )
    return 0


def _write_forward_packet_v2(
    predictions_path: Path,
    *,
    station_rules_path: Path,
    git_commit: str | None,
):
    return write_forward_packet_payload(
        predictions_path=predictions_path,
        output_path=predictions_path.parent / "forward_packet_v2.json",
        station_rules_path=station_rules_path,
        git_commit=git_commit,
    )


def _parse_as_of(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _guidance_comparison(summary: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "city",
        "platform",
        "market_type",
        "station_id",
        "target_date",
        "decision_time_label",
        "model_point_f",
        "nws_guidance_point_f",
        "model_minus_nws_f",
        "abs_model_minus_nws_f",
        "model_vs_nws_direction",
        "guidance_agreement",
        "model_q10_f",
        "model_q90_f",
        "nws_available_ts_utc",
        "nws_issue_ts_utc",
        "priority",
    ]
    if summary.empty or latest.empty:
        return pd.DataFrame(columns=columns)
    keys = ["city", "market_type", "station_id", "target_date"]
    merged = summary.merge(
        latest,
        on=keys,
        how="inner",
        suffixes=("_model", "_nws"),
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    output = pd.DataFrame(
        {
            "city": merged["city"],
            "platform": merged["platform"],
            "market_type": merged["market_type"],
            "station_id": merged["station_id"],
            "target_date": merged["target_date"],
            "decision_time_label": merged["decision_time_label"],
            "model_point_f": pd.to_numeric(merged["point_f"], errors="coerce"),
            "nws_guidance_point_f": pd.to_numeric(
                merged["guidance_point_f"],
                errors="coerce",
            ),
            "model_q10_f": pd.to_numeric(merged["q10_f"], errors="coerce"),
            "model_q90_f": pd.to_numeric(merged["q90_f"], errors="coerce"),
            "nws_available_ts_utc": merged["available_ts_utc"],
            "nws_issue_ts_utc": merged["issue_ts_utc"],
            "priority": merged["priority"],
        }
    )
    output["model_minus_nws_f"] = (
        output["model_point_f"] - output["nws_guidance_point_f"]
    )
    output["abs_model_minus_nws_f"] = output["model_minus_nws_f"].abs()
    output["model_vs_nws_direction"] = output["model_minus_nws_f"].map(_guidance_direction)
    output["guidance_agreement"] = output["abs_model_minus_nws_f"].map(_guidance_agreement)
    return output.loc[:, columns].sort_values(["city", "market_type"]).reset_index(drop=True)


def _guidance_direction(delta: object) -> str:
    if pd.isna(delta):
        return "unknown"
    value = float(delta)
    if value > 0:
        return "model_hotter"
    if value < 0:
        return "model_colder"
    return "aligned"


def _guidance_agreement(abs_delta: object) -> str:
    if pd.isna(abs_delta):
        return "unknown"
    value = float(abs_delta)
    if value >= 3.0:
        return "divergent"
    if value > 2.0:
        return "watch"
    return "aligned"


def _write_nws_guarded_predictions(
    *,
    raw_predictions_path: Path,
    adjusted_predictions_path: Path,
    guidance_latest: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_predictions = pd.read_csv(raw_predictions_path)
    adjusted_predictions = pd.read_csv(adjusted_predictions_path)
    raw_summary = _summarize_prediction_rows(raw_predictions)
    adjusted_summary = _summarize_prediction_rows(adjusted_predictions)
    fallback_rows = _nws_guarded_fallback_rows(
        raw_summary=raw_summary,
        adjusted_summary=adjusted_summary,
        guidance_latest=guidance_latest,
    )
    use_raw_cities = set(fallback_rows["city"].tolist())
    guarded_predictions = adjusted_predictions.copy()
    if use_raw_cities:
        guarded_predictions = pd.concat(
            [
                adjusted_predictions.loc[
                    ~adjusted_predictions["city"].isin(use_raw_cities)
                ],
                raw_predictions.loc[raw_predictions["city"].isin(use_raw_cities)],
            ],
            ignore_index=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    guarded_predictions.to_csv(output_dir / "predictions_nowcast.csv", index=False)
    fallback_rows.to_csv(output_dir / "nws_guarded_fallbacks.csv", index=False)
    return guarded_predictions, fallback_rows


def _summarize_prediction_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    summary_columns = [
        "city",
        "platform",
        "market_type",
        "station_id",
        "target_date",
        "decision_time_label",
        "source_policy",
        "point_f",
        "q10_f",
        "q90_f",
        "top_bin_label",
        "top_bin_probability",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=summary_columns)

    group_cols = [
        "city",
        "platform",
        "market_type",
        "station_id",
        "target_date",
        "decision_time_label",
    ]
    ordered = predictions.sort_values(
        ["city", "calibrated_probability", "model_probability", "bin_lower_f"],
        ascending=[True, False, False, True],
    )
    rows: list[dict[str, object]] = []
    for keys, group in ordered.groupby(group_cols, sort=True, dropna=False):
        top = group.iloc[0]
        rows.append(
            {
                "city": keys[0],
                "platform": keys[1],
                "market_type": keys[2],
                "station_id": keys[3],
                "target_date": keys[4],
                "decision_time_label": keys[5],
                "source_policy": top["source_policy"],
                "point_f": float(top["point_f"]),
                "q10_f": float(top["q10_f"]),
                "q90_f": float(top["q90_f"]),
                "top_bin_label": top["bin_label"],
                "top_bin_probability": float(top["calibrated_probability"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=summary_columns)
    return pd.DataFrame(rows, columns=summary_columns)


def _nws_guarded_fallback_rows(
    *,
    raw_summary: pd.DataFrame,
    adjusted_summary: pd.DataFrame,
    guidance_latest: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["city", "market_type", "station_id", "target_date"]
    latest = guidance_latest.loc[:, keys + ["guidance_point_f"]].copy()
    merged = adjusted_summary.merge(
        raw_summary,
        on=[
            "city",
            "platform",
            "market_type",
            "station_id",
            "target_date",
            "decision_time_label",
        ],
        how="inner",
        suffixes=("_adjusted", "_raw"),
    ).merge(latest, on=keys, how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "city",
                "market_type",
                "station_id",
                "target_date",
                "adjusted_point_f",
                "raw_point_f",
                "nws_guidance_point_f",
                "adjusted_abs_model_minus_nws_f",
                "raw_abs_model_minus_nws_f",
                "fallback_reason",
            ]
        )
    merged["adjusted_abs_model_minus_nws_f"] = (
        pd.to_numeric(merged["point_f_adjusted"], errors="coerce")
        - pd.to_numeric(merged["guidance_point_f"], errors="coerce")
    ).abs()
    merged["raw_abs_model_minus_nws_f"] = (
        pd.to_numeric(merged["point_f_raw"], errors="coerce")
        - pd.to_numeric(merged["guidance_point_f"], errors="coerce")
    ).abs()
    fallback = merged.loc[
        (
            pd.to_numeric(merged["point_f_adjusted"], errors="coerce").notna()
            & pd.to_numeric(merged["point_f_raw"], errors="coerce").notna()
            & pd.to_numeric(merged["guidance_point_f"], errors="coerce").notna()
            & (merged["source_policy_adjusted"] == merged["source_policy_raw"])
            & (
                merged["adjusted_abs_model_minus_nws_f"]
                > merged["raw_abs_model_minus_nws_f"] + 1.5
            )
        )
    ].copy()
    if fallback.empty:
        return pd.DataFrame(
            columns=[
                "city",
                "market_type",
                "station_id",
                "target_date",
                "adjusted_point_f",
                "raw_point_f",
                "nws_guidance_point_f",
                "adjusted_abs_model_minus_nws_f",
                "raw_abs_model_minus_nws_f",
                "fallback_reason",
            ]
        )
    fallback["fallback_reason"] = (
        "adjusted_degrades_nws_agreement_vs_raw_by_gt_1.5f"
    )
    return fallback.loc[
        :,
        [
            "city",
            "market_type",
            "station_id",
            "target_date",
            "point_f_adjusted",
            "point_f_raw",
            "guidance_point_f",
            "adjusted_abs_model_minus_nws_f",
            "raw_abs_model_minus_nws_f",
            "fallback_reason",
        ],
    ].rename(
        columns={
            "point_f_adjusted": "adjusted_point_f",
            "point_f_raw": "raw_point_f",
            "guidance_point_f": "nws_guidance_point_f",
        }
    ).sort_values(["city", "market_type"]).reset_index(drop=True)


def _render_guidance_comparison(comparison: pd.DataFrame) -> str:
    lines = [
        "# Model vs NWS Guidance",
        "",
        "Weather-only guidance comparison. No market prices, order books, private PnL labels, or trade instructions.",
        "",
    ]
    if comparison.empty:
        return "\n".join([*lines, "No comparable rows.", ""])
    lines.extend(
        [
            "| agreement | city | market | model | NWS | delta | direction | priority |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    ordered = comparison.sort_values(
        ["abs_model_minus_nws_f", "city"],
        ascending=[False, True],
    )
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.guidance_agreement} | {row.city} | {row.market_type} | "
            f"{row.model_point_f:.1f} | {row.nws_guidance_point_f:.1f} | "
            f"{row.model_minus_nws_f:+.1f} | {row.model_vs_nws_direction} | "
            f"{row.priority} |"
        )
    lines.extend(
        [
            "",
            "Agreement bands:",
            "- `aligned`: model and NWS are within 2F.",
            "- `watch`: model and NWS differ by more than 2F but less than 3F.",
            "- `divergent`: model and NWS differ by 3F or more.",
            "",
            (
                "Use this as a weather-desk sanity check only. Bobby/private audit "
                "decides whether any divergence is market-relevant."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _artifact_path(payload: dict[str, object], key: str) -> Path | None:
    artifact_paths = payload.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        return None
    value = artifact_paths.get(key)
    if not value:
        return None
    return Path(str(value))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

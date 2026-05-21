"""CLI for gating model artifacts against research-readiness thresholds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class GateCheck:
    name: str
    value: float | str
    threshold: float | str
    passed: bool
    detail: str


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"missing artifact: {path}")
    return pd.read_csv(path)


def _recommended_row(table: pd.DataFrame, *, split: str | None = None) -> pd.Series:
    if "recommended" not in table.columns:
        raise ValueError("artifact missing recommended column")
    rows = table[table["recommended"].astype(str).str.lower() == "true"]
    if split is not None and "split" in rows.columns:
        rows = rows[rows["split"].astype(str) == split]
    if rows.empty:
        raise ValueError("artifact has no recommended row")
    return rows.iloc[0]


def _policy_row(table: pd.DataFrame, policy: str) -> pd.Series:
    if "policy" not in table.columns:
        raise ValueError("artifact missing policy column")
    rows = table[table["policy"].astype(str) == policy]
    if rows.empty:
        raise ValueError(f"artifact has no policy row: {policy}")
    return rows.iloc[0]


def build_gate_checks(
    *,
    run_dir: Path,
    max_test_mae: float,
    min_interval_coverage: float,
    max_interval_width: float,
    max_recalibrated_brier: float,
    max_recalibrated_ece: float,
    min_brier_improvement: float,
    min_ece_improvement: float,
    expected_source: str | None = "gfs_ens",
) -> list[GateCheck]:
    """Build research-readiness checks for selected final model artifacts."""
    source_policy = _read_csv(run_dir / "source_selection" / "recommended_sources.csv")
    bias_policy = _read_csv(run_dir / "model_policy" / "bias_policy_comparison.csv")
    interval_policy = _read_csv(run_dir / "model_policy" / "interval_policy_comparison.csv")
    recalibration = _read_csv(
        run_dir / "probability_calibration" / "threshold_recalibration_comparison.csv"
    )

    checks: list[GateCheck] = []
    if expected_source is not None:
        if "selected_source" not in source_policy.columns:
            raise ValueError("recommended_sources.csv missing selected_source column")
        sources = sorted(str(value) for value in source_policy["selected_source"].dropna().unique())
        checks.append(
            GateCheck(
                name="source_policy",
                value=",".join(sources),
                threshold=expected_source,
                passed=sources == [expected_source],
                detail="all recommended source rows must match expected source",
            )
        )

    bias = _recommended_row(bias_policy)
    mae = float(bias["test_mae_corrected"])
    checks.append(
        GateCheck(
            name="test_mae_corrected",
            value=mae,
            threshold=max_test_mae,
            passed=mae <= max_test_mae,
            detail="recommended bias policy held-out MAE",
        )
    )

    interval = _recommended_row(interval_policy, split="test")
    coverage = float(interval["interval_coverage_raw"])
    width = float(interval["interval_width_raw"])
    checks.extend(
        [
            GateCheck(
                name="interval_coverage",
                value=coverage,
                threshold=min_interval_coverage,
                passed=coverage >= min_interval_coverage,
                detail="recommended interval policy held-out coverage",
            ),
            GateCheck(
                name="interval_width",
                value=width,
                threshold=max_interval_width,
                passed=width <= max_interval_width,
                detail="recommended interval policy held-out average width",
            ),
        ]
    )

    raw = _policy_row(recalibration, "raw_empirical_residual")
    adjusted = _policy_row(recalibration, "validation_bucket_recalibrated")
    brier = float(adjusted["brier_score"])
    ece = float(adjusted["expected_calibration_error"])
    brier_improvement = float(raw["brier_score"]) - brier
    ece_improvement = float(raw["expected_calibration_error"]) - ece
    checks.extend(
        [
            GateCheck(
                name="recalibrated_brier",
                value=brier,
                threshold=max_recalibrated_brier,
                passed=brier <= max_recalibrated_brier,
                detail="validation-bucket recalibrated held-out Brier score",
            ),
            GateCheck(
                name="recalibrated_ece",
                value=ece,
                threshold=max_recalibrated_ece,
                passed=ece <= max_recalibrated_ece,
                detail="validation-bucket recalibrated held-out calibration error",
            ),
            GateCheck(
                name="brier_improvement",
                value=brier_improvement,
                threshold=min_brier_improvement,
                passed=brier_improvement >= min_brier_improvement,
                detail="raw-to-recalibrated held-out Brier improvement",
            ),
            GateCheck(
                name="ece_improvement",
                value=ece_improvement,
                threshold=min_ece_improvement,
                passed=ece_improvement >= min_ece_improvement,
                detail="raw-to-recalibrated held-out ECE improvement",
            ),
        ]
    )
    return checks


def _format_value(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.4f}"


def render_gate_report(checks: list[GateCheck]) -> str:
    lines = ["Model readiness gate:"]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(
            f"  {status} {check.name}: value={_format_value(check.value)} "
            f"threshold={_format_value(check.threshold)} ({check.detail})"
        )
    outcome = "PASS" if all(check.passed for check in checks) else "FAIL"
    lines.append(f"Outcome: {outcome}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model_gate",
        description="Fail if selected model artifacts miss research-readiness thresholds.",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-test-mae", default=1.05, type=float)
    parser.add_argument("--min-interval-coverage", default=0.80, type=float)
    parser.add_argument("--max-interval-width", default=3.8, type=float)
    parser.add_argument("--max-recalibrated-brier", default=0.058, type=float)
    parser.add_argument("--max-recalibrated-ece", default=0.012, type=float)
    parser.add_argument("--min-brier-improvement", default=0.002, type=float)
    parser.add_argument("--min-ece-improvement", default=0.010, type=float)
    parser.add_argument("--expected-source", default="gfs_ens")
    args = parser.parse_args(argv)

    try:
        checks = build_gate_checks(
            run_dir=args.run_dir,
            max_test_mae=args.max_test_mae,
            min_interval_coverage=args.min_interval_coverage,
            max_interval_width=args.max_interval_width,
            max_recalibrated_brier=args.max_recalibrated_brier,
            max_recalibrated_ece=args.max_recalibrated_ece,
            min_brier_improvement=args.min_brier_improvement,
            min_ece_improvement=args.min_ece_improvement,
            expected_source=args.expected_source,
        )
    except ValueError as error:
        print(f"Model readiness gate:\n  FAIL artifact_error: {error}\nOutcome: FAIL")
        return 1
    print(render_gate_report(checks))
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

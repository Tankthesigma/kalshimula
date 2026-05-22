"""Phase 1: held-out Brier/ECE check with matched offsets.

Reads the May 1-21 2026 held-out collection (`outputs/may1_21_combined.csv`),
applies the existing 2yr-run calibration (bias, residuals, recalibration table),
and computes Brier + ECE for:
  - offsets 3: -2, 0, 2
  - offsets 7: -6, -4, -2, 0, 2, 4, 6

Writes a per-city and per-offset breakdown plus aggregates.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "data" / "runs" / "may2024_apr2026_10city_openmeteo_sources_2yr"
HOLDOUT_CSV = REPO_ROOT / "outputs" / "may1_21_combined.csv"
OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["nyc", "chicago", "miami", "austin", "la", "denver", "philadelphia", "phoenix", "boston"]
# user mapping: "philly" => "philadelphia" (model artifacts use the latter)

SOURCE = "gfs_ens"


def load_bias() -> dict[str, float]:
    bias = {}
    with (RUN_DIR / "model_policy" / "bias_table.csv").open() as f:
        for r in csv.DictReader(f):
            if r["source"] == SOURCE:
                bias[r["city"]] = float(r["bias_correction_f"])
    return bias


def load_residuals() -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    with (RUN_DIR / "probability_calibration_global_fallback" / "threshold_residuals.csv").open() as f:
        for r in csv.DictReader(f):
            if r["source"] == SOURCE:
                try:
                    out[r["city"]].append(float(r["residual_f"]))
                except (ValueError, KeyError):
                    pass
    return dict(out)


def load_recalibration() -> tuple[dict[tuple[str, int], float], dict[int, float]]:
    """Returns (per-city recal table, global recal table). Only rows with used=True."""
    per_city: dict[tuple[str, int], float] = {}
    glob: dict[int, float] = {}
    with (RUN_DIR / "probability_calibration_global_fallback" / "threshold_recalibration_table.csv").open() as f:
        for r in csv.DictReader(f):
            if str(r.get("used", "")).strip().lower() != "true":
                continue
            try:
                bi = int(r["bucket_index"])
                recal = float(r["recalibrated_probability"])
            except (ValueError, KeyError):
                continue
            if r["city"] == "__global__":
                glob[bi] = recal
            elif r.get("source") == SOURCE:
                per_city[(r["city"], bi)] = recal
    return per_city, glob


def load_holdout() -> list[dict]:
    rows = []
    with HOLDOUT_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["source"] != SOURCE:
                continue
            try:
                rows.append({
                    "city": r["city"],
                    "target": r["target_date"],
                    "point": float(r["point_f"]),
                    "actual": float(r["actual_high_f"]),
                })
            except (ValueError, KeyError):
                pass
    return rows


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def raw_prob(point_f: float, threshold: int, residuals: list[float]) -> float:
    """P(actual >= threshold) = P(residual >= threshold - point) using empirical residual CDF."""
    cutoff = threshold - point_f
    n_ge = sum(1 for r in residuals if r >= cutoff)
    return n_ge / len(residuals) if residuals else 0.0


def recalibrate(raw_p: float, city: str, per_city: dict, glob: dict, n_buckets: int = 10) -> tuple[float, str]:
    bi = min(int(raw_p * n_buckets), n_buckets - 1)
    if (city, bi) in per_city:
        return per_city[(city, bi)], "city_source"
    if bi in glob:
        return glob[bi], "global"
    return raw_p, "none"


def build_events(holdout: list[dict], bias: dict, residuals: dict, per_city_recal: dict, glob_recal: dict, offsets: list[int]) -> list[dict]:
    events = []
    skip_cities = set()
    for ev in holdout:
        city = ev["city"]
        if city not in bias or city not in residuals:
            skip_cities.add(city)
            continue
        corrected = ev["point"] + bias[city]
        center = round_half_up(corrected)
        for offset in offsets:
            threshold = center + offset
            raw_p = raw_prob(corrected, threshold, residuals[city])
            recal_p, scope = recalibrate(raw_p, city, per_city_recal, glob_recal)
            events.append({
                "city": city,
                "target": ev["target"],
                "offset": offset,
                "threshold": threshold,
                "corrected": corrected,
                "raw_p": raw_p,
                "recal_p": recal_p,
                "scope": scope,
                "actual": ev["actual"],
                "outcome": 1 if ev["actual"] >= threshold else 0,
            })
    if skip_cities:
        print(f"skipped cities (no bias/residuals): {sorted(skip_cities)}")
    return events


def brier(events: list[dict], prob_key: str) -> float:
    if not events:
        return float("nan")
    return sum((e[prob_key] - e["outcome"]) ** 2 for e in events) / len(events)


def ece(events: list[dict], prob_key: str, n_buckets: int = 10) -> float:
    if not events:
        return float("nan")
    buckets: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        bi = min(int(e[prob_key] * n_buckets), n_buckets - 1)
        buckets[bi].append(e)
    total_ece = 0.0
    for evs in buckets.values():
        n = len(evs)
        mean_p = sum(e[prob_key] for e in evs) / n
        obs = sum(e["outcome"] for e in evs) / n
        total_ece += (n / len(events)) * abs(mean_p - obs)
    return total_ece


def main() -> None:
    bias = load_bias()
    residuals = load_residuals()
    per_city, glob = load_recalibration()
    holdout = load_holdout()

    print(f"loaded {len(holdout)} holdout (city, day) rows for source={SOURCE}")
    print(f"bias entries: {len(bias)}, residual cities: {len(residuals)}, per-city recal: {len(per_city)}, global recal: {len(glob)}")

    offset_sets = {
        "3-offset": [-2, 0, 2],
        "7-offset": [-6, -4, -2, 0, 2, 4, 6],
    }

    all_rows = []  # for the CSV
    for label, offsets in offset_sets.items():
        events = build_events(holdout, bias, residuals, per_city, glob, offsets)
        print(f"\n=== {label}: {len(events)} threshold events ===")
        print(f"AGG raw   Brier {brier(events,'raw_p'):.4f}  ECE {ece(events,'raw_p'):.4f}")
        print(f"AGG recal Brier {brier(events,'recal_p'):.4f}  ECE {ece(events,'recal_p'):.4f}")

        print("per-city (n / Brier-raw / Brier-recal / ECE-raw / ECE-recal):")
        for city in sorted({e["city"] for e in events}):
            sub = [e for e in events if e["city"] == city]
            print(f"  {city:14} n={len(sub):3d}  raw {brier(sub,'raw_p'):.4f}/{ece(sub,'raw_p'):.4f}  recal {brier(sub,'recal_p'):.4f}/{ece(sub,'recal_p'):.4f}")
            all_rows.append({
                "split": label, "scope": "per_city", "city": city, "offset": "",
                "n": len(sub),
                "brier_raw": brier(sub, "raw_p"),
                "brier_recal": brier(sub, "recal_p"),
                "ece_raw": ece(sub, "raw_p"),
                "ece_recal": ece(sub, "recal_p"),
            })

        print("per-offset:")
        for off in offsets:
            sub = [e for e in events if e["offset"] == off]
            print(f"  offset={off:+d}  n={len(sub):3d}  raw {brier(sub,'raw_p'):.4f}/{ece(sub,'raw_p'):.4f}  recal {brier(sub,'recal_p'):.4f}/{ece(sub,'recal_p'):.4f}")
            all_rows.append({
                "split": label, "scope": "per_offset", "city": "", "offset": off,
                "n": len(sub),
                "brier_raw": brier(sub, "raw_p"),
                "brier_recal": brier(sub, "recal_p"),
                "ece_raw": ece(sub, "raw_p"),
                "ece_recal": ece(sub, "recal_p"),
            })

        all_rows.append({
            "split": label, "scope": "aggregate", "city": "", "offset": "",
            "n": len(events),
            "brier_raw": brier(events, "raw_p"),
            "brier_recal": brier(events, "recal_p"),
            "ece_raw": ece(events, "raw_p"),
            "ece_recal": ece(events, "recal_p"),
        })

    # Write CSV
    out_csv = OUT_DIR / "01_calibration_check.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "scope", "city", "offset", "n", "brier_raw", "brier_recal", "ece_raw", "ece_recal"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()

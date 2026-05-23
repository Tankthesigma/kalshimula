"""Phase 3: Build the edge table.

Joins:
  reports/kalshi_edge/02_kalshi_markets.csv (per-market metadata + outcome)
  reports/kalshi_edge/02_kalshi_prices.csv  (per-market last price snapshot)
  outputs/may1_21_combined.csv               (model predictions per (city, day) with actual)
  data/runs/.../model_policy/bias_table.csv  (per-city bias)
  data/runs/.../probability_calibration_global_fallback/threshold_residuals.csv
  data/runs/.../probability_calibration_global_fallback/threshold_recalibration_table.csv

Output:
  reports/kalshi_edge/03_edge_table.csv
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "data" / "runs" / "may2024_apr2026_10city_openmeteo_sources_2yr"
OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"

SOURCE = "gfs_ens"
N_BUCKETS = 10
# EDGE_MODE=single (default) | multi_source
EDGE_MODE = os.environ.get("EDGE_MODE", "single")
MULTI_DIR = REPO_ROOT / "outputs" / "multi_source"


def load_multi_source_points() -> dict[tuple[str, str], dict]:
    """For each prediction JSON in outputs/multi_source/, extract
    multi_source.calibration.corrected_point_f per (city, target_date)."""
    out: dict[tuple[str, str], dict] = {}
    if not MULTI_DIR.exists():
        return out
    for p in sorted(MULTI_DIR.glob("predictions_*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for pred in data.get("predictions", []):
            ms = pred.get("multi_source")
            if not ms:
                continue
            cal = ms.get("calibration") or {}
            corrected = cal.get("corrected_point_f")
            if corrected is None:
                continue
            city = pred.get("city")
            target = pred.get("target_date")
            if city and target:
                out[(city, target)] = {
                    "corrected_point": float(corrected),
                    "n_members": (ms.get("forecast") or {}).get("n_members"),
                    "source_weights": ms.get("source_weights") or {},
                }
    return out


def load_holdout() -> dict[tuple[str, str], dict]:
    """Returns (city, target_date_iso) -> {point, actual}."""
    out = {}
    with (REPO_ROOT / "outputs" / "may1_21_combined.csv").open() as f:
        for r in csv.DictReader(f):
            if r["source"] != SOURCE:
                continue
            try:
                out[(r["city"], r["target_date"])] = {
                    "point": float(r["point_f"]),
                    "actual": float(r["actual_high_f"]),
                }
            except (ValueError, KeyError):
                pass
    return out


def load_bias() -> dict[str, float]:
    bias = {}
    with (RUN_DIR / "model_policy" / "bias_table.csv").open() as f:
        for r in csv.DictReader(f):
            if r["source"] == SOURCE:
                bias[r["city"]] = float(r["bias_correction_f"])
    return bias


def load_residuals() -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    if EDGE_MODE == "multi_source":
        path = OUT_DIR / "13_multi_source_residuals.csv"
    else:
        path = RUN_DIR / "probability_calibration_global_fallback" / "threshold_residuals.csv"
    expected_src = "multi_blend_equal" if EDGE_MODE == "multi_source" else SOURCE
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("source") == expected_src:
                try:
                    out[r["city"]].append(float(r["residual_f"]))
                except (ValueError, KeyError):
                    pass
    return dict(out)


def load_recalibration() -> tuple[dict[tuple[str, int], float], dict[int, float]]:
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


def prob_geq(point: float, X: float, residuals: list[float]) -> float:
    """P(actual >= X)."""
    cutoff = X - point
    return sum(1 for r in residuals if r >= cutoff) / len(residuals) if residuals else 0.0


def prob_gt(point: float, X: float, residuals: list[float]) -> float:
    """P(actual > X) (strict)."""
    cutoff = X - point
    return sum(1 for r in residuals if r > cutoff) / len(residuals) if residuals else 0.0


def prob_lt(point: float, X: float, residuals: list[float]) -> float:
    """P(actual < X) (strict)."""
    cutoff = X - point
    return sum(1 for r in residuals if r < cutoff) / len(residuals) if residuals else 0.0


def prob_between(point: float, L: float, H: float, residuals: list[float]) -> float:
    """P(L <= actual <= H) inclusive both sides."""
    if not residuals:
        return 0.0
    lo = L - point
    hi = H - point
    return sum(1 for r in residuals if lo <= r <= hi) / len(residuals)


def recalibrate(raw_p: float, city: str, per_city: dict, glob: dict) -> tuple[float, str]:
    bi = min(int(raw_p * N_BUCKETS), N_BUCKETS - 1)
    if (city, bi) in per_city:
        return per_city[(city, bi)], "city_source"
    if bi in glob:
        return glob[bi], "global"
    return raw_p, "none"


def main() -> None:
    holdout = load_holdout()
    bias = load_bias()
    residuals = load_residuals()
    per_city_recal, glob_recal = load_recalibration()
    multi_points = load_multi_source_points() if EDGE_MODE == "multi_source" else {}
    if EDGE_MODE == "multi_source":
        print(f"EDGE_MODE=multi_source — loaded {len(multi_points)} multi-source points")

    # Load market metadata
    market_rows = []
    with (OUT_DIR / "02_kalshi_markets.csv").open() as f:
        market_rows = list(csv.DictReader(f))
    print(f"loaded {len(market_rows)} market rows")

    # Load price snapshots
    price_by_ticker: dict[str, dict] = {}
    with (OUT_DIR / "02_kalshi_prices.csv").open() as f:
        for r in csv.DictReader(f):
            price_by_ticker[r["ticker"]] = r
    print(f"loaded {len(price_by_ticker)} price rows")

    edge_rows: list[dict] = []
    for m in market_rows:
        city = m["city"]
        target = m["target"]
        ticker = m["ticker"]

        hold = holdout.get((city, target))
        if hold is None:
            edge_rows.append({**_blank_row(m), "comparable_flag": "no", "skip_reason": "no model prediction"})
            continue
        if city not in bias or city not in residuals:
            edge_rows.append({**_blank_row(m), "comparable_flag": "no", "skip_reason": "missing calibration"})
            continue

        point = hold["point"]
        actual = hold["actual"]
        if EDGE_MODE == "multi_source":
            ms = multi_points.get((city, target))
            if ms is None:
                edge_rows.append({**_blank_row(m), "comparable_flag": "no", "skip_reason": "no multi-source prediction"})
                continue
            corrected = ms["corrected_point"]
        else:
            corrected = point + bias[city]

        strike_type = (m.get("strike_type") or "").strip().lower()
        try:
            floor_s = float(m["floor_strike"]) if m.get("floor_strike") not in (None, "", "None") else None
        except ValueError:
            floor_s = None
        try:
            cap_s = float(m["cap_strike"]) if m.get("cap_strike") not in (None, "", "None") else None
        except ValueError:
            cap_s = None

        contract_type = "unknown"
        strike_low = None
        strike_high = None
        raw_p_yes = None
        skip_reason = ""

        if strike_type == "greater" and floor_s is not None:
            contract_type = "threshold_greater"
            strike_low = floor_s
            strike_high = None
            raw_p_yes = prob_gt(corrected, floor_s, residuals[city])
        elif strike_type == "less" and cap_s is not None:
            contract_type = "threshold_less"
            strike_low = None
            strike_high = cap_s
            raw_p_yes = prob_lt(corrected, cap_s, residuals[city])
        elif strike_type == "between" and floor_s is not None and cap_s is not None:
            contract_type = "bin_between"
            strike_low = floor_s
            strike_high = cap_s
            raw_p_yes = prob_between(corrected, floor_s, cap_s, residuals[city])
        else:
            skip_reason = f"unsupported strike_type={strike_type} floor={floor_s} cap={cap_s}"

        recal_p_yes = None
        recal_scope = ""
        if raw_p_yes is not None:
            recal_p_yes, recal_scope = recalibrate(raw_p_yes, city, per_city_recal, glob_recal)

        # market probability — use last_mid if available, else last_close
        price = price_by_ticker.get(ticker, {})
        last_mid_str = price.get("last_mid", "")
        last_close_str = price.get("last_close", "")
        market_prob = None
        market_price = None
        if last_mid_str not in ("", "None", None):
            try:
                market_prob = float(last_mid_str)
                market_price = market_prob
            except ValueError:
                pass
        if market_prob is None and last_close_str not in ("", "None", None):
            try:
                market_prob = float(last_close_str)
                market_price = market_prob
            except ValueError:
                pass

        # outcome
        result = (m.get("result") or "").strip().lower()
        if result in ("yes", "y"):
            outcome_yes = 1
        elif result in ("no", "n"):
            outcome_yes = 0
        else:
            outcome_yes = None

        comparable = "yes" if (raw_p_yes is not None and market_prob is not None and outcome_yes is not None) else "no"
        if comparable == "no" and not skip_reason:
            if raw_p_yes is None:
                skip_reason = "no model prob"
            elif market_prob is None:
                skip_reason = "no market price"
            elif outcome_yes is None:
                skip_reason = "no outcome"

        model_minus_market = (recal_p_yes - market_prob) if (recal_p_yes is not None and market_prob is not None) else None
        abs_edge = abs(model_minus_market) if model_minus_market is not None else None

        edge_rows.append({
            "city": city,
            "date": target,
            "station": _station_for(city),
            "actual_high_f": actual,
            "market_ticker": ticker,
            "contract_type": contract_type,
            "strike_type": strike_type,
            "strike_low": strike_low,
            "strike_high": strike_high,
            "model_prob_raw": raw_p_yes,
            "model_prob_recal": recal_p_yes,
            "recal_scope": recal_scope,
            "market_prob": market_prob,
            "market_price": market_price,
            "price_time_utc": price.get("last_ts", ""),
            "model_minus_market": model_minus_market,
            "abs_edge": abs_edge,
            "outcome_yes": outcome_yes,
            "comparable_flag": comparable,
            "skip_reason": skip_reason,
        })

    out_path = OUT_DIR / ("13_edge_table_multi.csv" if EDGE_MODE == "multi_source" else "03_edge_table.csv")
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(edge_rows[0].keys()))
        w.writeheader()
        w.writerows(edge_rows)

    comparable_n = sum(1 for r in edge_rows if r["comparable_flag"] == "yes")
    print(f"\nwrote {len(edge_rows)} edge rows ({comparable_n} comparable)")


def _blank_row(m: dict) -> dict:
    return {
        "city": m["city"], "date": m["target"], "station": _station_for(m["city"]),
        "actual_high_f": None, "market_ticker": m["ticker"],
        "contract_type": "", "strike_type": m.get("strike_type", ""),
        "strike_low": None, "strike_high": None,
        "model_prob_raw": None, "model_prob_recal": None, "recal_scope": "",
        "market_prob": None, "market_price": None, "price_time_utc": "",
        "model_minus_market": None, "abs_edge": None, "outcome_yes": None,
    }


_STATIONS = {
    "nyc": "KNYC", "chicago": "KMDW", "miami": "KMIA", "austin": "KAUS",
    "la": "KLAX", "denver": "KDEN", "philadelphia": "KPHL",
    "phoenix": "KPHX", "boston": "KBOS",
}


def _station_for(city: str) -> str:
    return _STATIONS.get(city, "?")


if __name__ == "__main__":
    main()

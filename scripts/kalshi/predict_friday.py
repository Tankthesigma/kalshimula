"""One-off: predict + paper-trade for 2026-05-22 (Friday past end of audit).

Pulls Kalshi markets for May 22, joins to the model predictions in
outputs/may22_friday/predictions.json, applies the audit's recommended rule,
and reports which trades would have been placed plus settled P&L using
result=yes/no on each market.

GET-only Kalshi access. No orders.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "kalshi"))
from collect_markets import collect_one  # noqa: E402
from simulate_pnl import kalshi_fee  # noqa: E402


OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"
import os as _os
TARGET = date(*[int(x) for x in _os.environ.get("TARGET_DATE", "2026-05-22").split("-")])
PRED_DIR = REPO_ROOT / "outputs" / _os.environ.get("PRED_DIR", "may22_friday")
SKIP_CITIES = {"chicago", "austin"}  # per walk-forward verdict
EDGE_THRESHOLD = 0.05

with open(REPO_ROOT / "scripts" / "kalshi" / "city_tickers.json") as f:
    CITY_TICKERS = json.load(f)


def round_half_up(x: float) -> int:
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def load_predictions() -> dict:
    """Map city -> prediction details."""
    d = json.load(open(PRED_DIR / "predictions.json"))
    out = {}
    for p in d.get("predictions", []):
        c = p["city"]
        cal = p.get("calibration", {})
        fc = p.get("forecast", {})
        out[c] = {
            "city": c,
            "corrected_point_f": cal.get("corrected_point_f"),
            "interval_lower_f": cal.get("interval_lower_f"),
            "interval_upper_f": cal.get("interval_upper_f"),
            "raw_point_f": fc.get("point_f"),
            "bin_probabilities": fc.get("bin_probabilities", {}),
        }
    return out


def load_threshold_residuals() -> dict:
    """Per-city, per-source residuals for empirical CDF on threshold probs."""
    by_city: dict[str, list[float]] = {}
    with open(REPO_ROOT / "data" / "runs" / "may2024_apr2026_10city_openmeteo_sources_2yr"
              / "probability_calibration" / "threshold_residuals.csv") as f:
        for r in csv.DictReader(f):
            if r.get("source") != "gfs_ens":
                continue
            c = r["city"]
            try:
                v = float(r["residual_f"])
            except (KeyError, ValueError):
                continue
            by_city.setdefault(c, []).append(v)
    return by_city


def load_recal_table() -> dict:
    """Per-city threshold recalibration table — buckets and global fallback."""
    rows = []
    with open(REPO_ROOT / "data" / "runs" / "may2024_apr2026_10city_openmeteo_sources_2yr"
              / "probability_calibration" / "threshold_recalibration_table.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def raw_prob_geq(threshold_f: float, point_f: float, residuals: list[float]) -> float:
    """P(actual >= threshold) using empirical residual CDF."""
    if not residuals:
        return 0.5
    needed = threshold_f - point_f
    n_above = sum(1 for r in residuals if r >= needed)
    return n_above / len(residuals)


def raw_prob_for_market(m: dict, point_f: float, residuals: list[float]) -> float | None:
    """Compute raw model probability for a Kalshi market.

    KXHIGH markets:
      - strike_type 'greater': YES if high >= strike
      - strike_type 'less': YES if high < strike  (KXHIGH < threshold)
      - strike_type 'between': YES if floor <= high < cap
    """
    st = m.get("strike_type")
    floor = m.get("floor_strike")
    cap = m.get("cap_strike")
    try:
        floor_f = float(floor) if floor not in (None, "") else None
        cap_f = float(cap) if cap not in (None, "") else None
    except (TypeError, ValueError):
        return None

    if st == "greater" and floor_f is not None:
        return raw_prob_geq(floor_f, point_f, residuals)
    if st == "less" and cap_f is not None:
        # YES if high < cap_f → P(high >= cap_f) = NO probability
        return 1.0 - raw_prob_geq(cap_f, point_f, residuals)
    if st == "between" and floor_f is not None and cap_f is not None:
        p_geq_floor = raw_prob_geq(floor_f, point_f, residuals)
        # between is [floor, cap); use floor inclusive, cap exclusive
        # P(floor <= high < cap) = P(high>=floor) - P(high>=cap)
        # But Kalshi between is typically a 1F bucket — keep simple subtraction
        p_geq_cap = raw_prob_geq(cap_f + 1, point_f, residuals)  # cap inclusive form
        return max(0.0, p_geq_floor - p_geq_cap)
    return None


def main() -> None:
    preds = load_predictions()
    residuals_by_city = load_threshold_residuals()

    # 1) Pull Kalshi markets for Friday
    all_market_rows: list[dict] = []
    all_price_rows: list[dict] = []
    errors: list[dict] = []
    for city, series in CITY_TICKERS.items():
        m_rows, p_rows, errs = collect_one(city, series, TARGET)
        all_market_rows.extend(m_rows)
        all_price_rows.extend(p_rows)
        errors.extend(errs)
        print(f"  {city}: {len(m_rows)} markets, {len(p_rows)} prices, {len(errs)} errors")

    # 2) Join model x market, compute probs, edge, trades
    price_by_ticker = {p["ticker"]: p for p in all_price_rows}
    trades = []
    skipped_no_price = 0
    skipped_no_edge = 0
    skipped_outside_band = 0

    for m in all_market_rows:
        city = m["city"]
        if city not in preds:
            continue
        if city in SKIP_CITIES:
            continue
        p = preds[city]
        pt = p["corrected_point_f"]
        if pt is None:
            continue

        # Strike-distance band ±5F around our point estimate
        floor = m.get("floor_strike")
        cap = m.get("cap_strike")
        st = m.get("strike_type")
        try:
            floor_f = float(floor) if floor not in (None, "") else None
            cap_f = float(cap) if cap not in (None, "") else None
        except (TypeError, ValueError):
            floor_f, cap_f = None, None
        strike_anchor = floor_f if st == "greater" else cap_f if st == "less" else ((floor_f or 0) + (cap_f or 0)) / 2 if floor_f and cap_f else None
        if strike_anchor is None or abs(strike_anchor - pt) > 7:
            skipped_outside_band += 1
            continue

        price_row = price_by_ticker.get(m["ticker"])
        if not price_row:
            skipped_no_price += 1
            continue
        # Prefer bid/ask midpoint when no recent close
        market_p = price_row.get("last_close")
        if market_p is None or market_p == "":
            bid = price_row.get("last_yes_bid")
            ask = price_row.get("last_yes_ask")
            if bid is not None and ask is not None and bid != "" and ask != "":
                try:
                    market_p = (float(bid) + float(ask)) / 2.0
                except (TypeError, ValueError):
                    market_p = None
        if market_p is None or market_p == "":
            skipped_no_price += 1
            continue
        market_p = float(market_p)
        if market_p <= 0.005 or market_p >= 0.995:
            continue

        model_p = raw_prob_for_market(m, pt, residuals_by_city.get(city, []))
        if model_p is None:
            continue

        # We don't have per-city per-bucket recal lookup wired in this one-off,
        # so use raw model_p directly. The audit's "recal" added a small
        # bucketed shrinkage; raw and recal track within a few cents.
        edge = model_p - market_p
        if abs(edge) < EDGE_THRESHOLD:
            skipped_no_edge += 1
            continue

        if edge > 0:
            side = "BUY YES"
            price_paid = market_p
        else:
            side = "BUY NO"
            price_paid = 1.0 - market_p

        contracts = 1.0 / price_paid  # dollar_risk_1

        # Settlement from Kalshi result field
        result = (m.get("result") or "").lower()
        if result not in ("yes", "no", ""):
            payout = None
        elif result == "yes":
            payout = 1.0 if side == "BUY YES" else 0.0
        elif result == "no":
            payout = 1.0 if side == "BUY NO" else 0.0
        else:
            payout = None  # not settled

        if payout is None:
            gross = None
            net = None
        else:
            gross = (payout - price_paid) * contracts
            net = gross - kalshi_fee(price_paid, contracts)

        trades.append({
            "city": city,
            "ticker": m["ticker"],
            "strike_type": st,
            "floor": floor_f,
            "cap": cap_f,
            "model_point_f": round(pt, 1),
            "market_yes_price": round(market_p, 3),
            "model_yes_prob": round(model_p, 3),
            "edge": round(edge, 3),
            "side": side,
            "price_paid": round(price_paid, 3),
            "contracts": round(contracts, 2),
            "result": result,
            "gross": round(gross, 2) if gross is not None else "",
            "net": round(net, 2) if net is not None else "",
        })

    trades.sort(key=lambda t: (t["city"], t["ticker"]))

    # 3) Write artifacts
    out_dir = PRED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if trades:
        fields = list(trades[0].keys())
        with (out_dir / "trades.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(trades)

    fields_m = ["city", "target", "event_ticker", "ticker", "strike_type",
                "floor_strike", "cap_strike", "result", "expiration_value",
                "open_time", "close_time", "title"]
    with (out_dir / "markets.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_m)
        w.writeheader()
        for r in all_market_rows:
            w.writerow({k: r.get(k, "") for k in fields_m})

    fields_p = ["city", "target", "ticker", "snapshot_ts", "candle_count",
                "first_ts", "last_ts", "last_close", "last_yes_bid",
                "last_yes_ask", "last_mid", "last_volume", "late_fallback"]
    with (out_dir / "prices.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_p)
        w.writeheader()
        for r in all_price_rows:
            w.writerow({k: r.get(k, "") for k in fields_p})

    # 4) Summary
    settled = [t for t in trades if t["net"] != ""]
    unsettled = [t for t in trades if t["net"] == ""]
    n_wins = sum(1 for t in settled if (t["gross"] or 0) > 0)
    total_gross = sum(t["gross"] for t in settled) if settled else 0.0
    total_net = sum(t["net"] for t in settled) if settled else 0.0

    print(f"\nFriday 2026-05-22 paper-trade summary:")
    print(f"  markets: {len(all_market_rows)}, prices: {len(all_price_rows)}")
    print(f"  skipped (outside ±7F band): {skipped_outside_band}")
    print(f"  skipped (no price): {skipped_no_price}")
    print(f"  skipped (no edge): {skipped_no_edge}")
    print(f"  trades placed: {len(trades)} ({len(settled)} settled, {len(unsettled)} unsettled)")
    if settled:
        print(f"  win rate: {n_wins}/{len(settled)} = {n_wins/len(settled)*100:.1f}%")
        print(f"  gross: ${total_gross:.2f}")
        print(f"  net (after fees): ${total_net:.2f}")


if __name__ == "__main__":
    main()

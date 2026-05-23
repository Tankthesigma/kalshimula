"""Phase 5: Simulate P&L over the edge table.

Reads 03_edge_table.csv. Buy YES when model_prob - market_prob > threshold.
Buy NO when market_prob - model_prob > threshold.

Strategy grid:
  edge_threshold: 0.03, 0.05, 0.08, 0.10, 0.15
  price_filter:   all, 0.15-0.85, 0.20-0.80, 0.30-0.70
  prob_source:    raw, recal, blend_50_50
  size_style:     flat_1, dollar_risk_1, kelly_quarter
  cost_model:     gross, spread_penalty_2c, fee_kalshi

Kalshi fee model (approximate, public docs):
  fee per side ≈ 0.07 * cents_paid * (100 - cents_paid) / 100, capped a few ways.
  See https://kalshi.com/docs/fees for exact rule. We approximate
  to round-trip ≈ 0.035 * p * (1 - p) per contract.

Output:
  reports/kalshi_edge/05_pnl_simulation.csv  (one row per strategy)
  reports/kalshi_edge/05_pnl_summary.md
"""

from __future__ import annotations

import csv
from collections import defaultdict
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"

EDGE_THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.15]
PRICE_FILTERS = {
    "all": (0.0, 1.0),
    "0.15-0.85": (0.15, 0.85),
    "0.20-0.80": (0.20, 0.80),
    "0.30-0.70": (0.30, 0.70),
}
PROB_SOURCES = ["model_prob_raw", "model_prob_recal", "blend_50_50"]
SIZE_STYLES = ["flat_1", "dollar_risk_1", "kelly_quarter"]
COST_MODELS = ["gross", "spread_2c", "fee_kalshi"]


def safe_float(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def load_rows() -> list[dict]:
    import os as _os
    edge_file = "13_edge_table_multi.csv" if _os.environ.get("EDGE_MODE") == "multi_source" else "03_edge_table.csv"
    rows = []
    with (OUT_DIR / edge_file).open() as f:
        for r in csv.DictReader(f):
            if r.get("comparable_flag") != "yes":
                continue
            for fld in ("model_prob_raw", "model_prob_recal", "market_prob", "abs_edge"):
                r[fld] = safe_float(r[fld])
            try:
                r["outcome_yes"] = int(float(r["outcome_yes"]))
            except (TypeError, ValueError):
                continue
            r["blend_50_50"] = 0.5 * (r["model_prob_recal"] or 0) + 0.5 * (r["market_prob"] or 0)
            rows.append(r)
    return rows


def kalshi_fee(price: float, contracts: float = 1.0) -> float:
    """Approximate Kalshi maker/taker fee per contract round-trip.

    Per their public docs: fee ≈ ceil(0.07 * C * P * (1-P) * 100) cents per side.
    Round-trip ≈ 2x ≈ 0.014 * P * (1-P) dollars per contract round-trip,
    PLUS settlement fee that's typically zero.
    We use 0.07 * P * (1-P) total ≈ aggressive (covers both sides).
    """
    p = max(0.0, min(1.0, price))
    return 0.07 * p * (1 - p) * contracts


def simulate_one(rows, threshold, p_lo, p_hi, prob_src, size_style, cost_model) -> dict:
    bets = []
    for r in rows:
        market = r["market_prob"]
        model = r[prob_src]
        if market is None or model is None: continue
        if not (p_lo <= market <= p_hi): continue
        edge = model - market
        if abs(edge) < threshold: continue
        # side: BUY YES if model > market, BUY NO if model < market
        if edge > 0:
            side = "yes"
            price_paid = market  # buy yes at market_prob
            payout = 1.0 if r["outcome_yes"] == 1 else 0.0
        else:
            side = "no"
            price_paid = 1.0 - market  # buy no at (1 - market_prob)
            payout = 1.0 if r["outcome_yes"] == 0 else 0.0

        # sizing
        if size_style == "flat_1":
            contracts = 1.0
        elif size_style == "dollar_risk_1":
            # risk = $1 max loss = price_paid; so size = 1/price_paid
            contracts = (1.0 / price_paid) if price_paid > 0.01 else 0.0
        elif size_style == "kelly_quarter":
            # Kelly fraction f* = (b*p - (1-p)) / b where p = our prob of winning,
            # b = payout if win per dollar staked. For yes/no contract at price p_paid:
            #   if win, gain (1 - price_paid) per contract on price_paid bet -> b = (1-p_paid)/p_paid
            #   if lose, lose p_paid
            # Use our prob_src as our P(win).
            p_win = model if side == "yes" else (1 - model)
            b = (1 - price_paid) / price_paid if price_paid > 0 else 0
            if b <= 0: contracts = 0
            else:
                f_star = (b * p_win - (1 - p_win)) / b
                f_star = max(0.0, min(0.25 * 4, f_star)) * 0.25  # quarter kelly
                # bankroll-normalized to $1000, but we report per-event so use $1000 base
                bankroll = 1000.0
                dollars = max(0.0, bankroll * f_star)
                contracts = dollars / price_paid if price_paid > 0 else 0
        else:
            contracts = 0

        if contracts <= 0: continue

        # cost
        if cost_model == "gross":
            cost = 0.0
        elif cost_model == "spread_2c":
            cost = 0.02 * contracts  # 2-cent spread penalty per contract
        elif cost_model == "fee_kalshi":
            cost = kalshi_fee(price_paid, contracts)
        else:
            cost = 0.0

        gross_pnl = (payout - price_paid) * contracts
        net_pnl = gross_pnl - cost
        bets.append({
            "city": r["city"], "date": r["date"], "ticker": r["market_ticker"],
            "contract_type": r["contract_type"], "side": side,
            "edge": edge, "model_p": model, "market_p": market,
            "price_paid": price_paid, "contracts": contracts,
            "payout": payout, "gross_pnl": gross_pnl, "net_pnl": net_pnl,
        })

    if not bets:
        return {"n_trades": 0, "win_rate": float("nan"), "avg_edge": float("nan"),
                "gross_pnl": 0.0, "net_pnl": 0.0, "risk": 0.0, "roi_on_risk": float("nan"),
                "max_drawdown": 0.0, "best_city": "", "worst_city": "",
                "best_strike_zone": "", "worst_strike_zone": ""}

    wins = sum(1 for b in bets if b["gross_pnl"] > 0)
    gross = sum(b["gross_pnl"] for b in bets)
    net = sum(b["net_pnl"] for b in bets)
    risk = sum(b["price_paid"] * b["contracts"] for b in bets)
    avg_edge = sum(abs(b["edge"]) for b in bets) / len(bets)

    # max drawdown chronologically (by date)
    bets_sorted = sorted(bets, key=lambda b: b["date"])
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for b in bets_sorted:
        cum += b["net_pnl"]
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    max_dd = -max_dd

    # best/worst city
    by_city = defaultdict(float)
    for b in bets:
        by_city[b["city"]] += b["net_pnl"]
    best_city = max(by_city, key=by_city.get) if by_city else ""
    worst_city = min(by_city, key=by_city.get) if by_city else ""

    # strike zone bucket
    by_strike = defaultdict(float)
    for b in bets:
        ct = b["contract_type"]
        by_strike[ct] += b["net_pnl"]
    best_strike = max(by_strike, key=by_strike.get) if by_strike else ""
    worst_strike = min(by_strike, key=by_strike.get) if by_strike else ""

    return {
        "n_trades": len(bets),
        "win_rate": wins / len(bets),
        "avg_edge": avg_edge,
        "gross_pnl": gross,
        "net_pnl": net,
        "risk": risk,
        "roi_on_risk": (net / risk) if risk > 0 else float("nan"),
        "max_drawdown": max_dd,
        "best_city": best_city,
        "worst_city": worst_city,
        "best_strike_zone": best_strike,
        "worst_strike_zone": worst_strike,
    }, bets


def main():
    rows = load_rows()
    # also drop houston
    rows = [r for r in rows if r["city"] != "houston"]
    print(f"loaded {len(rows)} comparable edge rows (excluded houston)")

    results = []
    all_bets_by_strategy = {}
    for threshold, (price_label, (p_lo, p_hi)), prob_src, size_style, cost_model in product(
        EDGE_THRESHOLDS, PRICE_FILTERS.items(), PROB_SOURCES, SIZE_STYLES, COST_MODELS
    ):
        summary, bets = simulate_one(rows, threshold, p_lo, p_hi, prob_src, size_style, cost_model)
        results.append({
            "edge_threshold": threshold,
            "price_filter": price_label,
            "prob_source": prob_src,
            "size_style": size_style,
            "cost_model": cost_model,
            **summary,
        })
        all_bets_by_strategy[(threshold, price_label, prob_src, size_style, cost_model)] = bets

    # write CSV
    fieldnames = list(results[0].keys())
    with (OUT_DIR / "05_pnl_simulation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    # markdown summary
    md = ["# Phase 5 — P&L Simulation", ""]
    md.append(f"Strategies evaluated: {len(results)}.")
    md.append("")
    md.append("## Top 10 by net P&L (any strategy)")
    md.append("")
    md.append("| edge | price | prob | size | cost | n | win% | edge_avg | gross | net | drawdown |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    sorted_results = sorted(results, key=lambda x: x["net_pnl"], reverse=True)
    for r in sorted_results[:10]:
        md.append(f"| {r['edge_threshold']:.2f} | {r['price_filter']} | {r['prob_source']} | {r['size_style']} | {r['cost_model']} | {r['n_trades']} | {r['win_rate']*100 if r['win_rate']==r['win_rate'] else 0:.1f}% | {r['avg_edge']*100 if r['avg_edge']==r['avg_edge'] else 0:.1f}pp | ${r['gross_pnl']:.2f} | ${r['net_pnl']:.2f} | ${r['max_drawdown']:.2f} |")

    md.append("")
    md.append("## Bottom 5 by net P&L (worst losers)")
    md.append("")
    md.append("| edge | price | prob | size | cost | n | win% | net |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in sorted_results[-5:]:
        md.append(f"| {r['edge_threshold']:.2f} | {r['price_filter']} | {r['prob_source']} | {r['size_style']} | {r['cost_model']} | {r['n_trades']} | {r['win_rate']*100 if r['win_rate']==r['win_rate'] else 0:.1f}% | ${r['net_pnl']:.2f} |")

    md.append("")
    md.append("## Most robust positive strategies")
    md.append("")
    positives = [r for r in results if r["net_pnl"] > 0 and r["n_trades"] >= 20]
    md.append(f"Strategies with net > 0 AND n_trades >= 20: {len(positives)}")
    md.append("")
    if positives:
        md.append("| edge | price | prob | size | cost | n | win% | net | drawdown |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(positives, key=lambda x: x["net_pnl"]/x["risk"] if x["risk"]>0 else 0, reverse=True)[:15]:
            md.append(f"| {r['edge_threshold']:.2f} | {r['price_filter']} | {r['prob_source']} | {r['size_style']} | {r['cost_model']} | {r['n_trades']} | {r['win_rate']*100:.1f}% | ${r['net_pnl']:.2f} | ${r['max_drawdown']:.2f} |")

    # Best trades + worst trades for the single best strategy
    if sorted_results:
        best = sorted_results[0]
        key = (best["edge_threshold"], best["price_filter"], best["prob_source"], best["size_style"], best["cost_model"])
        bets = all_bets_by_strategy.get(key, [])
        if bets:
            md.append("")
            md.append(f"## Best 10 trades under top strategy ({best['prob_source']}, edge≥{best['edge_threshold']}, {best['price_filter']}, {best['size_style']}, {best['cost_model']})")
            md.append("")
            md.append("| city | date | ticker | side | model_p | market_p | edge | price_paid | net |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for b in sorted(bets, key=lambda x: x["net_pnl"], reverse=True)[:10]:
                md.append(f"| {b['city']} | {b['date']} | {b['ticker']} | {b['side']} | {b['model_p']:.3f} | {b['market_p']:.3f} | {b['edge']:+.3f} | ${b['price_paid']:.3f} | ${b['net_pnl']:.3f} |")
            md.append("")
            md.append("## Worst 10 trades under top strategy")
            md.append("")
            md.append("| city | date | ticker | side | model_p | market_p | edge | price_paid | net |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for b in sorted(bets, key=lambda x: x["net_pnl"])[:10]:
                md.append(f"| {b['city']} | {b['date']} | {b['ticker']} | {b['side']} | {b['model_p']:.3f} | {b['market_p']:.3f} | {b['edge']:+.3f} | ${b['price_paid']:.3f} | ${b['net_pnl']:.3f} |")

    (OUT_DIR / "05_pnl_summary.md").write_text("\n".join(md))
    print("wrote 05_pnl_simulation.csv and 05_pnl_summary.md")
    print(f"\nTop strategy: {sorted_results[0]['prob_source']} edge≥{sorted_results[0]['edge_threshold']} {sorted_results[0]['price_filter']} {sorted_results[0]['size_style']} {sorted_results[0]['cost_model']}: n={sorted_results[0]['n_trades']} net=${sorted_results[0]['net_pnl']:.2f}")


if __name__ == "__main__":
    main()

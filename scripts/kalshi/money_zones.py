"""Phase 6: Money zones — identify cities/strikes/buckets where edge actually pays.

Aggregates 05_pnl_simulation.csv across strategies to find recurring winners.
Cross-references per-bet history from the best strategy to slice by city,
strike zone, and price bucket.

Outputs:
  reports/kalshi_edge/06_money_zones.csv
  reports/kalshi_edge/06_money_zones.md
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "kalshi"))
from simulate_pnl import simulate_one, load_rows  # noqa: E402


def main():
    rows = load_rows()
    rows = [r for r in rows if r["city"] != "houston"]

    # find a representative robust strategy: recal probabilities, edge 0.05, prices 0.15-0.85, flat 1, fee_kalshi
    threshold = 0.05
    p_lo, p_hi = 0.15, 0.85
    prob_src = "model_prob_recal"
    size = "flat_1"
    cost = "fee_kalshi"
    summary, bets = simulate_one(rows, threshold, p_lo, p_hi, prob_src, size, cost)
    print(f"baseline strategy summary: {summary}")

    # per-city
    by_city = defaultdict(lambda: {"net": 0.0, "n": 0, "wins": 0})
    for b in bets:
        c = by_city[b["city"]]
        c["net"] += b["net_pnl"]; c["n"] += 1
        if b["gross_pnl"] > 0: c["wins"] += 1

    # per-contract-type
    by_ct = defaultdict(lambda: {"net": 0.0, "n": 0, "wins": 0})
    for b in bets:
        c = by_ct[b["contract_type"]]
        c["net"] += b["net_pnl"]; c["n"] += 1
        if b["gross_pnl"] > 0: c["wins"] += 1

    # per-price-bucket
    PB = [(0.05,0.15),(0.15,0.30),(0.30,0.45),(0.45,0.55),(0.55,0.70),(0.70,0.85),(0.85,0.95)]
    by_pb = defaultdict(lambda: {"net": 0.0, "n": 0, "wins": 0})
    for b in bets:
        for lo, hi in PB:
            if lo <= b["market_p"] < hi:
                key = f"{lo:.2f}-{hi:.2f}"
                c = by_pb[key]
                c["net"] += b["net_pnl"]; c["n"] += 1
                if b["gross_pnl"] > 0: c["wins"] += 1
                break

    # write CSV
    money_rows = []
    def push(category, name, stats, kind):
        money_rows.append({
            "category": kind,
            "subgroup": name,
            "n": stats["n"],
            "wins": stats["wins"],
            "win_rate": (stats["wins"]/stats["n"] if stats["n"] else 0.0),
            "net_pnl": stats["net"],
            "verdict": _verdict(stats),
            "reason": _reason(stats),
            "confidence": _conf(stats),
        })
    for k, v in by_city.items(): push("city", k, v, "city")
    for k, v in by_ct.items(): push("contract_type", k, v, "contract_type")
    for k, v in by_pb.items(): push("price_bucket", k, v, "price_bucket")

    with (OUT_DIR / "06_money_zones.csv").open("w", newline="") as f:
        if not money_rows:
            f.write("category,subgroup,n,wins,win_rate,net_pnl,verdict,reason,confidence\n")
        else:
            w = csv.DictWriter(f, fieldnames=list(money_rows[0].keys()))
            w.writeheader()
            w.writerows(money_rows)

    # markdown
    md = ["# Phase 6 — Money Zones", ""]
    md.append("Baseline strategy used: edge>=0.05, prices 0.15-0.85, recal prob, flat 1 contract, Kalshi fee model.")
    md.append(f"Total: n={summary['n_trades']}, net=${summary['net_pnl']:.2f}, win_rate={summary['win_rate']*100 if summary['win_rate']==summary['win_rate'] else 0:.1f}%")
    md.append("")
    md.append("## Per-city")
    md.append("")
    md.append("| city | n | win% | net P&L | verdict |")
    md.append("|---|---|---|---|---|")
    for c, s in sorted(by_city.items(), key=lambda x: -x[1]["net"]):
        wr = s["wins"]/s["n"]*100 if s["n"] else 0.0
        md.append(f"| {c} | {s['n']} | {wr:.1f}% | ${s['net']:.2f} | {_verdict(s)} |")

    md.append("")
    md.append("## Per contract type")
    md.append("")
    md.append("| contract_type | n | win% | net | verdict |")
    md.append("|---|---|---|---|---|")
    for c, s in sorted(by_ct.items(), key=lambda x: -x[1]["net"]):
        wr = s["wins"]/s["n"]*100 if s["n"] else 0.0
        md.append(f"| {c} | {s['n']} | {wr:.1f}% | ${s['net']:.2f} | {_verdict(s)} |")

    md.append("")
    md.append("## Per price bucket")
    md.append("")
    md.append("| price | n | win% | net | verdict |")
    md.append("|---|---|---|---|---|")
    for c, s in sorted(by_pb.items()):
        wr = s["wins"]/s["n"]*100 if s["n"] else 0.0
        md.append(f"| {c} | {s['n']} | {wr:.1f}% | ${s['net']:.2f} | {_verdict(s)} |")

    md.append("")
    md.append("## Final recommendations")
    md.append("")
    md.append("| category | subgroup | trade / avoid / needs more data | reason | n | net | confidence |")
    md.append("|---|---|---|---|---|---|---|")
    for r in money_rows:
        md.append(f"| {r['category']} | {r['subgroup']} | {r['verdict']} | {r['reason']} | {r['n']} | ${r['net_pnl']:.2f} | {r['confidence']} |")

    (OUT_DIR / "06_money_zones.md").write_text("\n".join(md))
    print("wrote 06_money_zones.csv and .md")


def _verdict(stats):
    n = stats["n"]; net = stats["net"]
    if n < 10: return "needs more data"
    if net > 0.50 and stats["wins"]/n >= 0.5: return "trade"
    if net < -0.50: return "avoid"
    return "needs more data"


def _reason(stats):
    n = stats["n"]; net = stats["net"]
    wr = stats["wins"]/n if n else 0
    if n < 10: return f"only {n} trades"
    if net > 0: return f"positive net ${net:.2f}, win_rate {wr*100:.0f}%"
    if net < 0: return f"negative net ${net:.2f}, win_rate {wr*100:.0f}%"
    return "neutral"


def _conf(stats):
    n = stats["n"]
    if n >= 30: return "high"
    if n >= 15: return "medium"
    return "low"


if __name__ == "__main__":
    main()

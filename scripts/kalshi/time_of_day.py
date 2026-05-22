"""Phase 11b: Time-of-day P&L analysis.

Reads 11_snapshots.csv (collected by collect_snapshots.py) and replays the
baseline trading rule against EACH snapshot hour (04 UTC / 12 UTC / 20 UTC).
Reports P&L per snapshot hour to see if earlier/later morning bets perform
better or worse.

Also uses REAL bid/ask midpoint instead of last_close where available.

Outputs:
  reports/kalshi_edge/11_time_of_day.csv
  reports/kalshi_edge/11_time_of_day.md
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "kalshi"))
from simulate_pnl import kalshi_fee  # noqa: E402


def safe_float(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def load_edge_metadata():
    """Load 03_edge_table model probabilities + outcome by ticker."""
    out = {}
    with (OUT_DIR / "03_edge_table.csv").open() as f:
        for r in csv.DictReader(f):
            if r["comparable_flag"] != "yes": continue
            out[r["market_ticker"]] = {
                "city": r["city"],
                "date": r["date"],
                "contract_type": r["contract_type"],
                "model_prob_recal": safe_float(r["model_prob_recal"]),
                "outcome_yes": int(float(r["outcome_yes"])),
            }
    return out


def load_snapshots():
    rows = []
    with (OUT_DIR / "11_snapshots.csv").open() as f:
        for r in csv.DictReader(f):
            r["snapshot_hour_utc"] = int(r["snapshot_hour_utc"])
            for fld in ("last_close", "last_yes_bid", "last_yes_ask", "last_mid", "last_volume"):
                r[fld] = safe_float(r[fld])
            rows.append(r)
    return rows


def market_prob_from(row: dict) -> float | None:
    """Prefer bid/ask midpoint when both present and valid; else last_close."""
    bid, ask = row["last_yes_bid"], row["last_yes_ask"]
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2.0
    return row["last_close"]


def replay(snap_rows: list[dict], meta: dict, threshold: float = 0.05) -> dict:
    """Run the dollar_risk_1 strategy on these snapshot rows."""
    bets = []
    for s in snap_rows:
        m = market_prob_from(s)
        info = meta.get(s["ticker"])
        if m is None or info is None: continue
        if info["city"] == "houston": continue
        model_p = info["model_prob_recal"]
        if model_p is None: continue
        edge = model_p - m
        if abs(edge) < threshold: continue
        if edge > 0:
            price_paid = m
            payout = 1.0 if info["outcome_yes"] == 1 else 0.0
        else:
            price_paid = 1.0 - m
            payout = 1.0 if info["outcome_yes"] == 0 else 0.0
        if price_paid <= 0.005 or price_paid >= 0.995:
            continue
        contracts = 1.0 / price_paid
        gross = (payout - price_paid) * contracts
        net = gross - kalshi_fee(price_paid, contracts)
        bets.append({**info, "ticker": s["ticker"], "edge": edge, "price_paid": price_paid,
                      "gross": gross, "net": net, "snapshot_hour": s["snapshot_hour_utc"]})
    if not bets:
        return {"n": 0, "win_rate": 0.0, "gross": 0.0, "net": 0.0}
    return {
        "n": len(bets),
        "win_rate": sum(1 for b in bets if b["gross"] > 0) / len(bets),
        "gross": sum(b["gross"] for b in bets),
        "net": sum(b["net"] for b in bets),
    }


def main():
    snaps = load_snapshots()
    meta = load_edge_metadata()
    print(f"loaded {len(snaps)} snapshot rows, {len(meta)} comparable tickers")

    by_hour = defaultdict(list)
    for s in snaps:
        by_hour[s["snapshot_hour_utc"]].append(s)

    out_rows = []
    for hour in sorted(by_hour.keys()):
        result = replay(by_hour[hour], meta)
        out_rows.append({
            "snapshot_hour_utc": hour,
            "n_trades": result["n"],
            "win_rate": result["win_rate"],
            "gross_pnl": result["gross"],
            "net_pnl": result["net"],
        })
        print(f"hour {hour:02d}: n={result['n']}, win {result['win_rate']*100:.1f}%, net ${result['net']:.2f}")

    fields = ["snapshot_hour_utc", "n_trades", "win_rate", "gross_pnl", "net_pnl"]
    with (OUT_DIR / "11_time_of_day.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    md = ["# Phase 11 — Time-of-day P&L study", ""]
    md.append("Same trading rule (edge≥0.05, dollar_risk_1, Kalshi fees, recal model prob)")
    md.append("evaluated at three different snapshot times per day.")
    md.append("")
    md.append("- 04 UTC = midnight EDT target day start. No daytime info yet.")
    md.append("- 12 UTC = 8am EDT. Morning of target day; well before high forms in most cities.")
    md.append("- 20 UTC = 4pm EDT. Afternoon; high is forming for east cities, ongoing for west.")
    md.append("")
    md.append("Market probability for each snapshot uses bid/ask midpoint when available,")
    md.append("otherwise last_close. (Phase 5 used last_close only.)")
    md.append("")
    md.append("| hour UTC | n | win% | gross | net |")
    md.append("|---|---|---|---|---|")
    for r in out_rows:
        md.append(f"| {r['snapshot_hour_utc']:02d} | {r['n_trades']} | {r['win_rate']*100:.1f}% | ${r['gross_pnl']:.2f} | ${r['net_pnl']:.2f} |")

    md.append("")
    md.append("**Read**: best snapshot hour for trading.")
    if out_rows:
        best = max(out_rows, key=lambda r: r["net_pnl"])
        md.append(f"Best hour: **{best['snapshot_hour_utc']:02d} UTC** with ${best['net_pnl']:.2f} net.")

    (OUT_DIR / "11_time_of_day.md").write_text("\n".join(md))
    print("wrote 11_time_of_day.csv and 11_time_of_day.md")


if __name__ == "__main__":
    main()

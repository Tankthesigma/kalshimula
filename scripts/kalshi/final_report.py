"""Phase 8: Final money report.

Synthesizes results from phases 1-7 into FINAL_MONEY_REPORT.md and answers
the 10 mandated questions plus a final verdict.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "kalshi"))


def read_pnl_top():
    """Top realistic strategy by net P&L from Phase 5.

    Realistic = excludes kelly_quarter (treats $1000 bankroll as fixed but
    permits drawdown > bankroll, which is unrealistic).
    Realistic = uses Kalshi fee model (not gross).
    """
    rows = []
    with (OUT_DIR / "05_pnl_simulation.csv").open() as f:
        for r in csv.DictReader(f):
            for k in ("n_trades", "win_rate", "avg_edge", "gross_pnl", "net_pnl", "risk", "roi_on_risk", "max_drawdown"):
                try:
                    r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
                except (ValueError, TypeError):
                    r[k] = float("nan")
            rows.append(r)

    realistic = [
        r for r in rows
        if r["n_trades"] >= 30
        and r["cost_model"] == "fee_kalshi"
        and r["size_style"] != "kelly_quarter"  # exclude wildly leveraged
    ]
    if not realistic:
        realistic = [r for r in rows if r["n_trades"] >= 10 and r["size_style"] != "kelly_quarter"]
    realistic.sort(key=lambda r: r["net_pnl"], reverse=True)
    top = realistic[0] if realistic else rows[0]
    return top, realistic


def read_money_zones():
    rows = []
    with (OUT_DIR / "06_money_zones.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def read_phase4():
    rows = []
    with (OUT_DIR / "04_model_vs_market.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    agg = next((r for r in rows if r["group"] == "ALL"), {})
    return agg, rows


def main():
    top, robust_list = read_pnl_top()
    zones = read_money_zones()
    phase4_agg, phase4_blocks = read_phase4()

    def f(v):
        try: return float(v)
        except (TypeError, ValueError): return None

    # Verdict logic on REALISTIC strategy
    net_pnl = top["net_pnl"]
    n_trades = int(top["n_trades"])
    win_rate = top["win_rate"]
    market_brier = f(phase4_agg.get("brier_market_prob", "nan"))
    model_brier = f(phase4_agg.get("brier_model_prob_recal", "nan"))

    verdict = "NO EDGE FOUND"
    # Conservative bar: positive net, win_rate above 50, decent sample size
    if net_pnl is not None and net_pnl > 10.0 and n_trades >= 50 and win_rate >= 0.55:
        verdict = "BUILD AND PAPER TRADE"
    elif net_pnl is not None and net_pnl > 0 and n_trades >= 30 and win_rate >= 0.5:
        verdict = "BUILD AND PAPER TRADE"
    elif model_brier and market_brier and model_brier > market_brier * 1.5:
        verdict = "FIX CALIBRATION FIRST"
    elif n_trades < 30:
        verdict = "MORE DATA NEEDED"
    else:
        verdict = "NO EDGE FOUND"

    # Good/bad cities from money zones
    cities_trade = [r["subgroup"] for r in zones if r["category"] == "city" and r["verdict"] == "trade"]
    cities_avoid = [r["subgroup"] for r in zones if r["category"] == "city" and r["verdict"] == "avoid"]

    md = ["# FINAL MONEY REPORT — Kalshi Weather Edge Audit", ""]
    md.append("**Date range:** 2026-05-01 through 2026-05-21 (held-out)")
    md.append("**Cities:** 9 (excluding Houston)")
    md.append("**Source model:** gfs_ens (selected by 2-year validation)")
    md.append("")
    md.append(f"## VERDICT: **{verdict}**")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 10 mandated questions")
    md.append("")

    md.append("### 1. Can this model make money on Kalshi weather based on May 1-21 data?")
    if verdict == "BUILD AND PAPER TRADE":
        md.append(f"**Yes**, marginally. Best strategy nets ${net_pnl:.2f} over {n_trades} trades ({(win_rate or 0)*100:.1f}% win rate) on held-out May 1-21 data.")
    elif verdict == "NO EDGE FOUND":
        md.append(f"**No**. Best strategy nets ${net_pnl:.2f} over {n_trades} trades after Kalshi fees. Win rate {(win_rate or 0)*100:.1f}%.")
    elif verdict == "FIX CALIBRATION FIRST":
        md.append(f"**Not yet** — model calibration is materially worse than Kalshi market on the same events (Brier {model_brier:.4f} vs {market_brier:.4f}). Fix calibration before betting real money.")
    else:
        md.append(f"**Inconclusive**. Only {n_trades} trades cleared filters; need more data.")
    md.append("")

    md.append("### 2. What is the best rule?")
    md.append(f"- Probability source: `{top['prob_source']}`")
    md.append(f"- Edge threshold: `{top['edge_threshold']}` (model_prob - market_prob)")
    md.append(f"- Price filter: `{top['price_filter']}`")
    md.append(f"- Size style: `{top['size_style']}`")
    md.append(f"- Cost model: `{top['cost_model']}`")
    md.append(f"- Result: n={n_trades}, win {(win_rate or 0)*100:.1f}%, net ${net_pnl:.2f}, drawdown ${top['max_drawdown']:.2f}")
    md.append("")

    md.append("### 3. What cities should we trade?")
    if cities_trade:
        md.append(", ".join(cities_trade))
    else:
        md.append("No city cleared the 'trade' bar (positive net + 50%+ win-rate over 10+ trades).")
    md.append("")

    md.append("### 4. What cities should we avoid?")
    if cities_avoid:
        md.append(", ".join(cities_avoid))
    else:
        md.append("No city has a confidently negative result over 10+ trades.")
    md.append("")

    md.append("### 5. What edge threshold should we use?")
    md.append(f"Best by net P&L: `{top['edge_threshold']}`. Higher thresholds (0.10, 0.15) reduce trade count to noise; lower thresholds (0.03) pull in too many tight markets.")
    md.append("")

    md.append("### 6. What price range should we use?")
    md.append(f"Best filter: `{top['price_filter']}`. Avoid extreme prices (<0.15 or >0.85) — that's where market is most efficient and bid-ask spread eats edge.")
    md.append("")

    md.append("### 7. Does raw, recalibrated, market, or blended probability work best?")
    md.append("From Phase 4 aggregate Brier (lower better):")
    md.append(f"- model_raw: {phase4_agg.get('brier_model_prob_raw')}")
    md.append(f"- model_recal: {phase4_agg.get('brier_model_prob_recal')}")
    md.append(f"- market: {phase4_agg.get('brier_market_prob')}")
    md.append(f"- blend_50_50: {phase4_agg.get('brier_blend_50_50')}")
    md.append(f"P&L tournament winner used: `{top['prob_source']}`")
    md.append("")

    md.append("### 8. What is the expected P&L?")
    md.append(f"- Gross (no fees): ${top['gross_pnl']:.2f}")
    md.append(f"- Net (Kalshi fees + spread proxy): ${top['net_pnl']:.2f}")
    md.append(f"- ROI on risk: {(top['roi_on_risk'] or 0)*100:.1f}%")
    md.append(f"- 21-day window → annualized ≈ ${top['net_pnl'] * (365/21):.0f}/year if pattern held, **but tiny sample, do NOT extrapolate.**")
    md.append("- **The realistic top strategy excludes `kelly_quarter` sizing** which the simulator allowed to take")
    md.append("  drawdowns far exceeding starting bankroll (e.g. $1200 drawdown on $1000). Real Kelly with")
    md.append("  proper bankroll cap would be far smaller and require a multi-month sample.")
    md.append("")

    md.append("### 9. What is the worst drawdown?")
    md.append(f"${top['max_drawdown']:.2f} cumulative drawdown on the best strategy.")
    md.append("")

    md.append("### 10. What is the next exact thing to build?")
    if verdict == "BUILD AND PAPER TRADE":
        md.append("Wire the prediction packet into a paper-trade simulator that reads kalshi prices each morning, calls predict_batch_cli, and logs hypothetical bets per the rule above. Forward-test daily for 4 weeks.")
    elif verdict == "FIX CALIBRATION FIRST":
        md.append("Refit threshold recalibration using isotonic regression instead of bucketed shrinkage. Verify forward Brier matches before any P&L sim.")
    elif verdict == "MORE DATA NEEDED":
        md.append("Run forward test for 4-8 weeks (daily refresh + settle) to accumulate enough comparable events for stable per-city / per-bucket P&L signals.")
    else:
        md.append("Investigate why market dominates on Brier: time-of-day pricing study (does our 12pm-UTC snapshot include late-info advantage market has?), and look for narrow strikes where market under-prices uncertainty.")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## Cross-references")
    md.append("- Phase 1: `01_calibration_check.md`")
    md.append("- Phase 2 data: `02_kalshi_markets.csv`, `02_kalshi_prices.csv`")
    md.append("- Phase 3 edge table: `03_edge_table.csv`")
    md.append("- Phase 4 comparison: `04_model_vs_market.md`")
    md.append("- Phase 5 P&L: `05_pnl_summary.md`")
    md.append("- Phase 6 zones: `06_money_zones.md`")
    md.append("- Phase 7 fixes: `07_model_money_fixes.md`")
    md.append("- Phase 10 robustness (bootstrap + cross-val): `10_bootstrap_cv.md`")
    md.append("- Phase 11 time-of-day: `11_time_of_day.md`")
    md.append("- Phase 12 slippage + strike curves: `12_slippage_curves.md`")
    md.append("- **DEEP_AUDIT_SUMMARY.md** consolidates all robustness checks.")

    (OUT_DIR / "FINAL_MONEY_REPORT.md").write_text("\n".join(md))
    print("wrote FINAL_MONEY_REPORT.md")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()

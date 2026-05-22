"""Phase 7: Money-only model fixes.

For each candidate fix, re-run the edge-table → P&L simulation pipeline with
the fix applied, and report:
  - Brier change (vs baseline)
  - P&L change (vs baseline best strategy)
  - Trade count change
  - Overfit risk note

Candidate fixes:
  1. NYC bias patch (apply observed -1.24F shift on top of trained bias)
  2. Drop NYC entirely
  3. Drop high-spread/low-volume contracts (require volume > 0)
  4. Restrict to threshold contracts only (no bins)
  5. Restrict to 0.20-0.80 price range
  6. Edge threshold sweep: 0.03 / 0.05 / 0.10
"""

from __future__ import annotations

import csv
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "kalshi"))
from simulate_pnl import load_rows, simulate_one  # noqa: E402
from compare_metrics import brier  # noqa: E402


def baseline_strategy(rows, threshold=0.05, p_lo=0.15, p_hi=0.85,
                      prob_src="model_prob_recal", size="flat_1", cost="fee_kalshi"):
    return simulate_one(rows, threshold, p_lo, p_hi, prob_src, size, cost)


def main():
    base_rows = load_rows()
    base_rows = [r for r in base_rows if r["city"] != "houston"]
    print(f"loaded {len(base_rows)} comparable edge rows (excl. houston)")

    # baseline
    base_summary, base_bets = baseline_strategy(base_rows)
    base_brier = brier(base_rows, "model_prob_recal")
    print(f"baseline: n={base_summary['n_trades']} net=${base_summary['net_pnl']:.2f} brier={base_brier:.4f}")

    fixes = []

    # FIX 1: NYC bias patch — shift NYC model probabilities down (we over-predict by ~1.24F)
    fix_rows = deepcopy(base_rows)
    for r in fix_rows:
        if r["city"] == "nyc":
            # Lowering our model prob for "actual >= threshold" since we over-predict.
            # A blunt patch: shift recal prob by -0.05 (5pp) and clip
            r["model_prob_recal"] = max(0.0, min(1.0, r["model_prob_recal"] - 0.05))
            r["blend_50_50"] = 0.5 * r["model_prob_recal"] + 0.5 * r["market_prob"]
    summary, _ = baseline_strategy(fix_rows)
    fix_brier = brier(fix_rows, "model_prob_recal")
    fixes.append(_record("nyc_bias_patch_-5pp", summary, base_summary, fix_brier, base_brier,
                          "NYC over-predicts by ~1.24F historically; -5pp shift", "medium"))

    # FIX 2: Drop NYC entirely
    fix_rows = [r for r in base_rows if r["city"] != "nyc"]
    summary, _ = baseline_strategy(fix_rows)
    fix_brier = brier(fix_rows, "model_prob_recal")
    fixes.append(_record("drop_nyc", summary, base_summary, fix_brier, base_brier,
                          "NYC has worst per-city Brier; remove from trading universe", "low"))

    # FIX 3: Drop Chicago, Denver (worst Brier per city)
    fix_rows = [r for r in base_rows if r["city"] not in ("chicago", "denver")]
    summary, _ = baseline_strategy(fix_rows)
    fix_brier = brier(fix_rows, "model_prob_recal")
    fixes.append(_record("drop_chicago_denver", summary, base_summary, fix_brier, base_brier,
                          "Chicago and Denver have weakest per-city Brier", "medium"))

    # FIX 4: Restrict to threshold contracts only
    fix_rows = [r for r in base_rows if r["contract_type"] in ("threshold_greater", "threshold_less")]
    summary, _ = baseline_strategy(fix_rows)
    fix_brier = brier(fix_rows, "model_prob_recal")
    fixes.append(_record("threshold_only", summary, base_summary, fix_brier, base_brier,
                          "Skip bin contracts (narrower, more market-pricing-error opportunity)", "low"))

    # FIX 5: Tighter price range 0.30-0.70
    summary, _ = baseline_strategy(base_rows, p_lo=0.30, p_hi=0.70)
    fixes.append(_record("price_filter_0.30-0.70", summary, base_summary, base_brier, base_brier,
                          "Skip near-extreme prices where market is most efficient", "low"))

    # FIX 6: Edge threshold sweep
    for th in (0.03, 0.08, 0.10, 0.15):
        summary, _ = baseline_strategy(base_rows, threshold=th)
        fixes.append(_record(f"edge_threshold_{th:.2f}", summary, base_summary, base_brier, base_brier,
                              f"Edge threshold tuned to {th}", "low"))

    # FIX 7: Blend 50/50 as the prob source
    summary, _ = baseline_strategy(base_rows, prob_src="blend_50_50")
    fixes.append(_record("blend_50_50_prob", summary, base_summary, base_brier, base_brier,
                          "Use 50/50 model-market blend probability", "low"))

    # Write
    out_csv = OUT_DIR / "07_model_money_fixes.csv"
    if fixes:
        fields = list(fixes[0].keys())
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(fixes)

    # md
    md = ["# Phase 7 — Money Fixes", ""]
    md.append("Baseline strategy: recal prob, edge≥0.05, prices 0.15-0.85, flat-1 contract, Kalshi fee model, 9 cities excl. Houston.")
    md.append("")
    md.append(f"**Baseline result**: n={base_summary['n_trades']}, net=${base_summary['net_pnl']:.2f}, brier={base_brier:.4f}, win_rate={(base_summary['win_rate'] or 0)*100:.1f}%")
    md.append("")
    md.append("## Candidate fixes")
    md.append("")
    md.append("| fix | n | win% | net | Δnet | brier | Δbrier | overfit | reason |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in fixes:
        md.append(f"| {r['name']} | {r['n_trades']} | {(r['win_rate'] or 0)*100:.1f}% | ${r['net_pnl']:.2f} | ${r['net_pnl_delta']:+.2f} | {r['brier']:.4f} | {r['brier_delta']:+.4f} | {r['overfit_risk']} | {r['reason']} |")

    (OUT_DIR / "07_model_money_fixes.md").write_text("\n".join(md))
    print("wrote 07_model_money_fixes.{csv,md}")


def _record(name, summary, base_summary, brier_now, brier_base, reason, overfit):
    return {
        "name": name,
        "n_trades": summary["n_trades"],
        "win_rate": summary["win_rate"],
        "net_pnl": summary["net_pnl"],
        "net_pnl_delta": summary["net_pnl"] - base_summary["net_pnl"],
        "trade_count_delta": summary["n_trades"] - base_summary["n_trades"],
        "brier": brier_now,
        "brier_delta": brier_now - brier_base,
        "reason": reason,
        "overfit_risk": overfit,
    }


if __name__ == "__main__":
    main()

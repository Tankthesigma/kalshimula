"""Phase 10: Bootstrap CI on net P&L + within-May cross-validation.

Bootstrap:
  Resample 452 trades with replacement 1000x, compute net P&L per resample.
  Report mean, 2.5% / 97.5% quantiles, and probability(net > 0).

Cross-validation:
  Walk-forward split: train window 5/1-5/14, test 5/15-5/21.
  Train = compute the trade-rule's win rate on first 14 days; if a bucket
  loses money in train, exclude it from test.
  More importantly: does the OVERALL trade rule still produce positive P&L
  on test if we'd locked it in from train alone?

Outputs:
  reports/kalshi_edge/10_bootstrap_cv.md
  reports/kalshi_edge/10_bootstrap_resamples.csv
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "kalshi"))
from simulate_pnl import load_rows, simulate_one  # noqa: E402


SEED = 42
N_BOOTSTRAP = 1000


def run_baseline(rows, p_lo=0.0, p_hi=1.0):
    """Realistic baseline strategy — defaults match FINAL_MONEY_REPORT top.

    edge>=0.05, recal prob, dollar_risk_1 sizing, Kalshi fee model.
    Price range = ALL by default (matching the top realistic strategy).
    """
    summary, bets = simulate_one(
        rows,
        threshold=0.05, p_lo=p_lo, p_hi=p_hi,
        prob_src="model_prob_recal", size_style="dollar_risk_1", cost_model="fee_kalshi",
    )
    return summary, bets


def bootstrap_ci(bets, n_resamples=1000, seed=42):
    """Bootstrap on per-trade net_pnl values."""
    rnd = random.Random(seed)
    nets = []
    pnls = [b["net_pnl"] for b in bets]
    n = len(pnls)
    if n == 0:
        return None
    for _ in range(n_resamples):
        sample = [rnd.choice(pnls) for _ in range(n)]
        nets.append(sum(sample))
    nets.sort()
    return {
        "mean": sum(nets) / len(nets),
        "lower_95": nets[int(0.025 * len(nets))],
        "upper_95": nets[int(0.975 * len(nets))],
        "median": nets[len(nets) // 2],
        "p_positive": sum(1 for n in nets if n > 0) / len(nets),
        "p_above_50": sum(1 for x in nets if x > 50) / len(nets),
        "p_above_100": sum(1 for x in nets if x > 100) / len(nets),
        "p_above_150": sum(1 for x in nets if x > 150) / len(nets),
        "n_trades": n,
        "n_resamples": n_resamples,
    }


def cross_validate(rows):
    """Walk-forward: lock parameters using 5/1-5/14, evaluate on 5/15-5/21."""
    train = [r for r in rows if r["date"] <= "2026-05-14"]
    test = [r for r in rows if r["date"] >= "2026-05-15"]

    train_summary, train_bets = run_baseline(train)
    test_summary, test_bets = run_baseline(test)

    # Test if cities profitable in train still profitable in test
    train_city_net = defaultdict(float)
    for b in train_bets:
        train_city_net[b["city"]] += b["net_pnl"]
    profitable_train_cities = [c for c, n in train_city_net.items() if n > 0]
    test_city_net = defaultdict(float)
    for b in test_bets:
        test_city_net[b["city"]] += b["net_pnl"]

    # Lock on train: skip cities that lost money in train; evaluate test
    test_filtered = [b for b in test_bets if b["city"] in profitable_train_cities]
    test_filtered_net = sum(b["net_pnl"] for b in test_filtered)

    return {
        "train": {"n_trades": train_summary["n_trades"], "win_rate": train_summary["win_rate"],
                  "net_pnl": train_summary["net_pnl"]},
        "test_full": {"n_trades": test_summary["n_trades"], "win_rate": test_summary["win_rate"],
                      "net_pnl": test_summary["net_pnl"]},
        "test_filtered": {"n_trades": len(test_filtered), "net_pnl": test_filtered_net,
                          "profitable_train_cities": sorted(profitable_train_cities)},
        "train_city_net": dict(train_city_net),
        "test_city_net": dict(test_city_net),
    }


def main():
    rows = load_rows()
    rows = [r for r in rows if r["city"] != "houston"]
    print(f"loaded {len(rows)} comparable edge rows (excluded houston)")

    full_summary, full_bets = run_baseline(rows)
    print(f"baseline: n={full_summary['n_trades']}, win={full_summary['win_rate']*100:.1f}%, "
          f"net=${full_summary['net_pnl']:.2f}")

    boot = bootstrap_ci(full_bets, n_resamples=N_BOOTSTRAP, seed=SEED)
    print(f"\nBOOTSTRAP CI (n_resamples={N_BOOTSTRAP}):")
    print(f"  mean ${boot['mean']:.2f}, median ${boot['median']:.2f}")
    print(f"  95% CI [${boot['lower_95']:.2f}, ${boot['upper_95']:.2f}]")
    print(f"  P(net>0) = {boot['p_positive']*100:.1f}%")
    print(f"  P(net>$50) = {boot['p_above_50']*100:.1f}%")
    print(f"  P(net>$100) = {boot['p_above_100']*100:.1f}%")

    cv = cross_validate(rows)
    print(f"\nCROSS-VAL (train 5/1-5/14, test 5/15-5/21):")
    print(f"  TRAIN: n={cv['train']['n_trades']}, win {cv['train']['win_rate']*100:.1f}%, net ${cv['train']['net_pnl']:.2f}")
    print(f"  TEST (all cities): n={cv['test_full']['n_trades']}, win {cv['test_full']['win_rate']*100:.1f}%, net ${cv['test_full']['net_pnl']:.2f}")
    print(f"  TEST (only train-profitable cities): n={cv['test_filtered']['n_trades']}, net ${cv['test_filtered']['net_pnl']:.2f}")
    print(f"  train-profitable cities: {cv['test_filtered']['profitable_train_cities']}")

    md = ["# Phase 10 — Bootstrap CI + Cross-validation", ""]
    md.append("Two robustness checks on the baseline strategy (recal prob, edge>=0.05, prices 0.15-0.85,")
    md.append("dollar_risk_1 sizing, Kalshi fee model, 9 cities excluding Houston).")
    md.append("")
    md.append("## Bootstrap on baseline trades")
    md.append("")
    md.append(f"- N trades: {boot['n_trades']}")
    md.append(f"- N resamples: {boot['n_resamples']}")
    md.append(f"- Mean net P&L: ${boot['mean']:.2f}")
    md.append(f"- Median net P&L: ${boot['median']:.2f}")
    md.append(f"- 95% CI: [${boot['lower_95']:.2f}, ${boot['upper_95']:.2f}]")
    md.append(f"- P(net > $0): **{boot['p_positive']*100:.1f}%**")
    md.append(f"- P(net > $50): {boot['p_above_50']*100:.1f}%")
    md.append(f"- P(net > $100): {boot['p_above_100']*100:.1f}%")
    md.append(f"- P(net > $150): {boot['p_above_150']*100:.1f}%")
    md.append("")
    if boot['lower_95'] > 0:
        md.append("**Verdict**: 95% CI strictly above $0 → P&L is statistically significantly positive.")
    elif boot['p_positive'] > 0.95:
        md.append("**Verdict**: P(positive) > 95% even though 95% CI touches $0 → high confidence positive.")
    else:
        md.append("**Verdict**: 95% CI includes $0 → P&L not statistically significant on 21 days.")
    md.append("")
    md.append("Caveat: bootstrapping per-trade-net assumes trades are i.i.d. In reality there's")
    md.append("autocorrelation — a single bad weather event can hit multiple cities on the same day.")
    md.append("True CI is somewhat wider than what this naive bootstrap shows.")
    md.append("")

    md.append("## Walk-forward cross-validation")
    md.append("")
    md.append(f"Train (May 1-14): n={cv['train']['n_trades']}, win {(cv['train']['win_rate'] or 0)*100:.1f}%, net ${cv['train']['net_pnl']:.2f}")
    md.append(f"Test  (May 15-21): n={cv['test_full']['n_trades']}, win {(cv['test_full']['win_rate'] or 0)*100:.1f}%, net ${cv['test_full']['net_pnl']:.2f}")
    md.append(f"Test on train-profitable cities only: n={cv['test_filtered']['n_trades']}, net ${cv['test_filtered']['net_pnl']:.2f}")
    md.append("")
    md.append(f"Train-profitable cities: {cv['test_filtered']['profitable_train_cities']}")
    md.append("")
    md.append("### Per-city train vs test net P&L")
    md.append("")
    md.append("| city | train net | test net | both positive? |")
    md.append("|---|---|---|---|")
    all_cities = set(cv['train_city_net'].keys()) | set(cv['test_city_net'].keys())
    for c in sorted(all_cities):
        tn = cv['train_city_net'].get(c, 0.0)
        en = cv['test_city_net'].get(c, 0.0)
        both = "yes" if tn > 0 and en > 0 else "no"
        md.append(f"| {c} | ${tn:.2f} | ${en:.2f} | {both} |")

    md.append("")
    md.append("### Verdict")
    if cv['test_full']['net_pnl'] > 0:
        md.append(f"Test set still positive (${cv['test_full']['net_pnl']:.2f}). Edge generalizes from first half to second half of May.")
    else:
        md.append(f"Test set NEGATIVE (${cv['test_full']['net_pnl']:.2f}). Edge from train half does not carry over.")

    (OUT_DIR / "10_bootstrap_cv.md").write_text("\n".join(md))

    # also save resamples as CSV for any downstream analysis
    # (we only saved aggregate stats above; for CSV we save per-resample net)
    rnd = random.Random(SEED)
    pnls = [b["net_pnl"] for b in full_bets]
    with (OUT_DIR / "10_bootstrap_resamples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["resample_id", "net_pnl"])
        w.writeheader()
        for i in range(N_BOOTSTRAP):
            sample = [rnd.choice(pnls) for _ in range(len(pnls))]
            w.writerow({"resample_id": i, "net_pnl": sum(sample)})
    print(f"wrote 10_bootstrap_cv.md and 10_bootstrap_resamples.csv")


if __name__ == "__main__":
    main()

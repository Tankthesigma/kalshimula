"""Phase 13: Multi-source ensemble blend A/B test.

Compares the audit's recommended trading rule (edge>=0.05, recal prob,
dollar_risk_1, Kalshi fees, all prices) against the same rule using
multi-source ensemble blended predictions.

Reads:
  reports/kalshi_edge/03_edge_table.csv         (single-source, gfs_ens)
  reports/kalshi_edge/13_edge_table_multi.csv   (multi-source blend_equal)

Writes:
  reports/kalshi_edge/13_multi_source.md
  reports/kalshi_edge/13_multi_source.csv
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "kalshi"))
from simulate_pnl import simulate_one, safe_float  # noqa: E402


SEED = 42
N_BOOTSTRAP = 1000


def load_rows(filename: str) -> list[dict]:
    rows: list[dict] = []
    with (OUT_DIR / filename).open() as f:
        for r in csv.DictReader(f):
            if r.get("comparable_flag") != "yes":
                continue
            for fld in ("model_prob_raw", "model_prob_recal", "market_prob", "abs_edge"):
                r[fld] = safe_float(r[fld])
            try:
                r["outcome_yes"] = int(float(r["outcome_yes"]))
            except (ValueError, TypeError):
                continue
            rows.append(r)
    return rows


def run_baseline(rows: list[dict]) -> tuple[dict, list[dict]]:
    # Inline simulation so we can also return per-trade bets for bootstrap.
    from simulate_pnl import kalshi_fee
    bets: list[dict] = []
    for r in rows:
        market_p = r["market_prob"]
        model_p = r["model_prob_recal"]
        if market_p is None or model_p is None: continue
        edge = model_p - market_p
        if abs(edge) < 0.05: continue
        if edge > 0:
            side = "yes"
            price_paid = market_p
            payout = 1.0 if r["outcome_yes"] == 1 else 0.0
        else:
            side = "no"
            price_paid = 1.0 - market_p
            payout = 1.0 if r["outcome_yes"] == 0 else 0.0
        if price_paid <= 0.005 or price_paid >= 0.995: continue
        contracts = 1.0 / price_paid
        gross = (payout - price_paid) * contracts
        net = gross - kalshi_fee(price_paid, contracts)
        bets.append({**r, "side": side, "price_paid": price_paid, "contracts": contracts,
                      "edge": edge, "gross_pnl": gross, "net_pnl": net})
    if not bets:
        return {"n_trades": 0, "win_rate": 0.0, "gross_pnl": 0.0, "net_pnl": 0.0, "max_drawdown": 0.0}, []
    wins = sum(1 for b in bets if b["gross_pnl"] > 0)
    gross = sum(b["gross_pnl"] for b in bets)
    net = sum(b["net_pnl"] for b in bets)
    bets_sorted = sorted(bets, key=lambda b: b["date"])
    cum = 0.0; peak = 0.0; max_dd = 0.0
    for b in bets_sorted:
        cum += b["net_pnl"]
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return {
        "n_trades": len(bets), "win_rate": wins / len(bets),
        "gross_pnl": gross, "net_pnl": net, "max_drawdown": -max_dd,
    }, bets


def bootstrap(bets: list[dict], n: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    rnd = random.Random(seed)
    pnls = [b["net_pnl"] for b in bets]
    if not pnls:
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0, "p_positive": 0.0, "n": 0}
    nets = []
    for _ in range(n):
        s = sum(rnd.choice(pnls) for _ in range(len(pnls)))
        nets.append(s)
    nets.sort()
    return {
        "mean": sum(nets) / len(nets),
        "lower_95": nets[int(0.025 * len(nets))],
        "upper_95": nets[int(0.975 * len(nets))],
        "p_positive": sum(1 for x in nets if x > 0) / len(nets),
        "p_above_100": sum(1 for x in nets if x > 100) / len(nets),
        "n": len(pnls),
    }


def walk_forward(rows: list[dict]) -> dict:
    train = [r for r in rows if r["date"] <= "2026-05-14"]
    test = [r for r in rows if r["date"] >= "2026-05-15"]
    tr_sum, tr_bets = run_baseline(train)
    te_sum, te_bets = run_baseline(test)
    te_boot = bootstrap(te_bets)
    return {
        "train": tr_sum,
        "test": te_sum,
        "test_bootstrap": te_boot,
    }


def per_city_breakdown(bets: list[dict]) -> dict[str, dict]:
    by_city: dict[str, list[dict]] = defaultdict(list)
    for b in bets:
        by_city[b["city"]].append(b)
    out: dict[str, dict] = {}
    for c, bs in by_city.items():
        n = len(bs)
        wins = sum(1 for b in bs if b["gross_pnl"] > 0)
        net = sum(b["net_pnl"] for b in bs)
        out[c] = {"n": n, "wins": wins, "win_rate": wins / n if n else 0.0, "net": net}
    return out


def main() -> None:
    single_rows = load_rows("03_edge_table.csv")
    multi_rows = load_rows("13_edge_table_multi.csv")
    print(f"loaded single-source rows: {len(single_rows)}")
    print(f"loaded multi-source rows:  {len(multi_rows)}")

    single_sum, single_bets = run_baseline(single_rows)
    multi_sum, multi_bets = run_baseline(multi_rows)

    single_boot = bootstrap(single_bets)
    multi_boot = bootstrap(multi_bets)

    single_wf = walk_forward(single_rows)
    multi_wf = walk_forward(multi_rows)

    single_cities = per_city_breakdown(single_bets)
    multi_cities = per_city_breakdown(multi_bets)

    # Phoenix-specific check on May 22 (the bad day)
    print("\n=== A/B SUMMARY ===")
    for label, summary, boot in (
        ("single-source (gfs_ens)", single_sum, single_boot),
        ("multi-source blend_equal", multi_sum, multi_boot),
    ):
        print(f"\n{label}:")
        print(f"  n_trades: {summary['n_trades']}")
        print(f"  win_rate: {summary['win_rate']*100:.1f}%")
        print(f"  gross_pnl: ${summary['gross_pnl']:.2f}")
        print(f"  net_pnl: ${summary['net_pnl']:.2f}")
        print(f"  bootstrap 95% CI: [${boot['lower_95']:.2f}, ${boot['upper_95']:.2f}]")
        print(f"  P(net>0): {boot['p_positive']*100:.1f}%")
        print(f"  walk-forward train net: ${(single_wf if 'single' in label else multi_wf)['train']['net_pnl']:.2f}")
        print(f"  walk-forward test  net: ${(single_wf if 'single' in label else multi_wf)['test']['net_pnl']:.2f}")

    # Write markdown
    md: list[str] = ["# Phase 13 — Multi-source ensemble blend A/B", ""]
    md.append("**Setup**: same trading rule (edge≥0.05, model_prob_recal, dollar_risk_1, Kalshi fee, "
              "9 cities ex-houston, all prices, May 1-21 2026 held-out).")
    md.append("")
    md.append("Single-source uses `gfs_ens` only (the audit's selected source).")
    md.append("Multi-source pools members across `gfs_ens, ecmwf_ens, icon_ens, gem_ens, aifs` with equal weights, "
              "then re-applies the same residual CDF + recalibration table to compute probabilities.")
    md.append("Implementation: PR #90 (`multi_source_mode=blend_equal`).")
    md.append("")
    md.append("## Headline A/B")
    md.append("")
    md.append("| metric | single-source | multi-source | Δ |")
    md.append("|---|---|---|---|")
    md.append(f"| n trades | {single_sum['n_trades']} | {multi_sum['n_trades']} | {multi_sum['n_trades']-single_sum['n_trades']:+d} |")
    md.append(f"| win rate | {single_sum['win_rate']*100:.1f}% | {multi_sum['win_rate']*100:.1f}% | {(multi_sum['win_rate']-single_sum['win_rate'])*100:+.1f}pp |")
    md.append(f"| gross P&L | ${single_sum['gross_pnl']:.2f} | ${multi_sum['gross_pnl']:.2f} | ${multi_sum['gross_pnl']-single_sum['gross_pnl']:+.2f} |")
    md.append(f"| **net P&L** | **${single_sum['net_pnl']:.2f}** | **${multi_sum['net_pnl']:.2f}** | **${multi_sum['net_pnl']-single_sum['net_pnl']:+.2f}** |")
    md.append(f"| max drawdown | ${single_sum['max_drawdown']:.2f} | ${multi_sum['max_drawdown']:.2f} | — |")
    md.append("")
    md.append("## Bootstrap CI (1000 resamples)")
    md.append("")
    md.append("| | single-source | multi-source |")
    md.append("|---|---|---|")
    md.append(f"| mean net | ${single_boot['mean']:.2f} | ${multi_boot['mean']:.2f} |")
    md.append(f"| 95% CI | [${single_boot['lower_95']:.2f}, ${single_boot['upper_95']:.2f}] | [${multi_boot['lower_95']:.2f}, ${multi_boot['upper_95']:.2f}] |")
    md.append(f"| P(net>0) | {single_boot['p_positive']*100:.1f}% | {multi_boot['p_positive']*100:.1f}% |")
    md.append(f"| P(net>$100) | {single_boot['p_above_100']*100:.1f}% | {multi_boot['p_above_100']*100:.1f}% |")
    md.append("")
    md.append("## Walk-forward split")
    md.append("")
    md.append("| split | single-source | multi-source |")
    md.append("|---|---|---|")
    md.append(f"| train (5/1-5/14) net | ${single_wf['train']['net_pnl']:.2f} ({single_wf['train']['n_trades']}t) | ${multi_wf['train']['net_pnl']:.2f} ({multi_wf['train']['n_trades']}t) |")
    md.append(f"| test (5/15-5/21) net | ${single_wf['test']['net_pnl']:.2f} ({single_wf['test']['n_trades']}t) | ${multi_wf['test']['net_pnl']:.2f} ({multi_wf['test']['n_trades']}t) |")
    md.append(f"| test bootstrap 95% CI | [${single_wf['test_bootstrap']['lower_95']:.2f}, ${single_wf['test_bootstrap']['upper_95']:.2f}] | [${multi_wf['test_bootstrap']['lower_95']:.2f}, ${multi_wf['test_bootstrap']['upper_95']:.2f}] |")
    md.append(f"| test P(net>0) | {single_wf['test_bootstrap']['p_positive']*100:.1f}% | {multi_wf['test_bootstrap']['p_positive']*100:.1f}% |")
    md.append("")
    md.append("## Per-city net P&L")
    md.append("")
    md.append("| city | single n | single net | multi n | multi net | Δ |")
    md.append("|---|---|---|---|---|---|")
    all_cities = sorted(set(single_cities) | set(multi_cities))
    for c in all_cities:
        s = single_cities.get(c, {"n": 0, "net": 0.0})
        m = single_cities.get(c, {"n": 0, "net": 0.0})  # placeholder
        m = multi_cities.get(c, {"n": 0, "net": 0.0})
        delta = m["net"] - s["net"]
        md.append(f"| {c} | {s['n']} | ${s['net']:.2f} | {m['n']} | ${m['net']:.2f} | ${delta:+.2f} |")
    md.append("")

    # Verdict
    md.append("## Verdict")
    md.append("")
    improvement = multi_sum["net_pnl"] - single_sum["net_pnl"]
    if multi_sum["net_pnl"] >= single_boot["lower_95"] and multi_boot["lower_95"] > 0:
        if improvement > 0:
            verdict = "ADOPT multi-source"
            reason = f"Net P&L improves by ${improvement:+.2f}, 95% CI strictly above $0, bootstrap P(net>0) = {multi_boot['p_positive']*100:.1f}%."
        else:
            verdict = "EQUIVALENT — keep single for simplicity"
            reason = f"Net P&L within CI of single (${improvement:+.2f}); not enough delta to justify added complexity."
    elif multi_boot["lower_95"] <= 0:
        verdict = "DO NOT ADOPT multi-source"
        reason = f"Multi-source 95% CI touches $0 (${multi_boot['lower_95']:.2f}); statistical significance weaker than single-source."
    else:
        verdict = "INCONCLUSIVE"
        reason = "Mixed signals across metrics."
    md.append(f"**{verdict}** — {reason}")
    md.append("")

    (OUT_DIR / "13_multi_source.md").write_text("\n".join(md))

    # CSV: per-strategy summary
    csv_rows = [
        {**single_sum, "strategy": "single_source_gfs_ens", "bootstrap_lower_95": single_boot["lower_95"], "bootstrap_upper_95": single_boot["upper_95"], "p_positive": single_boot["p_positive"]},
        {**multi_sum, "strategy": "multi_source_blend_equal", "bootstrap_lower_95": multi_boot["lower_95"], "bootstrap_upper_95": multi_boot["upper_95"], "p_positive": multi_boot["p_positive"]},
    ]
    fields = ["strategy", "n_trades", "win_rate", "gross_pnl", "net_pnl", "max_drawdown",
              "bootstrap_lower_95", "bootstrap_upper_95", "p_positive"]
    with (OUT_DIR / "13_multi_source.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"\nVERDICT: {verdict}")
    print(f"wrote 13_multi_source.md and 13_multi_source.csv")


if __name__ == "__main__":
    main()

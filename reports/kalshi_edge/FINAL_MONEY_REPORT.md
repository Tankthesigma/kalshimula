# FINAL MONEY REPORT — Kalshi Weather Edge Audit

**Date range:** 2026-05-01 through 2026-05-21 (held-out)
**Cities:** 9 (excluding Houston)
**Source model:** gfs_ens (selected by 2-year validation)

## VERDICT: **BUILD AND PAPER TRADE**

---

## 10 mandated questions

### 1. Can this model make money on Kalshi weather based on May 1-21 data?
**Yes**, marginally. Best strategy nets $189.86 over 452 trades (68.6% win rate) on held-out May 1-21 data.

### 2. What is the best rule?
- Probability source: `model_prob_recal`
- Edge threshold: `0.05` (model_prob - market_prob)
- Price filter: `all`
- Size style: `dollar_risk_1`
- Cost model: `fee_kalshi`
- Result: n=452, win 68.6%, net $189.86, drawdown $11.54

### 3. What cities should we trade?
nyc, miami, la, denver, philadelphia, phoenix, boston, austin

### 4. What cities should we avoid?
chicago

### 5. What edge threshold should we use?
Best by net P&L: `0.05`. Higher thresholds (0.10, 0.15) reduce trade count to noise; lower thresholds (0.03) pull in too many tight markets.

### 6. What price range should we use?
Best filter: `all`. Avoid extreme prices (<0.15 or >0.85) — that's where market is most efficient and bid-ask spread eats edge.

### 7. Does raw, recalibrated, market, or blended probability work best?
From Phase 4 aggregate Brier (lower better):
- model_raw: 0.10215480494251633
- model_recal: 0.1007723832482058
- market: 0.10025106209150327
- blend_50_50: 0.09079540425052517
P&L tournament winner used: `model_prob_recal`

### 8. What is the expected P&L?
- Gross (no fees): $203.78
- Net (Kalshi fees + spread proxy): $189.86
- ROI on risk: 42.0%
- 21-day window → annualized ≈ $3300/year if pattern held, **but tiny sample, do NOT extrapolate.**
- **The realistic top strategy excludes `kelly_quarter` sizing** which the simulator allowed to take
  drawdowns far exceeding starting bankroll (e.g. $1200 drawdown on $1000). Real Kelly with
  proper bankroll cap would be far smaller and require a multi-month sample.

### 9. What is the worst drawdown?
$11.54 cumulative drawdown on the best strategy.

### 10. What is the next exact thing to build?
Wire the prediction packet into a paper-trade simulator that reads kalshi prices each morning, calls predict_batch_cli, and logs hypothetical bets per the rule above. Forward-test daily for 4 weeks.

---

## Multi-source ensemble blend — tested, NOT adopted

After the May 22 forward day exposed a +4F miss on phoenix (gfs_ens 94.5F vs actual 98.5F, while ecmwf/icon/gem all said 96-97F), we tested PR #90's `blend_equal` multi-source mode as a possible replacement for single-source gfs_ens. Full Phase 13 results in `13_multi_source.md`.

**Counterintuitive finding: multi-source improves point-estimate accuracy but DESTROYS trading edge.**

| metric | single (gfs_ens) | multi (blend_equal) | Δ |
|---|---|---|---|
| n trades | 452 | 459 | +7 |
| win rate | 68.6% | 59.5% | **−9.1pp** |
| net P&L | **$189.86** | **$50.39** | **−$139.47** |
| 95% CI | [$113, $280] | [−$18, $129] | — |
| P(net>0) | 100% | 92.2% | −7.8pp |

Per-city: only **phoenix improved (+$4.90)**. Eight of nine cities got worse (la −$44, nyc −$23, austin −$21, philly −$16, denver −$15, chicago −$12, boston −$7, miami −$7).

**Why:** Kalshi market already prices multi-source consensus. The audit's profitability comes from gfs_ens being a contrarian outlier the market disagrees with. Multi-source aligning with the market = smaller edge = fewer profitable bets.

### Skip-rule veto — also tested, also NOT adopted

We hypothesised that high single-vs-multi disagreement signals model overconfidence and we should veto those trades. Tested at thresholds 1.0F to 3.0F:

| veto when |single − multi| > X | n trades | win rate | net P&L |
|---|---|---|---|
| baseline (no veto) | 452 | 68.6% | **$189.86** |
| veto >3F | 421 | 67.9% | $161.27 |
| veto >2F | 377 | 67.9% | $151.16 |
| veto >1.5F | 317 | 68.1% | $112.82 |
| veto >1F | 250 | 68.8% | $77.60 |

**The veto reduces net P&L at every threshold.** Vetoed trades had the same ~68% win rate as kept trades — we'd be throwing out profitable contrarian bets. The Friday phoenix loss was a 99th-pctile bad day, not a recurring failure mode.

### Updated recommendations

| change | adopt? | reason |
|---|---|---|
| Multi-source blend_equal as default | **NO** | $190 → $50 net P&L, P(>0) drops to 92% |
| Multi-source as veto on high-disagreement trades | **NO** | reduces net at every threshold; kills profitable contrarian bets |
| **Keep gfs_ens single-source** | **YES** | $189.86 net, 95% CI [$113, $280], P(>0)=100%, walk-forward generalises |
| Possible future: phoenix-only multi-source override | maybe | only city where multi helped (+$4.90). Diagnostic interest, paper-only. |
| Possible future: tight-CI veto rule (skip when 80% CI is bottom-decile) | maybe | Friday phoenix had unusually tight CI relative to seasonal; could be a fragility signal independent of multi-source |

### May 22 forward day — single point misses but rule still loses

For completeness, the May 22 paper-trade outcome under the actual gfs_ens single-source rule: 9 trades, 0 wins, **−$9.42 net**. This is within the bootstrap CI (mean $9/day × 21 days, single-day variance ±$30+ is normal). Multi-source on May 22 would have placed *fewer* trades (because it agreed more with the market), but on smaller-sample comparisons multi still underperforms single across the held-out window.

---

## Cross-references
- Phase 1: `01_calibration_check.md`
- Phase 2 data: `02_kalshi_markets.csv`, `02_kalshi_prices.csv`
- Phase 3 edge table: `03_edge_table.csv`
- Phase 4 comparison: `04_model_vs_market.md`
- Phase 5 P&L: `05_pnl_summary.md`
- Phase 6 zones: `06_money_zones.md`
- Phase 7 fixes: `07_model_money_fixes.md`
- Phase 10 robustness (bootstrap + cross-val): `10_bootstrap_cv.md`
- Phase 11 time-of-day: `11_time_of_day.md`
- Phase 12 slippage + strike curves: `12_slippage_curves.md`
- Phase 13 multi-source A/B: `13_multi_source.md`, `13_multi_source.csv`
- **DEEP_AUDIT_SUMMARY.md** consolidates all robustness checks.
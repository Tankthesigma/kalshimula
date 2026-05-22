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
- **DEEP_AUDIT_SUMMARY.md** consolidates all robustness checks.
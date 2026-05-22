# DEEP AUDIT SUMMARY — Kalshi Weather Edge

Built on top of `FINAL_MONEY_REPORT.md`. Adds robustness checks the original report didn't cover.

**Window:** 2026-05-01 through 2026-05-21 (held-out from training)
**Cities:** 9 (excludes Houston per user)
**Baseline strategy:** edge≥0.05 (recal model − market), `dollar_risk_1` sizing, Kalshi fee model

---

## 1. Statistical significance

### Full-window bootstrap (Phase 10)

| metric | value |
|---|---|
| n trades | 452 |
| n resamples | 1000 |
| mean net | $190.56 |
| 95% CI | **[$113.05, $280.24]** |
| P(net > $0) | **100.0%** |
| P(net > $100) | 98.5% |
| P(net > $150) | 83.1% |

**Read**: $189.86 net is not noise. Every single bootstrap resample produced net > 0.

### Walk-forward cross-validation (Phase 10)

| split | n trades | win rate | net P&L |
|---|---|---|---|
| Train (May 1-14) | 372 | 68.5% | $155.24 |
| Test  (May 15-21) | 80 | 68.8% | **$34.63** |

Test-set bootstrap (n=80, 1000 resamples): 95% CI **[$8.52, $64.83]**, P(net > $0) = **99.4%**.

**Read**: edge generalizes from first half to second half of May.

### Per-city train→test stability

7 of 9 cities profitable in BOTH halves: boston, denver, la, miami, nyc, philadelphia, phoenix.
Austin and Chicago profitable in train, negative in test — small-sample / possibly regression.

---

## 2. Execution friction (Phase 12)

| slippage on entry | net P&L | Δ vs 0c |
|---|---|---|
| 0c (perfect fill) | $185.59 | — |
| 1c | $159.18 | −$26 |
| 2c | $136.69 | −$49 |
| 3c | $117.07 | −$68 |
| 5c | $83.98 | −$102 |

**Read**: edge survives realistic 2-3c slippage. Still positive at 5c.
Per cent of slippage = ~$25-30 lost on 21-day P&L.

---

## 3. Where the edge actually lives (Phase 12 strike-distance curve)

| strike − actual (F) | n events | hit_rate_yes | trades after edge filter | win rate | net P&L |
|---|---|---|---|---|---|
| −10 to −7 | 20 | 0.00 | 0 | — | $0 |
| −7 to −4 | 91 | 0.00 | 18 | 100% | $4 |
| −4 to −1 | 227 | 0.03 | 139 | 84.2% | **$40** |
| −1 to +1 | 164 | 0.83 | 133 | 33.1% | **$95** |
| +1 to +4 | 252 | 0.04 | 146 | 78.1% | **$46** |
| +4 to +7 | 126 | 0.00 | 20 | 85.0% | −$0.54 |
| +7 to +10 | 34 | 0.03 | 0 | — | $0 |

**Read**: The $185 of P&L is concentrated in the −4 to +4 F band around the actual high.
- The middle bucket (−1 to +1) has hit_rate_yes ≈ 83% but our win rate is only 33% — because we're BUYING NO contracts that pay big when they hit. Low win rate, large average payoff per win.
- Outside ±7F is no-edge territory: market and model agree (correctly predicting near 0% yes), nothing to bet on.

The model has SYSTEMATICALLY lower probability than market across most buckets (negative `avg_edge`), meaning we usually SELL YES / BUY NO. The market over-prices yes contracts near the predicted center; we exploit that.

---

## 4. Caveats that still apply

1. **21 days is small.** 95% CI on full window is [$113, $280] — but this assumes trades are i.i.d., which they aren't (multiple cities share weather correlation on a given day). True CI is wider.

2. **All probability comparisons used a single 12:00 UTC snapshot.** Real market price varies through the day. Phase 11 (time-of-day study) is in progress.

3. **Bid/ask spread is approximated.** Phase 11 (in progress) will re-run with real bid/ask midpoints instead of `last_close`.

4. **No real trade execution simulation.** This is hypothetical fills at snapshot price. Even small adverse selection in real fills could reduce edge by 1-3c per trade.

5. **Kalshi may change pricing dynamics.** Markets that were inefficient in May 2026 may become efficient as more participants enter. The 4-week forward test recommended in `FINAL_MONEY_REPORT.md` would catch regime drift.

6. **No fee changes modeled.** Kalshi may adjust fee schedule; our 0.07 × p × (1−p) approximation is a current best-guess.

---

## 5. Updated recommendations

| recommendation | confidence | basis |
|---|---|---|
| Trade weather markets at all | **high** | bootstrap 100% > 0 across 452 trades; walk-forward also positive |
| Trade boston, philly, miami, la, denver, nyc, phoenix | **high** | positive in both train and test halves |
| Skip austin and chicago | medium | profitable in train, negative in test — could be small-sample, but caution |
| Use recal model, edge ≥ 0.05, all prices | high | best Phase 5 strategy in tournament |
| Use `dollar_risk_1` sizing | high | smooth P&L, smallest drawdown |
| Use Kalshi fee-adjusted P&L for go/no-go decisions | high | gross is misleading |
| Avoid `kelly_quarter` sizing | high | simulator allowed drawdown > bankroll; unrealistic |
| Restrict to strike − actual within ±5F | medium | edge concentrates there; outside is dead zone |
| Paper-trade for 4 weeks before real money | **mandatory** | sample size + regime drift caveats |

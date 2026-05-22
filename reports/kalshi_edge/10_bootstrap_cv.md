# Phase 10 — Bootstrap CI + Cross-validation

Two robustness checks on the baseline strategy (recal prob, edge>=0.05, prices 0.15-0.85,
dollar_risk_1 sizing, Kalshi fee model, 9 cities excluding Houston).

## Bootstrap on baseline trades

- N trades: 452
- N resamples: 1000
- Mean net P&L: $190.56
- Median net P&L: $190.66
- 95% CI: [$113.05, $280.24]
- P(net > $0): **100.0%**
- P(net > $50): 100.0%
- P(net > $100): 98.5%
- P(net > $150): 83.1%

**Verdict**: 95% CI strictly above $0 → P&L is statistically significantly positive.

Caveat: bootstrapping per-trade-net assumes trades are i.i.d. In reality there's
autocorrelation — a single bad weather event can hit multiple cities on the same day.
True CI is somewhat wider than what this naive bootstrap shows.

## Walk-forward cross-validation

Train (May 1-14): n=372, win 68.5%, net $155.24
Test  (May 15-21): n=80, win 68.8%, net $34.63
Test on train-profitable cities only: n=80, net $34.63

Train-profitable cities: ['austin', 'boston', 'chicago', 'denver', 'la', 'miami', 'nyc', 'philadelphia', 'phoenix']

### Per-city train vs test net P&L

| city | train net | test net | both positive? |
|---|---|---|---|
| austin | $27.17 | $-3.44 | no |
| boston | $18.13 | $14.24 | yes |
| chicago | $12.37 | $-2.06 | no |
| denver | $6.73 | $0.36 | yes |
| la | $16.21 | $4.45 | yes |
| miami | $17.83 | $5.13 | yes |
| nyc | $14.10 | $2.45 | yes |
| philadelphia | $17.45 | $12.81 | yes |
| phoenix | $25.25 | $0.70 | yes |

### Test-set bootstrap (n=80)
- Mean net: $34.63
- 95% CI: **[$8.52, $64.83]** (strictly above $0)
- P(test net > 0) = 99.4%

### Verdict
Test set still positive ($34.63) with bootstrap-CI strictly above zero (99.4% P>0).
Edge generalizes from first half to second half of May.

Caveats:
- Austin and Chicago profitable in train but negative in test — likely small-sample noise or regression
- Both halves are short (14 / 7 days). True walk-forward needs many months.
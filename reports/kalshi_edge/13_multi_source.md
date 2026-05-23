# Phase 13 — Multi-source ensemble blend A/B

**Setup**: same trading rule (edge≥0.05, model_prob_recal, dollar_risk_1, Kalshi fee, 9 cities ex-houston, all prices, May 1-21 2026 held-out).

Single-source uses `gfs_ens` only (the audit's selected source).
Multi-source pools members across `gfs_ens, ecmwf_ens, icon_ens, gem_ens, aifs` with equal weights, then re-applies the same residual CDF + recalibration table to compute probabilities.
Implementation: PR #90 (`multi_source_mode=blend_equal`).

## Headline A/B

| metric | single-source | multi-source | Δ |
|---|---|---|---|
| n trades | 452 | 459 | +7 |
| win rate | 68.6% | 59.5% | -9.1pp |
| gross P&L | $203.78 | $65.17 | $-138.61 |
| **net P&L** | **$189.86** | **$50.39** | **$-139.47** |
| max drawdown | $11.54 | $35.39 | — |

## Bootstrap CI (1000 resamples)

| | single-source | multi-source |
|---|---|---|
| mean net | $190.56 | $50.09 |
| 95% CI | [$113.05, $280.24] | [$-17.74, $128.89] |
| P(net>0) | 100.0% | 92.2% |
| P(net>$100) | 98.5% | 7.4% |

## Walk-forward split

| split | single-source | multi-source |
|---|---|---|
| train (5/1-5/14) net | $155.24 (372t) | $37.27 (383t) |
| test (5/15-5/21) net | $34.63 (80t) | $13.13 (76t) |
| test bootstrap 95% CI | [$8.52, $64.83] | [$-7.65, $37.17] |
| test P(net>0) | 99.4% | 87.4% |

## Per-city net P&L

| city | single n | single net | multi n | multi net | Δ |
|---|---|---|---|---|---|
| austin | 47 | $23.72 | 46 | $2.46 | $-21.27 |
| boston | 52 | $32.36 | 48 | $25.08 | $-7.28 |
| chicago | 57 | $10.30 | 57 | $-2.16 | $-12.46 |
| denver | 55 | $7.09 | 58 | $-7.63 | $-14.72 |
| la | 47 | $20.66 | 56 | $-23.03 | $-43.69 |
| miami | 57 | $22.96 | 51 | $16.30 | $-6.67 |
| nyc | 48 | $16.55 | 55 | $-5.98 | $-22.53 |
| philadelphia | 44 | $30.26 | 42 | $14.50 | $-15.75 |
| phoenix | 45 | $25.95 | 46 | $30.86 | $+4.90 |

## Verdict

**DO NOT ADOPT multi-source** — Multi-source 95% CI touches $0 ($-17.74); statistical significance weaker than single-source.

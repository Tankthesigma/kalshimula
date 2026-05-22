# Phase 5 — P&L Simulation

Strategies evaluated: 540.

## Top 10 by net P&L (any strategy)

| edge | price | prob | size | cost | n | win% | edge_avg | gross | net | drawdown |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.03 | all | model_prob_recal | kelly_quarter | gross | 521 | 69.3% | 20.7pp | $27161.52 | $27161.52 | $1159.62 |
| 0.05 | all | model_prob_recal | kelly_quarter | gross | 456 | 68.0% | 23.0pp | $26880.36 | $26880.36 | $1172.59 |
| 0.08 | all | model_prob_recal | kelly_quarter | gross | 384 | 67.2% | 26.2pp | $26233.77 | $26233.77 | $1220.80 |
| 0.10 | all | model_prob_recal | kelly_quarter | gross | 346 | 67.1% | 28.1pp | $25540.72 | $25540.72 | $1220.80 |
| 0.03 | all | model_prob_recal | kelly_quarter | fee_kalshi | 521 | 69.3% | 20.7pp | $27161.52 | $25277.07 | $1238.37 |
| 0.05 | all | model_prob_recal | kelly_quarter | fee_kalshi | 456 | 68.0% | 23.0pp | $26880.36 | $25041.63 | $1250.48 |
| 0.03 | all | model_prob_raw | kelly_quarter | gross | 553 | 67.1% | 19.0pp | $24766.07 | $24766.07 | $1023.13 |
| 0.15 | all | model_prob_recal | kelly_quarter | gross | 271 | 67.5% | 32.3pp | $24635.76 | $24635.76 | $1274.13 |
| 0.08 | all | model_prob_recal | kelly_quarter | fee_kalshi | 384 | 67.2% | 26.2pp | $26233.77 | $24475.22 | $1295.54 |
| 0.05 | all | model_prob_raw | kelly_quarter | gross | 474 | 65.8% | 21.6pp | $24340.07 | $24340.07 | $1033.55 |

## Bottom 5 by net P&L (worst losers)

| edge | price | prob | size | cost | n | win% | net |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.30-0.70 | model_prob_raw | flat_1 | fee_kalshi | 207 | 59.4% | $8.50 |
| 0.05 | 0.30-0.70 | model_prob_raw | flat_1 | spread_2c | 207 | 59.4% | $7.82 |
| 0.03 | 0.30-0.70 | model_prob_raw | flat_1 | gross | 215 | 57.2% | $7.64 |
| 0.03 | 0.30-0.70 | model_prob_raw | flat_1 | fee_kalshi | 215 | 57.2% | $4.05 |
| 0.03 | 0.30-0.70 | model_prob_raw | flat_1 | spread_2c | 215 | 57.2% | $3.34 |

## Most robust positive strategies

Strategies with net > 0 AND n_trades >= 20: 540

| edge | price | prob | size | cost | n | win% | net | drawdown |
|---|---|---|---|---|---|---|---|---|
| 0.15 | all | blend_50_50 | dollar_risk_1 | gross | 133 | 63.9% | $100.28 | $8.13 |
| 0.15 | all | blend_50_50 | dollar_risk_1 | fee_kalshi | 133 | 63.9% | $95.07 | $8.57 |
| 0.15 | all | blend_50_50 | kelly_quarter | gross | 133 | 63.9% | $9266.87 | $790.01 |
| 0.15 | all | blend_50_50 | dollar_risk_1 | spread_2c | 133 | 63.9% | $86.69 | $10.22 |
| 0.15 | all | blend_50_50 | kelly_quarter | fee_kalshi | 133 | 63.9% | $8766.52 | $830.34 |
| 0.10 | all | blend_50_50 | dollar_risk_1 | gross | 221 | 63.8% | $139.54 | $8.44 |
| 0.15 | all | blend_50_50 | kelly_quarter | spread_2c | 133 | 63.9% | $8135.07 | $946.34 |
| 0.15 | 0.15-0.85 | blend_50_50 | dollar_risk_1 | gross | 117 | 69.2% | $69.72 | $5.13 |
| 0.10 | all | blend_50_50 | dollar_risk_1 | fee_kalshi | 221 | 63.8% | $131.37 | $9.07 |
| 0.08 | all | blend_50_50 | dollar_risk_1 | gross | 257 | 66.5% | $147.89 | $7.74 |
| 0.15 | all | model_prob_recal | dollar_risk_1 | gross | 271 | 67.5% | $153.11 | $8.01 |
| 0.15 | 0.15-0.85 | blend_50_50 | dollar_risk_1 | fee_kalshi | 117 | 69.2% | $65.57 | $5.38 |
| 0.10 | all | blend_50_50 | kelly_quarter | gross | 221 | 63.8% | $11339.68 | $673.44 |
| 0.15 | 0.15-0.85 | blend_50_50 | dollar_risk_1 | spread_2c | 117 | 69.2% | $64.32 | $5.47 |
| 0.08 | all | blend_50_50 | dollar_risk_1 | fee_kalshi | 257 | 66.5% | $138.91 | $8.40 |

## Best 10 trades under top strategy (model_prob_recal, edge≥0.03, all, kelly_quarter, gross)

| city | date | ticker | side | model_p | market_p | edge | price_paid | net |
|---|---|---|---|---|---|---|---|---|
| la | 2026-05-11 | KXHIGHLAX-26MAY11-T69 | yes | 0.987 | 0.085 | +0.902 | $0.085 | $2651.770 |
| boston | 2026-05-03 | KXHIGHTBOS-26MAY03-T56 | yes | 0.996 | 0.095 | +0.901 | $0.095 | $2370.548 |
| nyc | 2026-05-11 | KXHIGHNY-26MAY11-T62 | yes | 0.589 | 0.085 | +0.504 | $0.085 | $1483.227 |
| phoenix | 2026-05-05 | KXHIGHTPHX-26MAY05-B80.5 | yes | 0.420 | 0.080 | +0.340 | $0.080 | $1062.639 |
| miami | 2026-05-13 | KXHIGHMIA-26MAY13-T92 | yes | 0.999 | 0.195 | +0.804 | $0.195 | $1030.209 |
| austin | 2026-05-04 | KXHIGHAUS-26MAY04-B82.5 | yes | 0.275 | 0.055 | +0.220 | $0.055 | $998.194 |
| miami | 2026-05-14 | KXHIGHMIA-26MAY14-T93 | yes | 0.999 | 0.265 | +0.734 | $0.265 | $692.041 |
| phoenix | 2026-05-04 | KXHIGHTPHX-26MAY04-T81 | yes | 0.542 | 0.155 | +0.387 | $0.155 | $623.537 |
| boston | 2026-05-17 | KXHIGHTBOS-26MAY17-T88 | yes | 0.804 | 0.235 | +0.569 | $0.235 | $604.793 |
| miami | 2026-05-04 | KXHIGHMIA-26MAY04-B81.5 | yes | 0.510 | 0.150 | +0.360 | $0.150 | $599.607 |

## Worst 10 trades under top strategy

| city | date | ticker | side | model_p | market_p | edge | price_paid | net |
|---|---|---|---|---|---|---|---|---|
| denver | 2026-05-05 | KXHIGHDEN-26MAY05-T48 | no | 0.000 | 0.980 | -0.980 | $0.020 | $-249.877 |
| denver | 2026-05-13 | KXHIGHDEN-26MAY13-B88.5 | no | 0.000 | 0.440 | -0.440 | $0.560 | $-249.725 |
| chicago | 2026-05-14 | KXHIGHCHI-26MAY14-B64.5 | no | 0.001 | 0.455 | -0.454 | $0.545 | $-249.610 |
| miami | 2026-05-02 | KXHIGHMIA-26MAY02-T94 | yes | 0.999 | 0.170 | +0.829 | $0.170 | $-249.567 |
| chicago | 2026-05-11 | KXHIGHCHI-26MAY11-B55.5 | no | 0.001 | 0.270 | -0.269 | $0.730 | $-249.343 |
| chicago | 2026-05-10 | KXHIGHCHI-26MAY10-B61.5 | no | 0.001 | 0.195 | -0.194 | $0.805 | $-249.091 |
| phoenix | 2026-05-11 | KXHIGHTPHX-26MAY11-B108.5 | no | 0.004 | 0.655 | -0.651 | $0.345 | $-248.310 |
| phoenix | 2026-05-10 | KXHIGHTPHX-26MAY10-B105.5 | no | 0.004 | 0.560 | -0.556 | $0.440 | $-248.023 |
| chicago | 2026-05-12 | KXHIGHCHI-26MAY12-B73.5 | no | 0.001 | 0.085 | -0.084 | $0.915 | $-247.914 |
| miami | 2026-05-02 | KXHIGHMIA-26MAY02-B93.5 | no | 0.008 | 0.485 | -0.477 | $0.515 | $-246.101 |
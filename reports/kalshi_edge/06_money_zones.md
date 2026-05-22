# Phase 6 — Money Zones

Baseline strategy used: edge>=0.05, prices 0.15-0.85, recal prob, flat 1 contract, Kalshi fee model.
Total: n=323, net=$39.61, win_rate=69.3%

## Per-city

| city | n | win% | net P&L | verdict |
|---|---|---|---|---|
| miami | 40 | 72.5% | $8.34 | trade |
| philadelphia | 34 | 82.4% | $7.63 | trade |
| boston | 36 | 72.2% | $7.59 | trade |
| la | 36 | 75.0% | $6.68 | trade |
| austin | 34 | 67.6% | $4.05 | trade |
| nyc | 34 | 64.7% | $3.80 | trade |
| phoenix | 30 | 60.0% | $1.82 | trade |
| denver | 36 | 66.7% | $0.63 | trade |
| chicago | 43 | 62.8% | $-0.92 | avoid |

## Per contract type

| contract_type | n | win% | net | verdict |
|---|---|---|---|---|
| bin_between | 289 | 67.8% | $27.87 | trade |
| threshold_greater | 14 | 85.7% | $6.08 | trade |
| threshold_less | 20 | 80.0% | $5.66 | trade |

## Per price bucket

| price | n | win% | net | verdict |
|---|---|---|---|---|
| 0.15-0.30 | 129 | 81.4% | $23.44 | trade |
| 0.30-0.45 | 93 | 74.2% | $15.29 | trade |
| 0.45-0.55 | 60 | 53.3% | $0.71 | trade |
| 0.55-0.70 | 37 | 40.5% | $-0.32 | needs more data |
| 0.70-0.85 | 4 | 75.0% | $0.49 | needs more data |

## Final recommendations

| category | subgroup | trade / avoid / needs more data | reason | n | net | confidence |
|---|---|---|---|---|---|---|
| city | nyc | trade | positive net $3.80, win_rate 65% | 34 | $3.80 | high |
| city | chicago | avoid | negative net $-0.92, win_rate 63% | 43 | $-0.92 | high |
| city | miami | trade | positive net $8.34, win_rate 72% | 40 | $8.34 | high |
| city | la | trade | positive net $6.68, win_rate 75% | 36 | $6.68 | high |
| city | denver | trade | positive net $0.63, win_rate 67% | 36 | $0.63 | high |
| city | philadelphia | trade | positive net $7.63, win_rate 82% | 34 | $7.63 | high |
| city | phoenix | trade | positive net $1.82, win_rate 60% | 30 | $1.82 | high |
| city | boston | trade | positive net $7.59, win_rate 72% | 36 | $7.59 | high |
| city | austin | trade | positive net $4.05, win_rate 68% | 34 | $4.05 | high |
| contract_type | bin_between | trade | positive net $27.87, win_rate 68% | 289 | $27.87 | high |
| contract_type | threshold_less | trade | positive net $5.66, win_rate 80% | 20 | $5.66 | medium |
| contract_type | threshold_greater | trade | positive net $6.08, win_rate 86% | 14 | $6.08 | low |
| price_bucket | 0.30-0.45 | trade | positive net $15.29, win_rate 74% | 93 | $15.29 | high |
| price_bucket | 0.15-0.30 | trade | positive net $23.44, win_rate 81% | 129 | $23.44 | high |
| price_bucket | 0.45-0.55 | trade | positive net $0.71, win_rate 53% | 60 | $0.71 | high |
| price_bucket | 0.55-0.70 | needs more data | negative net $-0.32, win_rate 41% | 37 | $-0.32 | high |
| price_bucket | 0.70-0.85 | needs more data | only 4 trades | 4 | $0.49 | low |
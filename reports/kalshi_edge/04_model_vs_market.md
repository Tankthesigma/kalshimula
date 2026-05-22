# Phase 4 — Model vs Market

Brier / log-loss / ECE on comparable edge events. Lower is better.

## Aggregate

| metric | model raw | model recal | market | blend 75m/25mk | blend 50/50 | blend 25m/75mk |
|---|---|---|---|---|---|---|
| Brier  | 0.1022 | 0.1008 | 0.1003 | 0.0934 | 0.0908 | 0.0931 |
| LogLoss| 0.3123 | 0.3579 | 0.3078 | 0.2861 | 0.2801 | 0.2876 |
| ECE    | 0.0660 | 0.0530 | 0.0106 | 0.0542 | 0.0528 | 0.0403 |

(n = 918)

Full breakdown in `04_model_vs_market.csv`.

## Per-city Brier (recal vs market)

| city | n | model_recal | market | blend 50/50 |
|---|---|---|---|---|
| austin | 102 | 0.0935 | 0.0844 | 0.0796 |
| boston | 102 | 0.0881 | 0.1164 | 0.0921 |
| chicago | 102 | 0.1397 | 0.1096 | 0.1160 |
| denver | 102 | 0.1273 | 0.0898 | 0.0981 |
| la | 102 | 0.0793 | 0.0871 | 0.0717 |
| miami | 102 | 0.0857 | 0.1088 | 0.0825 |
| nyc | 102 | 0.1015 | 0.0985 | 0.0905 |
| philadelphia | 102 | 0.0900 | 0.1073 | 0.0930 |
| phoenix | 102 | 0.1018 | 0.1004 | 0.0936 |

## Per price bucket

| price bucket | n | model_recal | market | blend 50/50 |
|---|---|---|---|---|
| 0.05-0.15 | 157 | 0.0788 | 0.0972 | 0.0814 |
| 0.15-0.30 | 136 | 0.1309 | 0.1737 | 0.1384 |
| 0.30-0.45 | 119 | 0.2111 | 0.2300 | 0.2015 |
| 0.45-0.55 | 63 | 0.3226 | 0.2532 | 0.2615 |
| 0.55-0.70 | 38 | 0.3286 | 0.2459 | 0.2571 |
| 0.70-0.85 | 4 | 0.1982 | 0.0738 | 0.0995 |
| 0.85-0.95 | 1 | 0.5262 | 0.0121 | 0.1745 |

## Verdict

- **Best aggregate Brier**: `blend_50_50` at 0.0908
- model_recal Brier vs market Brier: 0.1008 vs 0.1003
- Market beats model on aggregate Brier — blending or filters needed.
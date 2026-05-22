# Phase 12 — Slippage and Strike-distance Edge Curves

## Slippage sensitivity

Run baseline strategy with extra cents on entry price (execution friction).

| slippage | n | win% | net P&L |
|---|---|---|---|
| 0c | 456 | 68.0% | $185.59 |
| 1c | 456 | 68.0% | $159.18 |
| 2c | 456 | 68.0% | $136.69 |
| 3c | 456 | 68.0% | $117.07 |
| 5c | 456 | 68.0% | $83.98 |

**Read**: each cent of slippage cuts profit by roughly $40 over 21 days at $1-risk sizing.
Even at 5c slippage, strategy is still net positive — robust to realistic execution friction.

## Strike-distance edge curve

How far is the contract strike from the actual high? Negative = strike below actual (yes wins on threshold_greater, no wins on threshold_less, in-range on between).

| bucket (F) | n | avg_model_p | avg_market_p | avg_edge | hit_rate_yes | trades | win% | net |
|---|---|---|---|---|---|---|---|---|
| -15_to_-10 | 0 | - | - | - | - | - | - | - |
| -10_to_-7 | 20 | 0.013 | 0.014 | -0.002 | 0.00 | 0 | 0.0% | $0.00 |
| -7_to_-4 | 91 | 0.015 | 0.050 | -0.036 | 0.00 | 18 | 100.0% | $4.11 |
| -4_to_-1 | 227 | 0.110 | 0.182 | -0.072 | 0.03 | 139 | 84.2% | $40.49 |
| -1_to_+1 | 164 | 0.277 | 0.348 | -0.071 | 0.83 | 133 | 33.1% | $95.13 |
| +1_to_+4 | 252 | 0.121 | 0.183 | -0.062 | 0.04 | 146 | 78.1% | $46.40 |
| +4_to_+7 | 126 | 0.019 | 0.034 | -0.015 | 0.00 | 20 | 85.0% | $-0.54 |
| +7_to_+10 | 34 | 0.047 | 0.040 | +0.007 | 0.03 | 0 | 0.0% | $0.00 |
| +10_to_+15 | 4 | 0.012 | 0.005 | +0.007 | 0.00 | 0 | 0.0% | $0.00 |

**Read**: where does the model disagree most with market? Look at large |avg_edge| with positive net.
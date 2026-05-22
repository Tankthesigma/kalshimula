# Phase 1 — Calibration Check (Held-Out May 1-21, 2026)

## Verdict

**Calibration is HONEST.** The Brier mismatch from the earlier overnight run was apples-to-oranges (3 vs 7 offsets), not a real divergence.

| metric | 7-offset held-out | backtest claim | verdict |
|---|---|---|---|
| Brier (recal) | **0.0555** | 0.0568 | match (within 2%) |
| ECE (recal) | **0.0182** | 0.0095 | 2× higher but small absolute |

## Detail

- Held-out window: 2026-05-01 through 2026-05-21 (model never saw these dates)
- Source: `gfs_ens` (the production single-source policy)
- Cities included: all 10 (houston kept in this phase for diagnostic, dropped in trading phase per user)
- Calibration pipeline applied: bias correction → empirical threshold prob → bucketed recalibration with global fallback

### 3-offset (Brier looks bad but isn't)

- n = 507 events (169 (city,day) × 3 offsets)
- Brier raw 0.1261 → recal 0.1253
- ECE  raw 0.0550 → recal 0.0484

The ±2/0 band targets the HARDEST predictions (near 50%). Aggregate Brier of 0.125 is not a model failure — it's measuring only the difficult middle.

### 7-offset (matches backtest claims)

- n = 1183 events (169 (city,day) × 7 offsets)
- Brier raw 0.0557 → recal 0.0555
- ECE  raw 0.0231 → recal 0.0182

The ±6/±4 offsets are very easy (predictions ~0% or ~100% are almost always correct), pulling aggregate Brier down. This is exactly how the backtest's 0.0568 was computed too.

### Per-offset Brier (recalibrated)

| offset | n | Brier recal | comment |
|---|---|---|---|
| -6 | 169 | 0.0002 | trivial (almost always yes) |
| -4 | 169 | 0.0061 | very easy |
| -2 | 169 | 0.0732 | meaningful uncertainty |
|  0 | 169 | **0.2382** | hardest (near 50%) |
| +2 | 169 | 0.0646 | meaningful uncertainty |
| +4 | 169 | 0.0062 | very easy |
| +6 | 169 | 0.0003 | trivial (almost always no) |

### Per-city Brier (7-offset, recalibrated)

| city | Brier recal | ECE recal | comment |
|---|---|---|---|
| philadelphia | 0.0295 | 0.0444 | tightest |
| phoenix | 0.0346 | 0.0230 | excellent |
| la | 0.0506 | 0.0432 | good |
| chicago | 0.0535 | 0.0641 | good |
| boston | 0.0586 | 0.0357 | good |
| denver | 0.0593 | 0.0341 | good |
| austin | 0.0571 | 0.0499 | good |
| miami | 0.0579 | 0.0252 | good |
| nyc | 0.0620 | 0.0788 | known +1.24F bias issue |
| houston | 0.0947 | 0.0688 | worst (user said skip Houston) |

## Implications for trading phases

1. The probability model is honest enough to use as the "model" side of model-vs-market edge.
2. **Edge will live near offset 0** where uncertainty is real. ±4/±6 offsets are unlikely to find market mispricing (market also gets these easy ones right).
3. **NYC has known over-prediction bias** (+1.24F on point, ECE 0.079). Worth bias-patching specifically for NYC trades.
4. Houston confirmed weakest probability calibration; user already excluded.

Inputs: `outputs/may1_21_combined.csv` (1353 rows, 169 gfs_ens events).
Per-row metrics: `reports/kalshi_edge/01_calibration_check.csv`.

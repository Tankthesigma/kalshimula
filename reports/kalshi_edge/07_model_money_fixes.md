# Phase 7 — Money Fixes

Baseline strategy: recal prob, edge≥0.05, prices 0.15-0.85, flat-1 contract, Kalshi fee model, 9 cities excl. Houston.

**Baseline result**: n=323, net=$39.61, brier=0.1008, win_rate=69.3%

## Candidate fixes

| fix | n | win% | net | Δnet | brier | Δbrier | overfit | reason |
|---|---|---|---|---|---|---|---|---|
| nyc_bias_patch_-5pp | 322 | 69.6% | $39.51 | $-0.10 | 0.1013 | +0.0005 | medium | NYC over-predicts by ~1.24F historically; -5pp shift |
| drop_nyc | 289 | 69.9% | $35.81 | $-3.80 | 0.1007 | -0.0001 | low | NYC has worst per-city Brier; remove from trading universe |
| drop_chicago_denver | 244 | 70.9% | $39.90 | $+0.30 | 0.0914 | -0.0093 | medium | Chicago and Denver have weakest per-city Brier |
| threshold_only | 34 | 82.4% | $11.74 | $-27.87 | 0.0289 | -0.0718 | low | Skip bin contracts (narrower, more market-pricing-error opportunity) |
| price_filter_0.30-0.70 | 190 | 61.1% | $15.68 | $-23.93 | 0.1008 | +0.0000 | low | Skip near-extreme prices where market is most efficient |
| edge_threshold_0.03 | 333 | 68.2% | $36.45 | $-3.16 | 0.1008 | +0.0000 | low | Edge threshold tuned to 0.03 |
| edge_threshold_0.08 | 302 | 70.2% | $39.88 | $+0.28 | 0.1008 | +0.0000 | low | Edge threshold tuned to 0.08 |
| edge_threshold_0.10 | 283 | 70.7% | $36.68 | $-2.93 | 0.1008 | +0.0000 | low | Edge threshold tuned to 0.1 |
| edge_threshold_0.15 | 243 | 72.4% | $36.21 | $-3.40 | 0.1008 | +0.0000 | low | Edge threshold tuned to 0.15 |
| blend_50_50_prob | 283 | 70.7% | $36.68 | $-2.93 | 0.1008 | +0.0000 | low | Use 50/50 model-market blend probability |
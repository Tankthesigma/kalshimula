# Overnight Mainline Report

Generated from mainline-safe diagnostics only. No Kalshi API, no market prices, no orders, no trading code.

## Verdict

The model survives a leakage-safe walk-forward smoke/full run over the two-year source-breakout artifact. `gfs_ens` beats `openmeteo_naive` on the walk-forward policy leaderboard and also dominates the contrarian diagnostics, but this is still a model/source finding, not a trading signal.

## Does The Model Survive No-Leak Walk-Forward?

Yes for the tested policies and cities. The run used prior-window calibration only and produced 6,570 predictions plus 45,990 threshold events.

Policy leaderboard:

- gfs_ens: MAE 1.015, Brier 0.056, worst city boston (1.330), promoted combos 8
- openmeteo_naive: MAE 1.222, Brier 0.066, worst city la (1.760), promoted combos 0

## Best Policy

`gfs_ens` is the best policy among the two real policies run end-to-end overnight: average MAE 1.015 vs 1.222 for `openmeteo_naive`, with lower Brier/logloss and better stability.

## Best Cities For gfs_ens

- phoenix: gfs_ens WF MAE 0.766, Brier 0.042
- austin: gfs_ens WF MAE 0.780, Brier 0.046
- la: gfs_ens WF MAE 0.851, Brier 0.048
- miami: gfs_ens WF MAE 0.886, Brier 0.051
- philadelphia: gfs_ens WF MAE 0.983, Brier 0.057

## Dangerous Cities For gfs_ens

- boston: gfs_ens WF MAE 1.330, Brier 0.072
- nyc: gfs_ens WF MAE 1.246, Brier 0.068
- denver: gfs_ens WF MAE 1.194, Brier 0.060
- chicago: gfs_ens WF MAE 1.103, Brier 0.056
- philadelphia: gfs_ens WF MAE 0.983, Brier 0.057

## Source Contrarian Verdict

Source-contrarian diagnostics produced 48,190 daily source-delta rows, 70 city/source summary rows, and 27 promoted combos.

Top promoted combos:

- phoenix/gfs_ens: CI lower 0.960, rate 0.974, MAE delta -0.967, abs delta 1.015
- phoenix/hrrr: CI lower 0.960, rate 0.974, MAE delta -0.967, abs delta 1.015
- austin/gfs_ens: CI lower 0.861, rate 0.886, MAE delta -1.028, abs delta 1.269
- austin/hrrr: CI lower 0.861, rate 0.886, MAE delta -1.028, abs delta 1.269
- la/gfs_ens: CI lower 0.822, rate 0.849, MAE delta -1.134, abs delta 2.307
- la/hrrr: CI lower 0.822, rate 0.849, MAE delta -1.134, abs delta 2.307
- philadelphia/gfs_ens: CI lower 0.808, rate 0.837, MAE delta -0.649, abs delta 1.092
- philadelphia/hrrr: CI lower 0.808, rate 0.837, MAE delta -0.649, abs delta 1.092
- houston/gfs_ens: CI lower 0.789, rate 0.818, MAE delta -0.859, abs delta 1.217
- houston/hrrr: CI lower 0.789, rate 0.818, MAE delta -0.859, abs delta 1.217
- chicago/gfs_ens: CI lower 0.783, rate 0.813, MAE delta -0.793, abs delta 1.109
- chicago/hrrr: CI lower 0.783, rate 0.813, MAE delta -0.793, abs delta 1.109

## Consensus Accuracy

Consensus does not win overall in this artifact for gfs_ens: many promoted `gfs_ens` rows have negative MAE delta, meaning the individual source beat openmeteo_naive on weather error. That supports Bobby's hypothesis that the useful signal may be source disagreement, but main cannot infer market edge.

## Duplicate/Artifact Warning

gfs_ens and hrrr are numerically identical for every city in this artifact; treat hrrr as duplicate/source artifact until fetcher provenance is confirmed.

This is the biggest warning in the source diagnostics. Bobby should avoid treating gfs_ens and hrrr as independent signals until provenance is verified.

## Climate Feature Verdict

Climate features are available for all 10 cities, but this branch only audits features; it does not prove climate adjustment improves walk-forward forecasts. Recent warming anomaly is negative across the run summary, likely reflecting the current seasonal window rather than a deployable signal.

Climate/trend features were implemented as leakage-safe diagnostics only. Do not use them for strategy until a later walk-forward comparison shows improvement.

## What Bobby Should Test Privately

Import `reports/BOBBY_PRIVATE_AUDIT_BRIDGE.csv` and the generated source diagnostics from:

- `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/source_contrarian/source_contrarian_summary.csv`
- `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/source_contrarian/daily_source_deltas.csv`
- `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/source_contrarian/source_threshold_grid.csv`
- `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/walkforward/walkforward_city_source_summary.csv`
- `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/policy_leaderboard/policy_leaderboard.csv`

Private audit tests should compare gfs_ens, openmeteo_naive, blend_equal/blend_mae_90d if available, promoted-only filters, city filters, price filters, and edge thresholds. Keep Kalshi API private/off-main and paper-only.

## What Tanmay Should Manually Inspect Tomorrow

Read `reports/TOMORROW_MODEL_PACKET.md`. Treat high/medium rows as manual paper-check candidates only. Confirm source availability, market parsing, price timestamp, and Bobby's private paper-PnL result before trusting any market edge.

## Kill Criteria

Do not use the contrarian filter yet if Bobby finds that promoted combos do not improve paper PnL, if the signal disappears after fees/slippage, or if the apparent edge is driven by duplicate gfs/hrrr artifacts.

## Next Branch

Next mainline branch should either harden source provenance/deduping or extend walk-forward policy evaluation to include `blend_equal`, `blend_mae_90d`, and selected-source policy if runtime allows.

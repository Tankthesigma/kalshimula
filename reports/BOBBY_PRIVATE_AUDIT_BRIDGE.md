# Bobby Private Audit Bridge

Mainline diagnostics are ready. These are model-only files; no Kalshi API or market prices were used.

## Files To Import

- `reports/BOBBY_PRIVATE_AUDIT_BRIDGE.csv`
- `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/source_contrarian/source_contrarian_summary.csv`
- `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/source_contrarian/daily_source_deltas.csv`
- `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/source_contrarian/source_threshold_grid.csv`
- `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/walkforward/walkforward_city_source_summary.csv`
- `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/policy_leaderboard/policy_leaderboard.csv`
- `reports/TOMORROW_MODEL_PACKET.csv`

## Join Keys

- City/source summary: `city`, `source`
- Daily source deltas: `city`, `source`, `target_date`
- Tomorrow packet: `city`, `selected_source`

## Top Candidates

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

## Required Private Tests

- gfs_ens vs openmeteo_naive
- blend_equal and blend_mae_90d if private prediction files exist
- promoted-only contrarian filter
- city-only filters for top promoted combos
- threshold offsets near corrected point
- price filters: 0.15-0.85, 0.20-0.80, 0.30-0.70
- edge thresholds: 0.03, 0.05, 0.08, 0.10
- fee and +1c/+2c/+3c slippage stress
- day-block bootstrap

## Warnings

- This is descriptive model behavior, not a trading signal.
- gfs_ens and hrrr are numerically identical for every city in this artifact; treat hrrr as duplicate/source artifact until fetcher provenance is confirmed.
- Mainline diagnostics do not know whether Kalshi prices resemble consensus.
- Keep all Kalshi API work private/off-main, GET-only, paper-only.

## Kill Criteria

Reject the contrarian filter if it removes winners as often as losers, fails after fees/slippage, depends on stale prices, or is dominated by duplicate source artifacts.

# Mainline Audit - 2026-05-24

Scope: mainline GitHub code and weather artifacts only. No Kalshi API code, market prices, order books, private PnL labels, secrets, or trading/execution code were audited into main.

## Verdict

Mainline is structurally clean and test-passing. The data artifact is internally consistent, the no-leak boundaries for nowcast observations and NWS guidance are explicit, and the new packet modes are candidate-only with raw `gfs_ens` still the default reference.

One actionable model bug was found and fixed: the heat-regime correction thresholds for Phoenix and Miami were too high, so the candidate heat packet failed to fire in exactly the mild-hot regimes where recent cold misses showed up.

## Checks Run

- `ruff check .` - passed
- `pytest` - 678 passed before the heat-threshold patch
- Focused post-patch tests:
  - `tests/test_heat_regime_correction.py`
  - `tests/test_weather_desk_cli.py`
  - `tests/test_weather_desk_refresh_cli.py`
  - 9 passed
- Secret/market-code scan over `src/` and `tests/` found no Kalshi/Polymarket API calls, order-book access, wallet/portfolio/order/cancel/bank code, private keys, tokens, or API keys. Hits were limited to guardrail text, tests, and unrelated `cancel_futures` executor calls.

## Data Integrity

Artifact: `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/rows.csv`

- Rows: 55,490
- Date range: 2024-05-01 through 2026-04-30
- Cities: 10
- Duplicate `city,target_date,source` rows: 0
- Null `point_f`: 0
- Null `actual_high_f`: 0
- Full 730-day coverage for `gfs_ens`, `ecmwf_ens`, `gem_ens`, `graphcast`, `hrrr`, `icon_ens`, and `openmeteo_naive`
- `aifs` has 439 days per city, expected partial coverage

Important data warning: `hrrr` is byte-identical to `gfs_ens` across all 7,300 matched city/date rows. It is not an independent source and must not be double-counted in blends or source diversity metrics. Mainline export marks `hrrr` with `source_independence_score = 0.0`; Bobby's private lane also excludes it from independent blends.

## Station Rules

`config/station_rule_table.csv` has 20 rows:

- High markets: 8 high-confidence rows, 2 medium-confidence rows
- Low markets: 10 medium-confidence rows

High-market confidence is acceptable for the current high-temperature model. Low-market work must remain candidate-only until Bobby privately validates low-market station/window settlement, especially DST/LST overnight handling.

## Leakage Review

Reviewed boundaries:

- `nowcast_features`: uses only observations with `available_ts_utc <= as_of_ts`.
- `guidance.latest_guidance_as_of`: uses only guidance rows with `available_ts_utc <= as_of_ts`.
- `predictions_nowcast` manifests emit `no_leak_max_observation_ts`.
- `lone_outlier` uses same-day NWS guidance and consensus only. Historical NWS guidance is not backfilled, so this mode is forward-test only.
- `heat_corrected` uses a fixed residual table from May 2024-April 2026 rows. That is safe for May 2026 forward packets, but any backtest inside that historical window should label the correction in-sample unless the table is re-fit using train-only rows.

## Model Findings

### Fixed: Phoenix and Miami heat thresholds were too high

Historical residual diagnostic used `actual_high_f - gfs_ens point_f`.

Phoenix:

- At `gfs_point >= 95F`: n=303, mean residual `+1.85F`, cold-miss rate 97%
- At `gfs_point >= 100F`: n=246, mean residual `+1.82F`, cold-miss rate 98%

The bias is already present around 95F, so a 100F trigger missed live cases like a 95-96F Phoenix forecast.

Miami:

- At `gfs_point >= 85F`: n=338, mean residual `+0.80F`, cold-miss rate 72%
- At `gfs_point >= 89F`: n=202, mean residual `+0.42F`, cold-miss rate 63%
- At `gfs_point >= 90F`: n=151, mean residual `+0.25F`, cold-miss rate 55%

The useful Miami correction is a warm-regime guard around 85F, not a 89F+ extreme-heat trigger.

Patch:

- `phoenix`: threshold `100.0 -> 95.0`, correction stays `+1.9F`
- `miami`: threshold `89.0 -> 85.0`, correction stays `+0.8F`
- Added tests proving PHX 95.7 and MIA 86.3 now fire in the candidate packet.

Smoke result for May 25 after patch:

- CHI 80.9 -> 82.4
- MIA 86.5 -> 87.3
- AUS 90.6 -> 92.0
- HOU 91.8 -> 93.7
- PHX 95.1 -> 97.0

Raw remains the default. Heat-corrected remains candidate-only until private paper PnL validates it.

### Candidate modes are correctly separated

- `predictions_nowcast_raw`: default/reference.
- `predictions_nowcast_adjusted`: weather-state PMF truncation; Bobby found it can hurt PnL, so it remains experimental.
- `predictions_nowcast_lone_outlier`: forward-only candidate because live NWS guidance is needed.
- `predictions_nowcast_heat_corrected`: candidate packet only; now patched to fire in the actual PHX/MIA warm regimes.

## Private-Lane Reconciliation

Bobby's private-lane audit reported:

- High severity: 7:07am market book is often too thin; May 24 had only 11/54 two-sided bins and only Phoenix was actually tradeable at that snapshot.
- High severity: settlement bin conventions are load-bearing and need standing re-checks as settled rows accumulate.
- Medium severity: private fee math is slightly optimistic because Kalshi rounding-to-cent is not yet modeled.
- Medium severity: fee and slippage are not yet combined into one conservative worst-case column.
- Verified: private lane is GET-only, secrets are clean, heat-corrected is candidate-only, and paper-PnL math selftests pass.

Mainline response:

- Do not promote any candidate packet from accuracy alone.
- Bobby should continue scoring raw, adjusted, lone_outlier, and heat_corrected forward.
- Private PnL should add rounded fee and combined fee-plus-slippage stress before any money-facing verdict.
- Later 10am/noon snapshots should be treated as separate decision labels, not replacements for 7:07am.

## Remaining Risks

1. Morning liquidity may be too thin for reliable paper-PnL conclusions.
2. Candidate corrections are weather-plausible but not market-proven.
3. Low-temperature markets are not validated enough for promotion.
4. `hrrr` duplication can still mislead any downstream analysis that ignores `source_independence_score`.
5. Heat-correction residuals are fixed from historical rows; future research should make them train-window-specific for historical backtests.

## Next Actions

1. Merge the PHX/MIA threshold patch after full tests pass.
2. Bobby adds rounded fee and combined fee-plus-slippage stress in private lane.
3. Keep raw `gfs_ens` as default until forward paper results prove otherwise.
4. Let the 7:07am, 10am, and noon decision labels accrue real forward samples.
5. Re-run this audit after 2-3 weeks of forward data.

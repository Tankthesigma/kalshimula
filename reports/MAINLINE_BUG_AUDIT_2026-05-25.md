# Mainline Bug Audit - 2026-05-25

Scope: mainline model and weather-only pipeline. This audit intentionally
excludes Bobby's private market-data lane except for the shared packet
interface. No Kalshi API, order, wallet, portfolio, bank, cancel, or execution
code was added.

## Summary

Two additional correctness bugs were found and fixed after the cold-bias root
cause patch:

1. **Low-market prediction export was unsafe.** The weather-desk CLIs exposed
   `--market-type low`, but the upstream prediction JSON still comes from the
   high-temperature model. That could silently relabel high-temperature PMFs as
   low-temperature packets. The model packet path now fails closed: high-market
   prediction export is the only supported mode until a separately trained
   low-temperature model exists.
2. **ASOS fallback settlement needed a two-day UTC fetch span.** The previous
   fallback could filter observations by station LST, but it still fetched only
   the target UTC date. For U.S. LST settlement days, late-evening observations
   may live on the next UTC date. The forward-test ASOS fallback now fetches
   `target_date` through `target_date + 1 day` before applying station LST
   filtering.

The earlier cold-bias root-cause audit is in
`reports/COLD_BIAS_ROOT_CAUSE_2026-05-25.md`.

## Checks Run

| Check | Result |
|---|---:|
| `ruff check .` | pass |
| targeted pytest after fixes | pass |
| 2-year rows | 55,490 |
| 2-year city-date labels | 7,300 |
| city-date groups with inconsistent `actual_high_f` across sources | 0 |
| generated `predictions_nowcast.csv` packet files scanned | 672 |
| generated packet probability-sum / PMF JSON-sum issues | 0 |
| high station-rule mismatches vs `config/stations.yaml` | 0 |

## Reviewed Areas

- Station config and shared station/rule table.
- NCEI training-label path and POWER fallback coverage.
- ASOS hourly fallback settlement path.
- Nowcast feature no-leak filter: `available_ts_utc <= as_of`.
- Target-day filtering by station local-standard offset.
- Observation coverage reconciliation for impossible `high_so_far`/`low_so_far`.
- Raw, adjusted, heat-corrected, and lone-outlier packet transforms.
- PMF normalization and frozen packet schema.
- Weather-desk refresh, schedule, and backfill entry points.
- Mainline market-data separation.

## Findings

### Fixed: Low Prediction Rows Were Not a Real Low Model

The low station rows and low NWS guidance are useful research inputs, but the
current prediction engine produces daily high-temperature forecasts. Emitting a
low-market packet from that JSON would create structurally wrong probabilities.

Changes:

- `src/models/nowcast_predictions.py` now rejects `market_type != "high"`.
- Weather-desk packet CLIs now expose only `--market-type high` for prediction
  packet generation.
- `docs/research/nowcast-output-schema.md` now states that low-market features
  and guidance are research-only until a true low model exists.
- Regression test added in `tests/test_nowcast_predictions.py`.

### Fixed: ASOS Fallback Needed Next-UTC-Day Rows

Forward settlement prefers NCEI. When NCEI is unavailable, ASOS is a preliminary
fallback. After the LST-date filter fix, the fetch still requested only one UTC
date, which can miss late local-standard observations in the settlement day.

Changes:

- `src/forward_test_settle_cli.py` now uses
  `fetch_asos_observation_csv(station, target, target + 1 day)` for fallback
  settlement before calling `daily_high_from_hourly(..., lst_offset_hours=...)`.
- `tests/test_forward_test_settle_cli.py` asserts the two-day fetch range.

### Verified: Mainline Still Has No Market Execution Surface

The mainline references `kalshi` only as a platform label, station-rule metadata,
documentation, and explicit "no market data/trading" warnings. No mainline
runtime code imports private Kalshi API clients or contains order/portfolio/
wallet/cancel/bank execution code.

### Verified: Packet PMFs Are Internally Normalized

All generated `predictions_nowcast.csv` files available locally were scanned.
For every `(city, platform, market_type, station_id, target_date,
decision_time_label)` group:

- `calibrated_probability` sums to 1.0 within tolerance.
- `pmf_degree_json` is unique per group.
- `pmf_degree_json` probabilities sum to 1.0 within tolerance.

### Verified: Training Labels Are Not Hourly ASOS

The 2-year artifact has one consistent actual high per city/date across all
model sources. The backing cache contains complete NCEI daily-summary `TMAX`
coverage and no POWER fallback files for the run.

## Open Risks

- `heat_corrected`, `adjusted`, and `lone_outlier` remain candidate modes. Raw
  remains the default until Bobby's private forward audit promotes a mode.
- The high model should not be used for lows. Low-temperature markets need a
  separate training target, settlement validation, calibration, and private
  PnL audit.
- ASOS fallback is still lower authority than final NCEI/NWS climate-report
  settlement. Bobby is switching private realized-high scoring to official daily
  highs, which is the right move.
- Current nowcast adjustment has a validated slice crossover: early slices are
  often harmful, late slices helpful. Promotion should be slice-gated and
  forward-tested.


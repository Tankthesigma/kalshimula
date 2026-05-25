# Cold Bias Root Cause Audit - 2026-05-25

Scope: mainline weather-only audit of the observed hot-day cold bias. This report
answers Bobby's H1/H2 split:

1. Did the historical training target accidentally use a UTC/hourly ASOS high
   instead of the settlement-day high?
2. Is the cold bias coming from raw `gfs_ens`, from the baseline bias
   correction, or from the heat-regime candidate?

## Verdict

The primary two-year training labels are not hourly ASOS labels. The artifact
uses NCEI daily-summary `TMAX` labels, with complete NCEI cache coverage:

| Check | Result |
|---|---:|
| City-date pairs in `rows.csv` | 7,300 |
| NCEI cached daily-high payloads | 7,300 |
| POWER fallback payloads | 0 |
| Cities with 730 target dates | 10 / 10 |
| City-date rows where `actual_high_f` differs across sources | 0 |

So the exact `high_so_far` UTC-vs-LST bug found in the nowcast backfill is not
the root cause of the historical training labels.

However, a related fallback risk did exist: preliminary forward settlement used
ASOS hourly rows filtered by calendar date. That is now patched to filter by the
station local-standard settlement day when NCEI is unavailable. This affects
fallback settlement, not the primary two-year training artifact.

The stronger root cause is:

- raw `gfs_ens` is genuinely cold on many hot regimes, especially Denver,
  Phoenix, Houston, Chicago, and Austin;
- the standard recent-90-day bias table already corrects most of that average
  cold bias;
- the candidate `heat_corrected` mode was applying a raw hot-regime correction
  on top of an already bias-corrected packet, which double-counted the same
  bias in several cities.

That explains why the heat candidate helped some cold-miss cases but created
bad over-warming/sign-conflict cases.

## H1 - Training Label Window

Main historical collection path:

- `src/collect.py` calls `_cached_ncei_actual(...)` first.
- NCEI fetcher requests Daily Summaries `TMAX` for `startDate=endDate=target`.
- POWER fallback is only used when NCEI has no high.
- The observed cache for the important 2-year run contains no POWER fallback
  files.

Relevant artifact:

`data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/rows.csv`

The historical label path is therefore official daily-summary station data, not
hourly ASOS aggregation. Bobby's hourly-ASOS-vs-NWS-settlement gap is real for
his private hourly obs store and for preliminary fallback logic, but it does not
describe the labels used to train this artifact.

## H2 - Raw GFS vs Baseline Calibration

Residual convention below: `actual_high_f - point_f`. Positive means the model
was too cold.

| City | Raw all-day residual | Current base correction | Residual after base correction | Hot-regime raw residual | Hot-regime residual after base |
|---|---:|---:|---:|---:|---:|
| austin | +1.135 | +0.799 | +0.336 | +1.266 | +0.467 |
| boston | +0.084 | +1.070 | -0.986 | -1.124 | -2.194 |
| chicago | +1.307 | +0.653 | +0.654 | +1.429 | +0.776 |
| denver | +1.637 | +0.942 | +0.694 | +1.941 | +0.999 |
| houston | +1.485 | +0.505 | +0.979 | +1.710 | +1.204 |
| la | +0.499 | +0.015 | +0.484 | +0.306 | +0.291 |
| miami | +1.001 | +1.275 | -0.274 | +0.795 | -0.480 |
| nyc | -0.325 | +0.248 | -0.573 | -1.312 | -1.560 |
| philadelphia | +0.840 | +1.189 | -0.349 | +1.025 | -0.164 |
| phoenix | +1.834 | +1.516 | +0.318 | +1.846 | +0.330 |

Takeaways:

- Raw `gfs_ens` is cold overall: +0.950F across all 7,300 rows.
- The current recent-90-day bias policy reduces that to +0.128F overall.
- Hot regimes are not uniformly cold after the base correction. Denver,
  Houston, Chicago, Austin, LA, and Phoenix remain cold; Miami, Philadelphia,
  NYC, and Boston are already neutral/hot after base correction.
- Therefore a global hot-day bump is wrong. The correction must be incremental
  after the base bias and city-specific.

## Heat Candidate Bug

Before this audit, `heat_corrected` used a table derived from raw hot-day
residuals, but it consumed `predictions_nowcast_raw/predictions_nowcast.csv`,
whose `calibrated_probability` and `point_f` already include
`bias_correction_f`.

That means the candidate often did:

```text
standard bias correction + raw hot-regime correction
```

instead of:

```text
standard bias correction + remaining hot-regime residual after standard bias
```

Patch applied:

- `heat_corrected` now infers the existing packet bias shift from the raw
  `model_probability` PMF versus the corrected point.
- It emits `correction_f = raw_hot_residual_f - existing_bias_shift_f`.
- `heat_corrections.csv` now includes both values for auditability.

This keeps the candidate weather-only and avoids double-counting the same cold
bias.

## Remaining Risks

- NCEI daily `TMAX` is official daily-summary station data, but Kalshi
  settlement ultimately follows NWS climate-report/settlement rules. Bobby
  should continue comparing official settled values to NCEI labels on the
  overlapping forward window.
- `heat_corrected` remains candidate-only. It should not be promoted unless
  Bobby's private forward paper-PnL gate and the weather-quality gate both pass.
- The candidate correction is still derived from May 2024-April 2026 data; it is
  clean for May 2026+ forward use, but in-sample for backtests inside that
  historical period.

## Actions Taken

- Patched ASOS hourly fallback settlement to respect station LST offset.
- Patched `heat_corrected` so it applies only the incremental hot-regime
  residual left after the existing base bias shift.
- Added tests for settlement-day ASOS filtering and heat-correction
  double-count prevention.


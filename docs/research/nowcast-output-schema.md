# Nowcast Output Schema Contract

Date: 2026-05-24

Status: frozen contract for Codex mainline output and Bobby private audit input.
Initial mainline implementation exists in:

- `src/models/station_rules.py`
- `src/models/nowcast_features.py`
- `src/models/nowcast_predictions.py`
- `src/models/nowcast_adjustment.py`
- `src/models/source_provenance.py`
- `src/models/nowcast_report.py`
- `src/nowcast_features_cli.py`
- `src/nowcast_predictions_cli.py`
- `src/nowcast_adjustment_cli.py`
- `src/source_provenance_cli.py`
- `src/nowcast_report_cli.py`

## Station Rule Table

Canonical path:

```text
config/station_rule_table.csv
```

Ownership:

- Bobby/private lane is the single writer.
- Codex/mainline consumes read-only.
- Mainline may carry a copied snapshot after approval because this is station metadata, not market data.

Columns:

```text
city
platform
market_type
settlement_station
station_name
timezone
lst_offset
dst_policy
unit
rounding_rule
settlement_source
rule_confidence
notes
```

## Mainline Nowcast Predictions

Canonical file:

```text
predictions_nowcast.csv
```

Granularity:

One row per:

```text
city
platform
market_type
target_date
decision_time_label
bin
```

Columns:

```text
model_version
city
platform
market_type
station_id
target_date
prediction_ts_utc
prediction_time_local
decision_time_label
as_of_ts_utc
bin_lower_f
bin_upper_f
bin_label
model_probability
calibrated_probability
point_f
q05_f
q10_f
q20_f
q25_f
q30_f
q40_f
q50_f
q60_f
q70_f
q75_f
q80_f
q90_f
q95_f
pmf_degree_json
source_policy
nowcast_veto_flag
weather_reason_codes
station_rule_confidence
source_independence_score
feature_hash
```

Probability semantics:

- Bobby/private audit compares `calibrated_probability` against market probability.
- `model_probability` is retained for raw/unadjusted diagnostics.
- `nowcast_veto_flag` is weather-only and may not use market prices, order books, price movement, liquidity, private PnL labels, or private audit labels.

Open-ended bin convention:

- Top edge contract: `bin_lower_f=X`, `bin_upper_f=` blank, `bin_label=">=X"`.
- Bottom edge contract: `bin_lower_f=` blank, `bin_upper_f=Y`, `bin_label="<Y"`.
- Closed middle bin: `bin_lower_f=X`, `bin_upper_f=Y`, `bin_label="X-Y"`.

Distribution convention:

- `pmf_degree_json` is a JSON object mapping integer Fahrenheit degree to probability.
- Probabilities in `pmf_degree_json` should sum to approximately 1.0 for the modeled support.
- Decile/quantile columns are included for reporting and sanity checks.
- Bobby should integrate `pmf_degree_json` for exact Kalshi/Polymarket daily bins when listed bins do not match mainline-generated bins.

Decision time labels:

```text
prev_evening
04
07
10
morning
noon
afternoon
final
```

## Manifest

Canonical file:

```text
predictions_nowcast_manifest.json
```

Fields:

```text
schema_version
generated_at
git_commit
model_version
input_hashes
station_table_hash
prediction_date_range
decision_time_labels
no_leak_max_observation_ts
source_independence_summary
row_count
notes
```

No-leak requirement:

- `no_leak_max_observation_ts <= as_of_ts_utc` for every row.
- Feature builders must fail tests if future observations can enter a prediction row.

## Private Audit Join

Bobby/private lane joins on:

```text
city
platform
market_type
target_date
decision_time_label
station_id
```

Then maps market bins to `pmf_degree_json` or exact `bin_lower_f` / `bin_upper_f` rows.

Promotion requires both:

1. Mainline accuracy/calibration gates.
2. Private paper-PnL / divergent-but-right gates.

Nothing promotes on MAE alone.

## Mainline Commands

Build weather-only nowcast features from an observation store:

```text
python -m src.nowcast_features_cli \
  --target-date 2026-05-24 \
  --as-of 2026-05-24T15:00:00Z \
  --decision-time-label morning \
  --observations-csv path/to/asos_observations.csv \
  --out-dir reports/overnight_model_intelligence/nowcast_features
```

Maintain a reusable ASOS observation cache while fetching live rows:

```text
python -m src.nowcast_features_cli \
  --target-date 2026-05-24 \
  --as-of 2026-05-24T15:00:00Z \
  --decision-time-label morning \
  --observation-store reports/overnight_model_intelligence/asos_observation_store.csv \
  --fetch-live \
  --update-observation-store \
  --out-dir reports/overnight_model_intelligence/nowcast_features
```

The store uses the canonical observation columns from `src.models.nowcast_features`.
Rows are de-duplicated by `station_id` and `obs_ts_utc`, keeping the latest
available row. If a live station fetch is rate-limited or fails, the feature
builder continues from cached rows and marks missing/stale stations with
weather-only veto reasons.

High markets are the default. Low markets use the same schema and can be built
explicitly:

```text
python -m src.nowcast_features_cli \
  --market-type low \
  --target-date 2026-05-24 \
  --as-of 2026-05-24T23:00:00Z \
  --decision-time-label evening \
  --observation-store reports/overnight_model_intelligence/asos_observation_store.csv \
  --out-dir reports/overnight_model_intelligence/low_nowcast_features
```

Build Bobby's frozen prediction input:

```text
python -m src.nowcast_predictions_cli \
  --predictions-json data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/latest_predictions.json \
  --market-type high \
  --nowcast-features reports/overnight_model_intelligence/nowcast_features/nowcast_features.csv \
  --decision-time-label morning \
  --out-dir reports/overnight_model_intelligence/nowcast_predictions
```

Audit source independence before treating source counts or blends as independent:

```text
python -m src.source_provenance_cli \
  --input data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/rows.csv \
  --out-dir reports/overnight_model_intelligence/source_provenance
```

The first real provenance run confirms `hrrr` and `gfs_ens` are identical across
all ten mainline cities in the two-year artifact, so `hrrr` should not be counted
as an independent source in blends or contrarian diagnostics.

Render a weather-only report from the frozen prediction export:

```text
python -m src.nowcast_report_cli \
  --predictions-nowcast reports/overnight_model_intelligence/nowcast_predictions/predictions_nowcast.csv \
  --out-dir reports/overnight_model_intelligence/nowcast_report
```

The report is model-readiness triage only. It does not contain market prices,
order books, private PnL labels, or trading instructions.

Apply a conservative weather-only PMF adjustment:

```text
python -m src.nowcast_adjustment_cli \
  --predictions-nowcast reports/overnight_model_intelligence/nowcast_predictions/predictions_nowcast.csv \
  --nowcast-features reports/overnight_model_intelligence/nowcast_features/nowcast_features.csv \
  --out-dir reports/overnight_model_intelligence/nowcast_predictions_adjusted
```

For high-temperature markets, this enforces the physical constraint that
final high cannot be below `high_so_far_f` as of the prediction time. It
truncates `pmf_degree_json` and `calibrated_probability` below that observed
floor, then renormalizes. It does not use market prices or private audit labels.
For low-temperature markets, the symmetric weather-only constraint is that final
low cannot be above `low_so_far_f`; the adjustment truncates probability mass
above that observed ceiling and renormalizes.

## Professional Guidance Contract

Direct LAMP/NBM/NWS guidance fetchers should normalize into this weather-only
schema before scoring or model use:

```text
city
source
station_id
market_type
target_date
issue_ts_utc
valid_ts_utc
available_ts_utc
guidance_point_f
guidance_q10_f
guidance_q50_f
guidance_q90_f
actual_high_f
raw_payload_hash
```

The no-leak rule is the same as the ASOS store:

```text
available_ts_utc <= as_of_ts_utc
```

Build diagnostics from normalized guidance:

```text
python -m src.guidance_diagnostics_cli \
  --input path/to/guidance_rows.csv \
  --as-of 2026-05-24T15:00:00Z \
  --target-date 2026-05-24 \
  --out-dir reports/overnight_model_intelligence/guidance
```

Outputs:

```text
guidance_latest.csv
guidance_score_summary.csv
guidance_report.md
guidance_manifest.json
```

This keeps professional guidance benchmarkable before it is allowed into the
structured nowcast model stack.

The first built-in guidance source is the public NWS forecast API:

```text
python -m src.nws_guidance_cli \
  --date 2026-05-24 \
  --market-type high \
  --cities nyc,chicago,miami,austin,la,denver,philadelphia,houston,phoenix,boston \
  --out reports/overnight_model_intelligence/guidance/nws_guidance_rows.csv
```

This fetches the existing `points -> forecast` API path already used by the NWS
daily-high fetcher and normalizes the daytime high into the professional
guidance schema. It is a weather-only source; it does not touch market APIs.
Use `--market-type low` or `--market-type both` to emit nighttime low guidance
rows. Low rows are inputs for weather research only until Bobby/private audit
empirically validates the low-market station and LST/DST settlement behavior.

## One-Command Weather Desk Pipeline

For operational use, run the mainline weather-only pipeline in one command:

```text
python -m src.weather_desk_cli \
  --predictions-json data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/latest_predictions.json \
  --market-type high \
  --target-date 2026-05-24 \
  --as-of 2026-05-24T15:00:00Z \
  --decision-time-label morning \
  --observation-store reports/overnight_model_intelligence/asos_observation_store.csv \
  --fetch-live \
  --update-observation-store \
  --include-nws-guidance \
  --out-dir reports/overnight_model_intelligence/weather_desk
```

It writes:

```text
nowcast_features/
predictions_nowcast_raw/
predictions_nowcast_adjusted/
nowcast_report/
guidance/
guidance_diagnostics/
weather_desk_manifest.json
```

`predictions_nowcast_adjusted/predictions_nowcast.csv` is the canonical
weather-adjusted model mode for Bobby/private audit. It uses the same frozen
schema as raw `predictions_nowcast.csv`, so private PnL tooling can compare raw
versus adjusted without a separate parser.
When `--include-nws-guidance` is set, the packet also writes
`guidance/model_vs_nws_guidance.csv`, a weather-only comparison of model point
versus public NWS guidance point. It is an accuracy/desk diagnostic only, not a
market signal.

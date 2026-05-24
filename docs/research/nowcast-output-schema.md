# Nowcast Output Schema Contract

Date: 2026-05-24

Status: frozen contract for Codex mainline output and Bobby private audit input.
Initial mainline implementation exists in:

- `src/models/station_rules.py`
- `src/models/nowcast_features.py`
- `src/models/nowcast_predictions.py`
- `src/models/source_provenance.py`
- `src/models/nowcast_report.py`
- `src/nowcast_features_cli.py`
- `src/nowcast_predictions_cli.py`
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

Build Bobby's frozen prediction input:

```text
python -m src.nowcast_predictions_cli \
  --predictions-json data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/latest_predictions.json \
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

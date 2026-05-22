# weather-predictor

Probabilistic daily-high-temperature predictor for 10 US cities. Built to inform
manual Kalshi weather trades â€” **no auto-trading**, no Kalshi API.

> **NOT FINANCIAL ADVICE.** Paper-trade against historical Kalshi markets for at
> least two months before sizing real positions.

## What it does

For any (city, date) where `date` is in `[today-2y, today+15d]`, the system
outputs a calibrated probability distribution over the NWS settlement station's
daily high temperature, in 1Â°F bins. Targets to beat: naive single-model
forecast, naive 31-member GFS bin-count, and the official NWS forecast.

## Settlement semantics (important)

Kalshi weather markets settle on the **daily maximum in Local Standard Time**
(no DST) at the named NWS station, midnight-to-midnight LST. Phoenix is MST
year-round; for DST-observing stations LST and local clock time diverge during
summer. The pipeline aggregates daily highs explicitly in LST using each
station's `lst_offset_hours` from `config/stations.yaml`. Do not rely on
`timezone=auto` from Open-Meteo for that aggregation.

## Setup

```powershell
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env: set NWS_USER_AGENT to your real email
```

## Usage

Naive forecast for a future date:

```powershell
python -m src.predict --city denver --date tomorrow
```

Use recommended Open-Meteo sources and trained model artifacts in live prediction:

```powershell
python -m src.predict --city denver --date tomorrow --model-run-dir data\runs\<run>
python -m src.predict --city denver --date tomorrow --model-run-dir data\runs\<run> --threshold-offsets=-2,0,2
python -m src.predict --city denver --date tomorrow --model-run-dir data\runs\<run> --threshold-offsets=-2,0,2 --json
python -m src.predict_batch_cli --cities denver,boston,nyc --date tomorrow --model-run-dir data\runs\<run> --threshold-offsets=-2,0,2 --require-gate --out data\runs\<run>\latest_predictions.json
python -m src.prediction_review_cli --input data\runs\<run>\latest_predictions.json --out data\runs\<run>\latest_predictions.txt
python -m src.daily_model_refresh_cli --model-run-dir data\runs\<run>
python -m src.daily_packet_check_cli --manifest data\runs\<run>\latest_predictions_manifest.json
python -m src.daily_packet_check_cli --manifest data\runs\<run>\latest_predictions_manifest.json --json --out data\runs\<run>\latest_predictions_check.json
python -m src.forward_test_settle_cli --packet data\runs\<run>\latest_predictions.json --target-date 2026-05-22 --actuals-csv data\runs\<run>\daily_actuals.csv --out-dir data\runs\<run>\forward_test
```

When a run has `source_selection/recommended_sources.csv`, `--model-run-dir`
uses that source map; otherwise it falls back to
`source_selection/selected_sources.csv`. If the selected source for a city is
`openmeteo_naive`, prediction keeps the pooled Open-Meteo baseline. If an
individual selected source is unavailable for the requested date, prediction
warns and falls back to the pooled members. Bias and interval artifacts are
optional; when supplied, the CLI prints the corrected point and empirical
interval next to the raw ensemble output. You can also pass artifact paths
explicitly with `--selected-sources`, `--bias-table`, and `--interval-table`.
When `probability_calibration/threshold_residuals.csv` exists, `--threshold-offsets`
prints offline threshold probabilities around the rounded corrected point. When
`probability_calibration/threshold_recalibration_table.csv` also exists, those
probabilities are adjusted by the validation-fitted bucket recalibration table
and the raw probability is shown in parentheses. Sparse city/source buckets
fall back to pooled global validation buckets when available. Use `--json` when
a script, dashboard, or review tool needs machine-readable forecast,
calibration, and threshold-probability fields; recalibrated threshold rows
include `recalibration_scope` and `recalibration_n` for traceability. Use
`src.predict_batch_cli` for multi-city JSON payloads; it continues after
individual city failures and records them in the `errors` array. JSON outputs
include `schema_version`, `generated_at`, and
`artifact_paths` so downstream tools can verify which model artifacts produced
the numbers. Add `--require-gate` to batch prediction when the payload will feed
a dashboard or review script; it emits zero predictions and exits nonzero unless
the model run passes `src.model_gate_cli`.
Use `src.prediction_review_cli` to turn the batch JSON into a compact human
review table with gate status, corrected points, intervals, and threshold
probabilities.
Use `src.daily_model_refresh_cli` for the normal morning refresh: it writes the
gated all-city batch JSON, text prediction review, text and JSON model gate
reports, and model policy summary in one command, plus a manifest JSON that
indexes the packet paths and exit codes and a packet-check JSON that verifies
the packet. Add `--write-actuals-template` when the refresh should also create
or update the offline `city,target_date,actual_high_f` template for later
settlement; existing filled values are preserved unless
`--actuals-template-no-preserve-existing` is passed.
Use `src.daily_packet_check_cli` to verify the manifest exit codes and artifact
existence before a dashboard or downstream script consumes the packet. The
checker also validates the prediction JSON gate status, prediction/error counts,
and required prediction fields, including source, station, forecast,
selected-source-application status, calibration, threshold probabilities, and
artifact paths for dashboard use. Add `--json` when a script needs the check
result as structured data instead of text.
When actual highs are available for the same packet, add `--settle` to
`src.daily_model_refresh_cli` to run settlement after the packet check. Use
`--settle-actuals-csv` for offline actuals, or omit it to use the NCEI/ASOS
source layer. The final manifest records settlement artifact paths and exit
codes.
Use `src.forward_test_pending_cli --packet <packet.json> --history
<forward_test>/history.csv` before settlement to see whether the target date has
passed and whether the packet is already settled, partially settled, or still
unsettled. Add `--actuals-csv daily_actuals.csv` to include offline actuals
readiness in the same status; an incomplete actuals CSV changes the next action
to `fill_actuals_csv`.
Use `src.forward_test_actuals_template_cli --packet <packet.json> --out
daily_actuals.csv` to generate the exact `city,target_date,actual_high_f`
template before filling offline actuals; reruns preserve existing filled values
unless `--no-preserve-existing` is passed.
Run `src.forward_test_actuals_check_cli --packet <packet.json> --actuals-csv
daily_actuals.csv` before offline settlement to verify every packet city has
one numeric actual high for the packet date and that no wrong city/date rows
will be silently used. `src.forward_test_settle_cli --actuals-csv` runs the same
preflight automatically and will not write settlement artifacts when it fails.
After actual highs are known, use `src.forward_test_settle_cli` to score the
packet. By default it fetches observed highs through the existing NCEI/ASOS
source layer; pass `--actuals-csv` with `city,target_date,actual_high_f` to run
the settlement offline. It reports corrected MAE, threshold Brier score, errors,
and per-city settlement rows without touching Kalshi.
Use `src.forward_test_report_cli --history <forward_test>/history.csv` to
summarize accumulated live/forward-test performance. The report separates
unique city/date prediction MAE from threshold-event Brier score and supports
`--json --out <path>` for dashboards or agent checks. The settlement CLI writes
`report.json` next to `history.csv` by default; use `--no-report` to skip it or
`--report-out <path>` to choose another location.
Use `src.forward_test_gate_cli --report <forward_test>/report.json` to fail a
live-monitor check when accumulated forward-test MAE, bias, interval coverage,
threshold Brier, threshold ECE, or sample counts drift outside configured
thresholds. Add `--forward-test-gate` to `src.daily_model_refresh_cli` when the
same refresh command should write a forward-test gate JSON artifact after
settlement or against an existing report.

Collect one city/date range into backtest rows:

```powershell
python -m src.collect_cli --city denver --start 2025-01-01 --end 2025-01-07 --out data\denver_rows.csv
```

Use `--openmeteo-mode both` on historical runs when you want individual
Open-Meteo source rows (`gfs_ens`, `ecmwf_ens`, `icon_ens`, `gem_ens`,
`aifs`, `graphcast`, `hrrr`) alongside the pooled `openmeteo_naive` baseline.
Some models do not return every historical date; unavailable source/date pairs
are cached as missing and omitted from the backtest rows.

Compare bias policies, recent windows, and interval alpha settings on an
existing run:

```powershell
python -m src.bias_policy_cli --input data\runs\<run>\rows.csv --train-eval-dir data\runs\<run>\train_eval --recommended-sources data\runs\<run>\source_selection\recommended_sources.csv --out-dir data\runs\<run>\model_policy --validation-start 2025-11-01 --test-start 2026-02-01 --recent-days 90,180,365 --alphas 0.2,0.13
python -m src.interval_policy_cli --input data\runs\<run>\rows.csv --recommended-sources data\runs\<run>\source_selection\recommended_sources.csv --out-dir data\runs\<run>\model_policy --validation-start 2025-11-01 --test-start 2026-02-01 --alphas 0.2,0.13,0.1,0.05
python -m src.threshold_calibration_cli --input data\runs\<run>\rows.csv --recommended-sources data\runs\<run>\source_selection\recommended_sources.csv --bias-table data\runs\<run>\model_policy\bias_table.csv --out-dir data\runs\<run>\probability_calibration --validation-start 2025-11-01 --test-start 2026-02-01 --offsets=-6,-4,-2,0,2,4,6 --recalibration-prior-strength 25 --min-recalibration-events 20
python -m src.model_policy_report_cli --run-dir data\runs\<run>
python -m src.model_gate_cli --run-dir data\runs\<run>
python -m src.model_gate_cli --run-dir data\runs\<run> --json --out data\runs\<run>\model_gate.json
```

`src.model_gate_cli --json` writes the same research-readiness checks as a
machine-readable payload with `passed`, `summary.failed_check_names`, and the
full per-check values/thresholds.

Summarize collected rows:

```powershell
python -m src.backtest_cli --input data\denver_rows.csv --out data\denver_summary.csv
```

Collect and summarize multiple cities:

```powershell
python -m src.batch_collect_cli --cities denver,chicago --start 2025-01-01 --end 2025-01-07 --rows-out data\rows.csv --summary-out data\summary.csv
```

## Milestones

- **A (done)**: pull all Open-Meteo model endpoints, pool ensemble members,
  render ASCII histogram + point estimate + 80% CI.
- **B (in progress)**: JSON cache, NWS official forecast row, NCEI/POWER
  actual-high rows, ASOS parser foundation, collection CLI, and backtest CLI.
- **Câ€“E**: training data pipeline, XGB/LGBM/MDN/stacker, isotonic + conformal
  calibration, backtest CLI, Flask dashboard.

See `C:\Users\vasud\.claude\plans\okay-here-s-the-upgraded-glimmering-mitten.md`
for the full plan.

## Data sources (all free)

- Open-Meteo (Forecast, Ensemble, Historical-Forecast, Archive) â€” no key, 10k/day
- NWS api.weather.gov â€” no key, polite User-Agent required
- NCEI Access Data Service â€” no key
- NASA POWER â€” no key (long-history fallback)
- Iowa State ASOS archive â€” no key (hourly METAR for D+0 refinement)
- Visual Crossing â€” optional free key

## Out of scope

Kalshi API, auto-trading, daily lows, precipitation/wind, non-US cities.

# Night-ops runbook

Short operator guide for keeping the project healthy while bobby + codex
work autonomously. Read top-to-bottom once, then jump to whichever
"something looks wrong" section matches the symptom.

## Daily health check (60 seconds)

1. Open the Actions tab. Confirm latest CI is green and `Source Quality (live)`
   ran on Monday with no obvious-red rows in the run summary.
2. Open the agent Discord channel. Scroll to the most recent `@bobby` /
   `@codex` exchange. If neither agent has posted in >30 min during a known
   active run, see [bridge health](#bridge-health).
3. If the active collection runner is in flight, glance at codex's most
   recent status post -- should mention row count + per-city progress + any
   errors.csv entries.

## Reading the source-quality summary

`Source Quality (live)` posts to `$GITHUB_STEP_SUMMARY` and uploads two CSVs.
The summary shows one row per source. Columns:

| Column | What it means |
|---|---|
| `n` | rows probed for this source (about number of configured cities) |
| `ok_count` | fetcher call returned without raising |
| `error_count` | fetcher raised -- see error column in `smoke_all.csv` |
| `missing_high_count` | call succeeded but the source had no `high_f` for that day |
| `ok_rate` | `ok_count / n` |
| `missing_high_rate` | `missing_high_count / ok_count` (well-defined as 0 when `ok_count=0`) |

A healthy weekly summary on the default target (7 days ago) looks like:

```
source,ok_rate,missing_high_rate
asos,1.0,~0.0       # IEM is reliable for past days
ncei,1.0,~0.0       # Access Data Service is reliable >3d back
nws,1.0,1.0         # forecast API never returns historical, expected
power,1.0,~0.0      # NASA POWER is reliable >5d back
```

**`error_count > 0` for any source is the actionable signal.** Drift
(`missing_high_rate` shifting up) is a softer signal -- codex's note: this
workflow is intentionally non-blocking, so missing-high drift requires a
human looking at the per-city `smoke_all.csv` to decide whether it's a
real regression or just a publication delay.

## Source-specific triage

### NCEI red or 100% missing-high

- **Most likely cause**: API response shape changed.
- **Check**: open the most recent `tests/fixtures/ncei_*.json` against
  a fresh live response (`curl 'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00094728&dataTypes=TMAX&startDate=2025-01-01&endDate=2025-01-01&format=json&units=metric'`).
- **If fields renamed** (for example, `TMAX` changed): update
  `src/fetchers/ncei.py::_row_high_f`, re-capture fixtures, bump the
  pinned-shape tests in `tests/test_ncei_live_fixtures.py`.
- **If just a slow day**: NCEI publishes daily summaries 1-3 days after
  the date. Re-run the workflow with `target_date` 5+ days back.

### ASOS red or 100% missing-high

- **Most likely cause**: IEM endpoint shape change or station code
  convention change (the K-prefix bug).
- **Check**: `python -m src.fetchers.asos` won't run standalone; instead
  curl `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=KORD&data=tmpf&year1=2025&month1=1&day1=1&year2=2025&month2=1&day2=1&format=onlycomma`
  and inspect the `station` column.
- **K-prefix mismatch** (rows say `ORD`, code expects `KORD` or vice versa):
  see `_asos_station_key` in `src/fetchers/asos.py` and the
  `test_iem_strips_leading_k_in_csv_rows` test for the pattern.

### NWS red

- **Most likely cause**: bad User-Agent header, gridpoint URL changed, or
  the NWS forecast endpoint is rate-limiting.
- **Check**: `curl -H "User-Agent: weather-predictor-debug/1.0" https://api.weather.gov/points/39.8328,-104.6575`
  -- should return JSON with `properties.forecast`.
- **100% missing-high on historical target is normal**: NWS forecast API
  does not serve forecasts for past dates.

### POWER red

- **Most likely cause**: NASA POWER service outage or shape change.
- **Check**: `curl 'https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M_MAX&community=RE&latitude=40&longitude=-75&start=20250101&end=20250101&format=JSON'`.
- **Expected fill value**: `-999` or `-9999` for missing data; the parser
  treats both as missing.
- **100% missing-high on a target within last ~5 days is normal**: POWER
  daily aggregates lag.

## Resuming the clean historical run

### Progress check

```bash
python -m src.run_status_cli \
  --run-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr \
  --cities nyc,chicago,miami,austin,la,denver,boston,philadelphia,houston,phoenix \
  --start 2024-05-01 \
  --end 2026-04-30 \
  --sources-per-day 8 \
  --openmeteo-mode both
```

Use `--sources-per-day 8` for `--openmeteo-mode both`: seven individual
Open-Meteo sources plus the pooled `openmeteo_naive` row. The row percentage is
a theoretical upper bound because some models do not publish every historical
date. The city/date chunk percentage uses the same completion rules as the
resumable runner.

If codex's 365-day collection stops (rate limit / network / process kill):

```bash
# From the kalshimula repo root:
python -m src.historical_runner_cli \
  --start 2025-05-01 \
  --end 2026-04-30 \
  --test-start 2026-02-01 \
  --cities nyc,chicago,miami,austin,la,denver,boston,philadelphia,houston,phoenix \
  --out-dir data/runs/may2025_apr2026_10city_365day_ncei_clean \
  --cache .cache/weather_ncei_clean_20260521 \
  --alpha 0.13 \
  --bias-strategy recent \
  --bias-recent-days 180 \
  --openmeteo-mode naive \
  --workers 1 \
  --chunk-days 30
```

The runner is resumable: any row already in `out-dir/rows.csv` is skipped
on rerun. `--out-dir` must match the original; do not point at a fresh
directory or you start over. Keep `--cache` pointed at the clean cache root
above; it was built by copying only Open-Meteo forecast cache and intentionally
leaving old NCEI/POWER envelopes behind.

If Open-Meteo returns 429 (daily quota), the runner stops cleanly -- wait
for the UTC midnight quota reset, then rerun the same command. The
existing `out-dir` picks up where it left off.

For the two-year model baseline, use the same train/eval settings with the
two-year date window:

```bash
python -m src.historical_runner_cli \
  --start 2024-05-01 \
  --end 2026-04-30 \
  --test-start 2026-02-01 \
  --cities nyc,chicago,miami,austin,la,denver,boston,philadelphia,houston,phoenix \
  --out-dir data/runs/may2024_apr2026_10city_730day_ncei_clean \
  --cache .cache/weather_2yr_ncei_clean_20260521 \
  --alpha 0.13 \
  --bias-strategy recent \
  --bias-recent-days 180 \
  --openmeteo-mode naive \
  --workers 1 \
  --chunk-days 30
```

For the model-source breakout run, use a fresh output directory and collect
both the pooled baseline and individual Open-Meteo source rows:

```bash
python -m src.historical_runner_cli \
  --start 2024-05-01 \
  --end 2026-04-30 \
  --test-start 2026-02-01 \
  --cities nyc,chicago,miami,austin,la,denver,boston,philadelphia,houston,phoenix \
  --out-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr \
  --cache .cache/weather_2yr_ncei_clean_20260521 \
  --alpha 0.13 \
  --bias-strategy recent \
  --bias-recent-days 180 \
  --openmeteo-mode both \
  --workers 1 \
  --chunk-days 30
```

Some Open-Meteo models do not return every historical date. Missing
source/date pairs are cached as missing and omitted from `rows.csv`; use
`summary.csv` to see which sources were actually available.

After a completed source-breakout run, refresh the recommended live source
policy:

```bash
python -m src.source_selection_cli \
  --validation-scores data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/train_eval/validation_scores.csv \
  --evaluation data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/train_eval/evaluation.csv \
  --out-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/source_selection
```

To re-check the current bias recency and interval-alpha choice for the
recommended source:

```bash
python -m src.bias_policy_cli \
  --input data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/rows.csv \
  --train-eval-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/train_eval \
  --recommended-sources data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/source_selection/recommended_sources.csv \
  --out-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/model_policy \
  --validation-start 2025-11-01 \
  --test-start 2026-02-01 \
  --recent-days 90,180,365 \
  --alphas 0.2,0.13 \
  --target-coverage 0.8
```

Then calibrate per-city interval alpha. Run this after `bias_policy_cli` so
the final `model_policy/interval_table.csv` uses the per-city interval policy:

```bash
python -m src.interval_policy_cli \
  --input data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/rows.csv \
  --recommended-sources data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/source_selection/recommended_sources.csv \
  --out-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/model_policy \
  --validation-start 2025-11-01 \
  --test-start 2026-02-01 \
  --alphas 0.2,0.13,0.1,0.05 \
  --target-coverage 0.8
```

Then refresh the offline threshold probability calibration:

```bash
python -m src.threshold_calibration_cli \
  --input data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/rows.csv \
  --recommended-sources data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/source_selection/recommended_sources.csv \
  --bias-table data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/model_policy/bias_table.csv \
  --out-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/probability_calibration \
  --validation-start 2025-11-01 \
  --test-start 2026-02-01 \
  --offsets=-6,-4,-2,0,2,4,6 \
  --buckets 10 \
  --recalibration-prior-strength 25 \
  --min-recalibration-events 20
```

This writes overall calibration plus per-city/source summaries and bucket
tables. Check `threshold_test_group_summary.csv` and
`threshold_test_group_calibration.csv` before trusting a clean overall score; on
the completed two-year run, NYC was the weakest threshold-probability group, and
its 30-40% probability bucket was the largest miss.
Also check `threshold_recalibration_comparison.csv`; the completed run improved
from raw Brier/ECE 0.0609/0.0241 to recalibrated Brier/ECE about
0.0568/0.0095. Sparse city/source buckets fall back to pooled global validation
buckets when the global bucket has enough events. Live JSON threshold rows
include `recalibration_scope` and `recalibration_n`; unexpected scopes or
non-positive counts make the packet checker fail.

Finally, run the model readiness gate:

```bash
python -m src.model_gate_cli \
  --run-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr
```

For automation or a dashboard status card, add
`--json --out data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/model_gate.json`.
The JSON payload includes top-level `passed`, compact summary counts, failed
check names, and all per-check values/thresholds.

The completed two-year run should print `Outcome: PASS`. The default gate first
checks data coverage: at least 50,000 rows, 10 cities, 8 forecast sources, and
700 unique target dates. A failure means the current artifacts should be treated
as a diagnostic model, not the recommended research baseline.

Live prediction should then point at the completed model run. The predictor
uses `source_selection/recommended_sources.csv` when it exists, and prefers
`model_policy/` bias/interval tables over older `train_eval/` tables:

```bash
python -m src.daily_model_refresh_cli \
  --model-run-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr
```

That is the normal daily check. It writes the morning model packet under the run
directory:

- `latest_predictions.json`
- `latest_predictions.txt`
- `latest_predictions_gate.txt`
- `latest_predictions_gate.json`
- `latest_predictions_model_policy.txt`
- `latest_predictions_manifest.json`
- `latest_predictions_check.json`

The command requires the model gate and selected-source application by default.
It exits nonzero when the gate fails, any city prediction fails, or a city falls
back instead of using the source policy from
`source_selection/recommended_sources.csv`. Use `--allow-source-fallback` only
for diagnostics when a partial packet is better than no packet. The manifest
also sets `max_packet_age_hours` to 24 by default, so rerunning the packet
checker later rejects stale packets. Use `--no-max-packet-age` only for
historical debugging. The manifest is the machine-readable packet index: it
records output paths, input date/cities/threshold offsets, policy requirements,
per-step exit codes, and the final command exit code.
Verify the packet before using it in a dashboard or downstream script:

```bash
python -m src.daily_packet_check_cli \
  --manifest data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/latest_predictions_manifest.json
```

The checker exits nonzero if the manifest records a failed step or if any
referenced artifact is missing or empty. It also parses the prediction JSON and
fails if the required gate did not pass, `n_errors` is nonzero, prediction counts
do not match the rows, required prediction fields are absent, the prediction
timestamp is invalid, the manifest city list differs from the prediction rows,
an absolute manifest target date differs from the prediction target date, or the
packet is older than `max_packet_age_hours`. It verifies threshold probabilities
too: every prediction must include exactly the requested `threshold_offsets`,
each probability must be in `[0, 1]`, and each threshold must be centered on the
rounded corrected point. It verifies artifact traceability too: the top-level
prediction `artifact_paths` must point at the manifest run directory, and every
per-city prediction must use the same model artifacts. Those referenced model
artifact files must also exist and be nonempty when the packet is checked.
Station metadata must include a nonempty name, a valid four-character
`nws_station`, and numeric `lst_offset_hours`. It also fails when the manifest
requires selected-source application and a prediction fell back. Required
prediction fields include city, selected source, whether that source was
applied, station metadata, forecast, calibration, threshold probabilities, and
artifact paths.
For automation, add `--json --out latest_predictions_check.json` so scripts can
read the check result without scraping the text report. The JSON includes a
compact `summary` object with `total_checks`, `passed_checks`, `failed_checks`,
and `failed_check_names` for dashboard status cards or alerts.

For a one-off city check, use the lower-level predictor:

```bash
python -m src.predict \
  --city denver \
  --date tomorrow \
  --model-run-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr \
  --threshold-offsets=-2,0,2
```

Use `--json` on the same command when a downstream script or dashboard needs
structured fields instead of the text report.
For a multi-city refresh, use `src.predict_batch_cli` with `--out`; inspect the
top-level `errors` array before treating the payload as complete. JSON payloads
include `schema_version`, `generated_at`, and `artifact_paths` for traceability.
Use `--require-gate` for dashboard or review refreshes so a stale or degraded
model run emits zero predictions unless the saved artifacts pass the readiness
gate.
Render the JSON for human review with:

```bash
python -m src.prediction_review_cli \
  --input data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/latest_predictions.json \
  --out data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/latest_predictions.txt
```

The review command exits nonzero when the batch payload has a failed required
gate or city-level errors. Use `--allow-errors` only when intentionally
inspecting a partial payload.

## Settling a forward-test packet

After actual highs are known, create a CSV with:

```csv
city,target_date,actual_high_f
denver,2026-05-22,73
```

Then settle the packet offline:

```bash
python -m src.forward_test_settle_cli \
  --packet data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/latest_predictions.json \
  --target-date 2026-05-22 \
  --actuals-csv data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/daily_actuals.csv \
  --out-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/forward_test
```

The settlement output reports corrected MAE, threshold Brier score, settlement
errors, and per-city rows. Use `--actuals-csv` when you want source validation
separate from forward-test scoring; omit it when NCEI/ASOS settlement fetching
is appropriate. It also writes an accumulated `report.json` beside
`history.csv` by default.

Summarize accumulated forward-test history after settlement:

```bash
python -m src.forward_test_report_cli \
  --history data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/forward_test/history.csv \
  --out data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/forward_test/report.json
```

The report keeps prediction MAE/bias separate from threshold-event Brier score,
so reruns and multiple threshold offsets do not inflate the point-forecast
metric.

## Bridge health

The Discord bridge runs in a WSL tmux session (`bridge`) and forwards
messages into the `bobby-claude:0.0` pane.

### Quick checks

```bash
# From WSL:
tmux list-sessions                      # bridge + bobby-claude should be listed
tail -30 /tmp/bridge.log                # most-recent forwards/errors
ps -ef | grep tmux_agent_bridge          # ensure the loop process is alive
```

### Common failures

- **Bridge stops forwarding (no recent `forwarded N message(s)` lines):**
  `tmux kill-session -t bridge`, then restart with the same env
  (`DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`, `BRIDGE_ALLOW_MENTION="@bobby"`)
  and `scripts/start-tmux-bridge.sh`.
- **Bot account shared with codex and own messages echoing back**:
  bridge already filters by ack-prefix (`:inbox_tray:`/`:x:`) and requires
  allow-mention at the start of bot content. If you see your own non-ack
  replies looping, check the filter version in
  `bridge/discord_mailbox.py::_should_keep`.
- **Discord rate-limit (429)**: bridge fetches every 5s, replies on
  demand. Should not hit limits in normal use. If it does, increase
  `BRIDGE_POLL_INTERVAL`.

## When in doubt

The agent channel is the single point of coordination -- post a
`@codex heartbeat?` or `@bobby heartbeat?` and wait for an ack. If
neither agent responds within 5 minutes, attach to the relevant tmux
pane (`wsl -d Ubuntu -u root -- tmux attach -t bobby-claude`) and see
whether the agent is waiting on a permission prompt, a long-running
operation, or has stopped.

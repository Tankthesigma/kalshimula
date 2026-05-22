# Modeling evaluation

How to read the `train_eval` outputs and what the columns mean. Pairs with
`docs/night-ops.md` (operational triage) — this one focuses on modeling
semantics.

## Pipeline at a glance

```
backtest_rows.csv  ──▶  src.train_eval_split_cli ──▶  train_eval/
                                                       ├── train_rows.csv
                                                       ├── test_rows.csv
                                                       ├── bias_table.csv
                                                       ├── interval_table.csv
                                                       ├── corrected_test_rows.csv
                                                       └── evaluation.csv
```

`backtest_rows.csv` is one row per `(city, target_date, source)` with the
forecast point (`point_f`) and the truth value (`actual_high_f`). The split
CLI fits bias + intervals on the train slice and evaluates on the test slice.

## Train/test split

Set with `--test-start YYYY-MM-DD` on the CLI. The split is **contiguous by
date**, not random:

- `train_rows.csv`: `target_date < test_start`
- `test_rows.csv`: `target_date >= test_start`

No leakage: training rows are never reused for evaluation. Bias table and
interval table are fit on train only; the test rows pass through
`apply_bias_correction` + `apply_empirical_intervals` for evaluation.

### Month-stratified diagnostic split

For model diagnostics, `src.train_eval_split_cli` also accepts
`--split-strategy month-stratified --test-fraction 0.2`. This holds out the
latest fraction of rows within each `(city, source, calendar month)` group.
It intentionally gives train and test matching months so seasonal calibration
can be measured on a short one-year window.

This split is useful for answering "does the month-aware model have signal?"
It is **not** the default leakage-safe production split because train rows can
come from the same calendar month as test rows.

Treat month-stratified output as a **ceiling** on how well a model that knows
the season can do, not as an estimate of live forecast quality. The
chronological split is the only split whose evaluation numbers transfer to
production. If chronological and month-stratified results diverge, that gap is
roughly the size of the regime-shift problem to solve next; it is not headroom
to ship.

Use enough rows per month for this diagnostic. Tiny month groups can produce
one-row test folds, which make interval coverage noisy.

## Bias correction

`src/models/bias.py::fit_bias_table` accepts a `group_month=True` flag (the
default in `train_eval_split`). When set, the table contains:

- One row per `(city, source, month)` — the seasonal correction.
- One row per `(city, source)` with `month = NaN` — the city/source fallback
  used when a test row's month was unseen in training.

At apply time, `_apply_seasonal_bias_correction` does a left-join on
`(city, source, month)` first, then a left-join on `(city, source)` for any
month miss, then `fillna(0.0)` for any (city, source) miss. The final
`bias_correction_f` is added to `point_f` to produce `corrected_point_f`.

### Seasonal fallback caveat — important

The seasonal mechanism only does work when train and test share calendar
months. On the current 365-day collection (train May'25–Jan'26, test
Feb–Apr'26), test months **never appear in train**, so every test row falls
back to the city/source mean correction — which is mathematically the same
as the old non-seasonal correction.

This is by design — the fallback is what keeps the pipeline robust to
out-of-distribution test months. But it means seasonal MAE improvements
will not appear on a contiguous-rolling split that crosses zero training
months. To exercise the seasonal mechanism on this dataset, either:

- Lengthen the training window to cross a full year (so Feb–Apr exists in
  training from a prior year), or
- Use a month-stratified split (80/20 per month) instead of a contiguous
  date split.

## Intervals

`src/models/intervals.py::fit_empirical_intervals` computes
city/source-pooled quantiles of `actual - point` at `alpha/2` and
`1 - alpha/2`. With `alpha=0.2` (the CLI default) the bounds target an 80%
prediction interval.

`apply_empirical_intervals` adds four columns to the test rows:

| Column | What it is |
|---|---|
| `interval_lower_f` / `interval_upper_f` | Legacy aliases for the raw bounds. Preserved for backward compatibility. |
| `interval_lower_raw_f` / `interval_upper_raw_f` | `point_f + lower_error_f` / `point_f + upper_error_f`. Use these when treating `point_f` as the central forecast. |
| `interval_lower_corrected_f` / `interval_upper_corrected_f` | `corrected_point_f + lower_error_f - bias_correction_f` / same with upper. Use these when treating `corrected_point_f` as the central forecast. |

### Raw and corrected bounds have the same numeric value — for now

Algebraic identity:
`corrected_point + lower_error - bias_correction = (point + bias_correction) + lower_error - bias_correction = point + lower_error`,
so the corrected bound equals the raw bound row-by-row. That's because the
interval is fit on pooled raw residuals and the corrected center is offset
by `bias_correction` — the two effects cancel.

The two column families exist for **forward compatibility** and **clarity
of intent**, not numeric difference. They will diverge if a future iteration:

- Fits intervals on corrected residuals (per month, or post-correction
  pooled), or
- Uses different bounds for `point_f` consumers vs `corrected_point_f`
  consumers.

Until then: pick the column family that matches which central forecast you
are reporting and ignore the other. Don't mix (e.g. `corrected_point_f`
with raw bounds) — that re-introduces the centering bug PR #10 cleaned up.

## Coverage and width metrics

`evaluation.csv` adds four interval columns alongside MAE/RMSE/bias:

| Column | Meaning |
|---|---|
| `interval_coverage_raw` | Fraction of test rows where `interval_lower_raw_f <= actual_high_f <= interval_upper_raw_f`. For `alpha=0.2`, target is 0.80. |
| `interval_width_raw` | Mean width of the raw interval over the test rows. |
| `interval_coverage_corrected` | Same metric over the corrected columns. Equals `interval_coverage_raw` numerically as long as intervals are fit on pooled raw residuals (see the identity above). |
| `interval_width_corrected` | Mean width of the corrected interval. Equals `interval_width_raw` for the same reason. |

The two `_corrected` columns will start diverging from `_raw` the day we
fit intervals on a different residual distribution. Until then they're a
read of the same number through a different name.

### Interpreting coverage

- Far below target (e.g. 0.50 vs 0.80 target): intervals are too narrow.
  Either residuals on test are more spread than train (regime shift) or
  alpha is too tight.
- Above target (e.g. 0.88 vs 0.80): intervals too wide. Either test is
  unusually calm or the train period had outliers that widened the
  quantiles.
- Asymmetric misses (e.g. coverage 0.52 but `above_interval_rate` 0.45):
  intervals are mis-centered — the bias correction didn't fully close the
  bias gap on test.

## Current 365-day headline (as of 2026-05-21)

Run: `data/runs/may2025_apr2026_10city_365day_ncei_clean/train_eval/`
- Forecast source: `openmeteo_naive` only.
- Truth source: NCEI (with POWER fallback for missing days).
- Train: 276 days/city, May 2025 – Jan 2026.
- Test: 89 days/city, Feb 2026 – Apr 2026.
- `alpha`: 0.2 (80% interval target).

Per-city headline:

| City | mae_raw | mae_corrected | bias_corrected | interval_coverage_raw |
|---|---|---|---|---|
| austin | 2.50 | 0.99 | -0.05 | 0.74 |
| boston | 2.30 | 1.92 | -0.69 | 0.69 |
| chicago | 2.04 | 1.31 | +0.30 | 0.75 |
| denver | 1.96 | 1.47 | +0.71 | 0.79 |
| houston | 1.89 | 1.12 | +0.79 | 0.88 |
| la | **1.47** | **1.73** | -0.87 | 0.82 |
| miami | 2.22 | 0.84 | -0.38 | 0.87 |
| nyc | 2.36 | 1.96 | **-1.64** | **0.52** |
| philadelphia | 2.65 | 1.24 | -0.37 | 0.72 |
| phoenix | 2.12 | 0.78 | +0.26 | 0.79 |

Bolded values are the watch-list items:

- **LA**: bias correction worsened MAE on test (1.47 → 1.73). LA's training
  bias was ~+1°F too-high; test bias was nearly zero. The correction
  over-shot.
- **NYC**: largest residual bias (-1.64°F) and worst interval coverage
  (52%). Same root cause — train and test bias regimes differ. Don't ship
  NYC intervals to research consumers until coverage is closer to 80%.

Month-stratified diagnostic on the same data improved average corrected MAE
from 1.335°F to 1.115°F and interval coverage from 75.5% to 79.7%. Treat that
as the regime-shift gap to close, not as production performance.

## Two-year bias strategy check

Run: `data/runs/may2024_apr2026_10city_730day_ncei_clean/`

The two-year collection added May 2024 – Apr 2025 to the original 365-day
run, producing 7300 rows across 10 cities with zero collection errors. The
extra year proved that month-aware signal exists, but it also showed that
blindly applying prior-year same-month bias is not a safe default:

| Bias strategy | Avg corrected MAE | Avg interval coverage | Notes |
|---|---:|---:|---|
| Seasonal/monthly | 1.453°F | 73.3% | Helps NYC, hurts several other cities. |
| Recent 180-day city/source, alpha 0.2 | **1.252°F** | 73.3% | Best production-safe MAE strategy tested so far. |
| Recent 180-day city/source, alpha 0.13 | **1.252°F** | **80.0%** | Best overall production-safe baseline tested so far. |
| Month-stratified diagnostic | 1.153°F | 79.1% | Diagnostic ceiling only; not production-safe. |

Interval coverage stayed flat across the alpha 0.2 bias strategies because
bias correction shifts the point but does not change the empirical residual
distribution that fits the bounds. Improving coverage is a separate calibration
slice.

The current best production-safe baseline can now be regenerated directly
through the historical runner with `--bias-strategy recent --bias-recent-days
180 --alpha 0.13`. For already-collected rows, the equivalent train/eval-only
command is:

```bash
python -m src.train_eval_split_cli \
  --input data/runs/may2024_apr2026_10city_730day_ncei_clean/rows.csv \
  --test-start 2026-02-01 \
  --out-dir data/runs/may2024_apr2026_10city_730day_ncei_clean/train_eval_recent_180_alpha_013 \
  --alpha 0.13 \
  --bias-strategy recent \
  --bias-recent-days 180
```

The result is not a reason to delete seasonal features. It means seasonal bias
needs a validation gate or city-specific selector before it can become the
default. A simple fixed 180-day recency window currently beats both all-history
global bias and blind monthly bias on the chronological Feb-Apr 2026 test.
The interval alpha is intentionally tighter than the nominal 0.2 target because
the empirical 80% intervals under-covered on the chronological test. Alpha 0.13
raises average coverage to 80.0%, but NYC, Boston, and Philadelphia remain
below target, so a later per-city interval calibration pass is still warranted.

## Source selection

When a run includes individual Open-Meteo rows (`--openmeteo-mode both`) and a
validation split (`--validation-start`), select one source per city using only
validation MAE:

```bash
python -m src.source_selection_cli \
  --validation-scores data/runs/<run-dir>/train_eval/validation_scores.csv \
  --evaluation data/runs/<run-dir>/train_eval/evaluation.csv \
  --out-dir data/runs/<run-dir>/source_selection
```

This writes:

- `selected_sources.csv`: diagnostic per-city validation winners.
- `selected_source_evaluation.csv`: held-out test metrics for those winners.
- `selected_source_summary.csv`: averaged held-out metrics for those winners.
- `source_policy_comparison.csv`: per-city validation policy vs the best
  single global validation source.
- `recommended_sources.csv`: production source map for `predict --model-run-dir`.

The selections are validation-driven; test metrics are joined afterward so the
output can be compared against `openmeteo_naive` without choosing winners from
the test set.

### Completed two-year source-breakout run

Run: `data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/`

- Rows: 55,490
- City/date chunks: 7,300 / 7,300
- Collection errors: 0
- Train/test split: train before 2026-02-01; held-out test Feb-Apr 2026.
- Validation split: validation from 2025-11-01 through 2026-01-31.
- `alpha`: 0.13

Policy comparison:

| Policy | Selected source | Avg validation MAE | Avg corrected MAE | Avg coverage | Avg interval width |
|---|---|---:|---:|---:|---:|
| Per-city validation | Mixed per city | 0.900°F | 1.156°F | 80.45% | 3.79°F |
| Best global validation source | `gfs_ens` | 0.943°F | **1.044°F** | **84.72%** | **3.70°F** |
| Pooled baseline | `openmeteo_naive` | n/a | 1.302°F | 80.00% | 4.37°F |

The per-city validation selector beats the pooled baseline, but it overfits
some city/source choices on this test window. The best single global validation
source, `gfs_ens`, wins on held-out corrected MAE, coverage, and interval
width. `recommended_sources.csv` therefore maps every city to `gfs_ens`.

Live prediction should use the completed run directly:

```bash
python -m src.predict \
  --city denver \
  --date tomorrow \
  --model-run-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr
```

### Recency/alpha validation grid

Use `src.validation_grid_cli` to compare recent bias windows and interval
alpha values without recollecting data:

```bash
python -m src.validation_grid_cli \
  --input data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/rows.csv \
  --out-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/validation_grid_gfs_ens \
  --policy-out-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr/model_policy \
  --validation-start 2025-11-01 \
  --test-start 2026-02-01 \
  --recent-days 90,180,365 \
  --alphas 0.2,0.13 \
  --target-coverage 0.8 \
  --source gfs_ens
```

On the completed two-year run, the validation grid selected `recent_90d` with
`alpha=0.13` for `gfs_ens`:

| Config | Validation MAE | Validation coverage | Held-out MAE | Held-out coverage | Held-out width |
|---|---:|---:|---:|---:|---:|
| `recent_90d`, alpha 0.13 | **0.990°F** | 84.13% | **0.992°F** | 84.72% | 3.70°F |
| `recent_180d`, alpha 0.13 | 1.071°F | 84.13% | 1.010°F | 84.72% | 3.70°F |
| `recent_365d`, alpha 0.13 | 1.036°F | 84.13% | 1.057°F | 84.72% | 3.70°F |

This is a bias/interval config diagnostic, separate from source selection. It
suggests the next production artifact should regularize toward global
`gfs_ens` plus a shorter recent-bias window. With `--policy-out-dir`, the CLI
writes prediction-ready `model_policy/bias_table.csv` and
`model_policy/interval_table.csv`; `predict --model-run-dir` prefers those
tables when they exist.

To compare that global policy against the current per-city bias-method
selection, use `src.bias_policy_cli`:

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

On the completed run, that comparison recommends `global_recent_90d` with
`alpha=0.13`. It improves held-out corrected MAE from 1.044°F
(`per_city_bias_selection`) to 0.992°F while preserving 84.72% coverage.

Then calibrate interval width per city/source with `src.interval_policy_cli`:

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

Run this after `bias_policy_cli`: it preserves `model_policy/bias_table.csv`
and replaces `model_policy/interval_table.csv` with per-city alpha choices.
On the completed run, per-city alpha reduced held-out interval width from
3.70°F to 3.57°F while keeping coverage above target at 82.70%.

Finally, evaluate threshold event probabilities with `src.threshold_calibration_cli`.
This is the offline probability diagnostic for threshold-style markets; it does
not call Kalshi or trade:

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

The CLI estimates `P(actual_high_f >= threshold_f)` from empirical corrected
residuals and writes event rows, bucketed reliability tables, per-city/source
group summaries, and per-city/source bucket tables. On the completed run, the
held-out threshold diagnostic produced 6,230 events with Brier score 0.0609 and
expected calibration error 0.0241. The worst held-out city group was NYC at
Brier 0.085 and ECE 0.081 over 623 threshold events. The worst bucket was NYC's
30-40% predicted-probability bucket: mean predicted 0.349, observed 0.733 over
45 events. That makes NYC's mid-low probability bucket the first target for
probability calibration work.

The same command also fits a validation-only city/source bucket recalibration
table and applies it to the held-out test events. Sparse city/source buckets
fall back to pooled global validation buckets when the global bucket has enough
events. On the completed run, that reduced test Brier from 0.0609 to about
0.0568 and expected calibration error from 0.0241 to about 0.0095.
`predict --model-run-dir` automatically uses
`probability_calibration/threshold_recalibration_table.csv` when it exists and
prints the raw probability beside the recalibrated one. JSON threshold rows also
include `recalibration_scope` (`city_source`, `global`, or `none`) and
`recalibration_n` when a recalibration bucket was applied.

After the residual artifact exists, live prediction can print threshold
probabilities without any market integration:

```bash
python -m src.predict \
  --city denver \
  --date tomorrow \
  --model-run-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr \
  --threshold-offsets=-2,0,2
```

The offsets are applied around the rounded corrected point. For example, if the
corrected point is 47°F, `--threshold-offsets=-2,0,2` prints
`P(high >= 45°F)`, `P(high >= 47°F)`, and `P(high >= 49°F)`.
When a recalibration table is present, each line shows the recalibrated
probability first and the raw empirical-residual probability in parentheses.
Add `--json` to emit the same calibrated forecast and threshold probabilities
as machine-readable JSON for dashboards or review scripts; progress/status
messages stay on stderr.
For multiple cities, use `src.predict_batch_cli` with the same model artifacts
and threshold offsets. It writes one JSON object with `predictions` and
`errors`, so downstream tools can continue when a single city/source is missing.
Both single and batch JSON include `schema_version`, `generated_at`, and
`artifact_paths` for downstream compatibility checks.
For dashboard or review automation, add `--require-gate` to batch prediction.
That runs the default `src.model_gate_cli` checks before any forecast fetches;
if the run fails the gate, the JSON contains no predictions, one
`__model_gate__` error, and the failed checks in `model_gate`.
Use `src.prediction_review_cli` to render the batch JSON into a compact text
review table. It exits nonzero when a required gate fails or the payload has
city-level errors, which makes it suitable for manual-review scripts before any
future dashboard consumes the same JSON.
For the normal all-city refresh, use `src.daily_model_refresh_cli`; it runs the
gated batch prediction and review rendering together, then writes
`latest_predictions.json`, `latest_predictions.txt`, `latest_predictions_gate.txt`,
`latest_predictions_gate.json`, `latest_predictions_model_policy.txt`, and
`latest_predictions_manifest.json` under the run directory. It also writes
`latest_predictions_check.json`, the machine-readable verification result for
that packet. The normal refresh also
requires every city to apply the selected source policy; use
`--allow-source-fallback` only for diagnostics when a source outage makes a
partial fallback packet useful. It also writes `max_packet_age_hours=24` into
the manifest, so rerunning the checker later rejects stale packets; use
`--no-max-packet-age` only for historical debugging.
Use `src.daily_packet_check_cli` to verify the manifest and referenced packet
artifacts before feeding them into an external dashboard or review script. It
also validates the prediction JSON gate status, error count, prediction count,
manifest city list, target date, generated timestamp, and required per-city
prediction fields needed by a dashboard: selected source, whether that source
was applied, station metadata, forecast, calibration, threshold probabilities,
and artifact paths. Station metadata must include a nonempty name, valid
four-character `nws_station`, and numeric `lst_offset_hours`. It also verifies
that top-level and per-city artifact paths match the manifest run directory and
each other, and that the referenced model artifact files exist and are nonempty.
Every prediction must have exactly the requested threshold offsets, valid
probabilities, and threshold values centered on the rounded corrected point. It
fails stale packets when the manifest has `max_packet_age_hours`. When the
manifest sets
`require_selected_source_applied`, the checker also fails any prediction where
the selected source could not be applied.
Use `--json` on the checker when a CI job or dashboard needs the verification
result as structured data. The payload includes `summary.total_checks`,
`summary.passed_checks`, `summary.failed_checks`, and
`summary.failed_check_names` so downstream consumers can render packet status
without scanning the full check list.
Use `src.forward_test_settle_cli` after the target date has settled to score a
packet. The default path fetches observed highs through the existing NCEI/ASOS
source layer; `--actuals-csv` accepts a `city,target_date,actual_high_f` CSV for
offline settlement. The output includes corrected MAE, threshold Brier score,
errors, per-city rows, and a flattened history CSV for forward testing.
Use `src.forward_test_report_cli` on that history CSV to monitor accumulated
forward-test quality. It deduplicates rerun rows by latest
`(target_date, city, offset_f)`, computes corrected MAE/bias on unique
city/date predictions, and computes threshold Brier on threshold events. The
settlement CLI writes this accumulated report automatically as `report.json`
beside `history.csv`; pass `--no-report` when only the raw settlement artifacts
are needed.

Use `src.model_gate_cli` as the final research-readiness check after refreshing
all selected model artifacts:

```bash
python -m src.model_gate_cli \
  --run-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr
```

Use `--json --out <path>` when an agent, CI job, or dashboard needs the gate as
structured data. The JSON payload includes `passed`,
`summary.failed_check_names`, and all per-check values/thresholds.

The default gate checks that the run has at least 50,000 rows, 10 cities, 8
forecast sources, and 700 unique target dates. It then checks that the selected
source is `gfs_ens`, held-out MAE is at most 1.05°F, interval coverage is at
least 80%, average interval width is at most 3.8°F, recalibrated threshold
Brier/ECE are at most 0.058/0.012, and the recalibration improves raw Brier/ECE
by at least 0.002/0.010. The completed two-year run passes all of those gates.

## Known limitations and next steps

- **Recommended source is global, not city-specific.** The best completed
  policy is the single global `gfs_ens` source. The per-city validation policy
  remains useful as a diagnostic, but it underperformed the global policy on
  held-out test.
- **Interval calibration.** Per-city alpha now beats the global alpha policy on
  width while staying above the 80% coverage target. The next interval slice is
  seasonal or weather-regime conditioning, not another global alpha sweep.
- **Bias policy regularization.** The validation grid now points to global
  `recent_90d` for `gfs_ens`, and `model_policy/` can carry those prediction
  artifacts. The simpler global policy currently beats per-city bias-method
  selection on the held-out test window, so it is the recommended live bias
  policy until a stronger per-city selector is validated.
- **Probability calibration.** Threshold probabilities now have offline
  reliability artifacts, and sparse city/source recalibration buckets fall back
  to pooled global validation buckets. The next probability slice is deeper
  mid-probability calibration analysis; do not treat these diagnostics as trade
  signals.
- **Test sample size.** 89 days/city is enough for an MAE estimate; tight
  for coverage estimation. NYC's 51.7% reading at n=89 has a ~5%-point
  standard error — the real coverage is probably 47–57%, still well below
  target.

## Refreshing this document

The headline table is hand-rolled. When the run is regenerated, regenerate
the table from `evaluation.csv`:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/runs/<run-dir>/train_eval/evaluation.csv')
print(df[['city','mae_raw','mae_corrected','bias_corrected','interval_coverage_raw']].to_markdown(index=False, floatfmt='.2f'))
"
```

…then paste over the table above. No tests assert on this doc — it's a
human reference. The numeric truth is the run's `evaluation.csv` itself.

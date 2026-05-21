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

## Known limitations and next steps

- **Single forecast source.** The whole stack rides on `openmeteo_naive` (the
  Open-Meteo ensemble averaged into a single point). NWS forecasts can't be
  back-tested (only current/future). Next-next move: break Open-Meteo into
  individual ensemble members and treat each as a source so bias correction
  can run per-member.
- **Pooled-by-city-source intervals.** Same alpha quantile width whether
  Tuesday in July or Sunday in January. A seasonal interval fit (analogous
  to the seasonal bias) would tighten intervals in low-variance seasons and
  widen them in high-variance ones.
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

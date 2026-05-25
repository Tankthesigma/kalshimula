# Mainline Audit - 2026-05-25

Scope: mainline weather code, station metadata, model artifacts, nowcast packet contract, scheduler/as-of handling, and test coverage. This audit intentionally excludes private market prices, order books, PnL labels, secrets, and execution code.

## Verdict

Mainline remains weather-only and structurally clean. The nowcast feature path enforces point-in-time observation availability before computing same-day features, the local-time scheduler correctly converts each city's local decision hour to its own UTC `as_of`, and candidate packet modes remain separated from the raw default.

One no-leak hardening change was made during this audit: ASOS rows fetched by mainline now receive a default 10-minute `available_ts_utc` lag instead of being treated as available at the exact observation timestamp. Existing observation stores that already carry explicit `available_ts_utc` continue to be respected.

## Checks Run

- `ruff check src/models/nowcast_features.py src/weather_desk_backfill_cli.py tests/test_nowcast_features.py tests/test_weather_desk_backfill_cli.py` - passed
- `pytest tests/test_nowcast_features.py tests/test_weather_desk_backfill_cli.py` - 16 passed
- `ruff check .` - passed
- `pytest` - 694 passed

## Findings

### High Severity

No high-severity mainline leakage or market-data contamination found.

### Medium Severity

1. Historical schedule backfills are valid for raw-vs-adjusted nowcast comparison, but they are not a perfect historical forecast replay unless all upstream forecast inputs were archived as-of. The schedule CLI correctly enforces ASOS observation as-of; the base forecast JSON still comes from the current model-run machinery.
2. Houston was recently corrected from KIAH to KHOU. The station config is now correct, but any pre-fix Houston residual/calibration artifact remains stale until rebuilt on KHOU history.
3. Low-temperature station/window validation is still medium confidence. Low packets should stay candidate-only until Bobby privately validates low-market settlement windows, especially DST/LST overnight behavior.

### Low Severity / Hardened

1. Fetched ASOS observation rows previously used `available_ts_utc = obs_ts_utc`. That is mildly optimistic for historical backfills. Mainline now applies a 10-minute default availability lag when converting fetched ASOS rows.
2. Running multi-day schedule backfills by hand is error-prone. Added `weather_desk_backfill_cli` as a thin reproducible wrapper around the existing local-time schedule CLI.

## No-Leak Review

Reviewed paths:

- `src/models/nowcast_features.py`
  - Parses ISO and common epoch timestamps defensively.
  - Filters `available_ts_utc <= as_of_ts` before same-day station selection.
  - Computes `high_so_far_f`, `low_so_far_f`, slopes, and coverage only from the filtered observation set.
  - Emits `no_leak_max_observation_ts` and `observation_coverage.csv`.
- `src/models/nowcast_adjustment.py`
  - Reads the frozen raw packet and filtered nowcast features.
  - Uses only weather columns (`high_so_far_f`, `low_so_far_f`, remaining heating/cooling estimates).
  - Does not read prices, order books, PnL, or private audit labels.
- `src/weather_desk_schedule_cli.py`
  - Converts each city/date/local decision hour using the station rule timezone.
  - Passes that UTC timestamp into the same weather desk path.
  - Writes one packet per city/slice, making time-zone boundaries explicit.
- `src/weather_desk_backfill_cli.py`
  - Delegates each date to `weather_desk_schedule_cli`.
  - Adds no new modeling logic and no market data.

## Station Rules

Current high-market stations:

- NYC `KNYC`
- Chicago `KMDW`
- Miami `KMIA`
- Austin `KAUS` with medium confidence due to KAUS/KATT ambiguity in Bobby's sample
- LA `KLAX`
- Denver `KDEN`
- Philadelphia `KPHL`
- Houston `KHOU`
- Phoenix `KPHX`
- Boston `KBOS`

Mainline station table and `config/stations.yaml` agree on Houston `KHOU`. Low rows exist but remain medium-confidence.

## Market-Separation Review

Search over `src`, `config`, `tests`, `docs`, and `reports` found Kalshi/market language in:

- station/rule metadata,
- docs/research/reporting,
- explicit guardrail strings,
- tests using `platform=kalshi`,
- old audit reports.

No mainline executable market API, order, wallet, portfolio, cancel, bank, or execution path was found in `src`.

## Overnight Backfill Command

Use this to produce the multi-day weather-only packet set Bobby can aggregate:

```bash
/mnt/c/Users/vasud/OneDrive/Documents/kalshimula/.venv/Scripts/python.exe -m src.weather_desk_backfill_cli \
  --model-run-dir data/runs/may2024_apr2026_10city_openmeteo_sources_2yr \
  --start-date 2026-05-01 \
  --end-date 2026-05-24 \
  --cities nyc,chicago,miami,austin,la,denver,philadelphia,houston,phoenix,boston \
  --decision-hours 04,07,10,13,15 \
  --threshold-offsets=-6,-4,-2,0,2,4,6 \
  --multi-source-mode single \
  --station-rules config/station_rule_table.csv \
  --market-type high \
  --observation-store outputs/weather_desk_backfill/asos_observation_store.csv \
  --update-observation-store \
  --fetch-live \
  --include-nws-guidance \
  --no-require-gate \
  --continue-on-error \
  --out-dir outputs/weather_desk_backfill/may2026_highs
```

Important interpretation:

- Weather-quality metrics should be computed only on `observation_coverage.csv` rows with `coverage_ok=true`.
- Tradeability/PnL remains Bobby-private and must be reported separately from weather accuracy.
- `predictions_nowcast_raw` remains the reference/default. `adjusted`, `heat_corrected`, and `lone_outlier` remain candidate modes until forward private validation promotes them.

## Clean-Up Notes

- Do not commit `outputs/`; it is generated packet/backfill data.
- Do not commit `PROJECT_HANDOFF.md` unless Tanmay explicitly asks.
- `hrrr` remains non-independent from `gfs_ens`; downstream analysis should keep using `source_independence_score`.

## Next Actions

1. Run the multi-day backfill if runtime/network budget allows.
2. Hand Bobby the backfill output root and manifest.
3. Bobby aggregates adjusted-vs-raw weather quality on green coverage rows, and separately reports paper-PnL/tradeability using private market snapshots.

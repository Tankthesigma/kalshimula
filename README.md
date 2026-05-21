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

Collect one city/date range into backtest rows:

```powershell
python -m src.collect_cli --city denver --start 2025-01-01 --end 2025-01-07 --out data\denver_rows.csv
```

Use `--openmeteo-mode both` on historical runs when you want individual
Open-Meteo source rows (`gfs_ens`, `ecmwf_ens`, `icon_ens`, `gem_ens`,
`aifs`, `graphcast`, `hrrr`) alongside the pooled `openmeteo_naive` baseline.
Some models do not return every historical date; unavailable source/date pairs
are cached as missing and omitted from the backtest rows.

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

# weather-predictor

Probabilistic daily-high-temperature predictor for 10 US cities. Built to inform
manual Kalshi weather trades — **no auto-trading**, no Kalshi API.

> **NOT FINANCIAL ADVICE.** Paper-trade against historical Kalshi markets for at
> least two months before sizing real positions.

## What it does

For any (city, date) where `date` is in `[today-2y, today+15d]`, the system
outputs a calibrated probability distribution over the NWS settlement station's
daily high temperature, in 1°F bins. Targets to beat: naive single-model
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

## Usage (Milestone A — naive ensemble)

```powershell
python -m src.predict --city denver --date tomorrow
```

## Milestones

- **A (current)**: pull all 6 Open-Meteo model endpoints, pool ~150 ensemble
  members, render ASCII histogram + point estimate + 80% CI.
- **B**: SQLite cache, NWS official forecast row, NCEI bias correction, all 10
  cities, polished CLI.
- **C–E**: training data pipeline, XGB/LGBM/MDN/stacker, isotonic + conformal
  calibration, backtest CLI, Flask dashboard.

See `C:\Users\vasud\.claude\plans\okay-here-s-the-upgraded-glimmering-mitten.md`
for the full plan.

## Data sources (all free)

- Open-Meteo (Forecast, Ensemble, Historical-Forecast, Archive) — no key, 10k/day
- NWS api.weather.gov — no key, polite User-Agent required
- NCEI Access Data Service — no key
- NASA POWER — no key (long-history fallback)
- Iowa State ASOS archive — no key (hourly METAR for D+0 refinement)
- Visual Crossing — optional free key

## Out of scope

Kalshi API, auto-trading, daily lows, precipitation/wind, non-US cities.

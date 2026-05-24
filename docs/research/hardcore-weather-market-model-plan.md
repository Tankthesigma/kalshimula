# Hardcore Weather Market Model Plan

Date: 2026-05-24

Status: planning artifact for Codex + Bobby review. No trading automation. No orders. No wallet/portfolio/cancel/execution code. Mainline stays weather-only. Bobby/private lanes may use read-only market data for paper validation.

## Core Thesis

The best model is not a single better daily-high predictor. It is a professional weather-desk stack:

1. Exact settlement station/rule map.
2. Point-in-time live station observations.
3. LAMP/NBM/HRRR/RAP + current model stack.
4. Station-level MOS-style post-processing.
5. Timestamped high/low/hourly probability models.
6. Private Kalshi/Polymarket read-only market archive.
7. Paper-PnL validation by time, city, bucket, side, and platform.
8. Local LLM analyst only for reports/risk flags, never numeric probabilities.

Forecast skill is not market edge. A model must improve weather accuracy and survive private market validation.

## Hard Constraints

- Mainline: no Kalshi API, no Polymarket API, no secrets, no trading code.
- Private audit: read-only market data only; no POST/PUT/PATCH/DELETE to trading endpoints.
- No auto-trading, no order placement, no wallet/bank/portfolio/cancel.
- Generated market data and secrets never committed.
- All model evaluation must be no-leak and timestamp-aware.

## Immediate Hardcore Build Order

### 1. Settlement Station And Rule Map

Build before more modeling. Wrong station labels can manufacture fake edges.

Shared cross-lane interface contract:

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

This table is one shared artifact, not separate Codex/Bobby tables. Bobby's private station validation is the source of truth for Kalshi station truth; mainline consumes the validated rows and keeps the same column names so the private archiver and mainline feature store cannot drift.

Priority audits:

- Chicago: Bobby empirically confirmed Kalshi `KXHIGHCHI` matches `KMDW` Midway, not `KORD` O'Hare. Treat the memo's `KORD` claim as corrected; do not change current Chicago labels to O'Hare.
- Houston: verify exact station before promotion.
- LA: verify station because LA market station naming can be ambiguous.
- NYC: verify `KNYC`, not JFK/LGA.
- Polymarket global markets: do not assume city equals station. Resolve rule text per market.

Current station status from Bobby private audit:

- All 9 private-audit cities matched Kalshi expiration values within +/-0.1F over the checked window.
- Chicago/Austin false positives are therefore genuine model/market weaknesses, not station plumbing bugs.
- Low-temperature markets still need DST/local-standard-time review because settlement windows can matter more near midnight/sunrise.

Acceptance:

- Every promoted city has exact station and settlement source.
- Historical actuals match settlement station better than alternatives.
- Any uncertain station gets `rule_confidence=low` and is blocked from promotion.

### 2. Kalshi Private Archiver Before Full Polymarket

Bobby/private lane should start with Kalshi because current model is US-focused.

Private outputs:

```text
market_metadata
rule_text
orderbook_snapshots
candlesticks
price_history
settlement_outcomes
paper_pnl_grid
```

Polymarket plan:

- Light polling for overlapping or high-liquidity weather markets now.
- Full Polymarket WebSocket archive after station/rule map and international model coverage exist.
- Archive rule text and station source for every Polymarket market because international settlement can differ by city/source/unit/rounding.

Acceptance:

- No private market code in mainline.
- Bobby can replay market state at decision times.
- Paper PnL can be grouped by timestamp, city, side, price bucket, and edge threshold.

### 3. Live ASOS/METAR Observation Store

Mainline weather-only. This is the missing meteorologist layer.

Observation columns:

```text
station_id
obs_ts
available_ts
temperature_f
dewpoint_f
wind_speed_kt
wind_direction_deg
gust_kt
cloud_cover
pressure_mb
precip_flag
raw_metar
source
```

Derived same-day columns:

```text
high_so_far_f
low_so_far_f
temp_1h_slope_f
temp_3h_slope_f
dewpoint_depression_f
hours_since_sunrise
hours_to_solar_noon
hours_to_sunset
radiative_cooling_index
remaining_heating_estimate_f
remaining_cooling_estimate_f
```

Acceptance:

- Feature builder never uses observations after prediction timestamp.
- Backfill and live fetch have same schema.
- Missing observation fields degrade gracefully.
- Exact settlement station is used.

### 4. Timestamped Nowcast Models

Nowcast is mostly defense/risk-veto, not guaranteed edge. It should answer: "Is the market probably right because it has live intraday information?"

High-temp prediction windows:

```text
04:00 local
07:00 local
10:00 local
13:00 local
15:00 local
```

Low-temp prediction windows:

```text
previous 18:00 local
previous 21:00 local
00:00 local
04:00 local
sunrise
06:00 local
```

Hourly-temp windows:

```text
T-6h
T-3h
T-1h
```

Model outputs:

```text
final_high_distribution
final_low_distribution
hourly_temp_distribution
bin_probabilities
nowcast_veto_flag
reason_codes
```

Mainline `nowcast_veto_flag` must be weather-only. It can use observations and model state, for example `high_so_far_f >= forecast_bin_upper`, unusually high remaining-heating uncertainty, missing station observations, or cloud/wind/dewpoint regime risk. It must not use market prices, order books, price movement, liquidity, or private audit labels. The market-confirmed version of this idea lives only in Bobby's private lane.

Acceptance:

- Beats current model on no-leak forecast metrics.
- Also improves private divergent-but-right rate or paper PnL.
- Fails if it only improves MAE by collapsing toward market/consensus and removing profitable divergence.

### 5. LAMP/NBM Direct Ingest

Add professional guidance as inputs and baselines.

Targets:

- NBM MaxT/MinT percentiles.
- LAMP hourly station guidance.
- HRRR/RAP short-range guidance if direct retrieval is feasible.

Acceptance:

- Every guidance row has issue time, valid time, and availability time.
- Raw LAMP/NBM are benchmark baselines.
- Post-processed model must beat raw LAMP/NBM before promotion.

### 6. Structured Probability Model Stack

Numeric probabilities must come from structured models.

Candidates:

- LightGBM/CatBoost/XGBoost for station-level MOS.
- TabPFN as a benchmark on small/medium slices.
- Quantile regression or conformal calibration for intervals.
- Isotonic/beta calibration for bin probabilities.

Required baselines:

- gfs_ens
- openmeteo_naive
- LAMP raw
- NBM raw
- persistence/current observation baseline
- climatology

Acceptance:

- No-leak walk-forward wins on Brier/log loss/ECE.
- Interval coverage holds by city/season.
- Private lane shows paper-PnL or false-positive reduction.

### 7. LLM Weather Desk Analyst

LLM is analyst/reranker/report writer only.

Inputs:

- Structured model probabilities.
- Observation facts.
- Station/rule confidence.
- Source disagreement.
- Private audit labels.
- Known city false-positive flags.

Outputs:

- Morning weather desk report.
- Midday nowcast report.
- Evening low-temp report.
- Skip reasons.
- Confidence label.
- Strongest pro/con facts.

Forbidden:

- LLM cannot create probabilities.
- LLM cannot override calibrated probabilities.
- LLM cannot produce trading instructions.

Acceptance:

- Reports are reproducible from structured facts.
- Probabilities are byte-identical with and without LLM.
- LLM improves skip/review quality in human audit before any fine-tune.

## Evaluation Gates

### Forecast Gates

- Brier score.
- Log loss.
- ECE/calibration.
- MAE.
- CRPS if full distributions exist.
- Interval coverage.
- City/season/time-of-day slices.

### Market-Relative Gates

Private audit must test:

- `model_prob - market_prob`
- YES/NO side separately.
- price buckets: `0.05-0.95`, `0.10-0.90`, `0.15-0.85`, `0.20-0.80`, `0.30-0.70`.
- edge thresholds: `0.02`, `0.03`, `0.05`, `0.08`, `0.10`, `0.15`.
- fees/slippage stress.
- stale price risk.
- time-of-day edge decay.

Promotion is a two-key handshake. A model/city/market type is promoted only when both lanes pass:

1. Mainline weather gates:
   - better no-leak forecast metrics,
   - stable calibration,
   - no station/rule ambiguity,
   - no source duplication artifact.

2. Bobby private market gates:
   - better or safer paper-PnL,
   - improved divergent-but-right rate or false-positive reduction,
   - no one-day/one-city artifact,
   - survives fees/slippage and stale-price checks.

Nothing promotes on MAE alone.

## Known Risks

- Chicago station mismatch hypothesis is closed for current Kalshi high-temp market: `KMDW` is correct. Chicago remains a false-positive risk for model/market reasons.
- Austin/Chicago are known false-positive risks.
- Miami/Phoenix can be traps when market has live heat info.
- hrrr may duplicate gfs_ens in current artifact.
- Polymarket historical depth may not be recoverable after settlement.
- Best MAE source may not be best market edge source.
- LAMP/NBM may improve accuracy but reduce exploitable divergence.

## Codex / Bobby Split

### Codex Mainline

1. Settlement station/rule table.
2. ASOS/METAR observation fetch/store.
3. No-leak feature builder.
4. Source provenance audit.
5. LAMP/NBM ingestion.
6. Structured nowcast/high/low/hourly models.
7. Model-only reports.
8. LLM analyst wrapper, only after structured outputs exist.

### Bobby Private

1. Kalshi station/rule validation from live contract text.
2. Kalshi read-only snapshot/candlestick archiver.
3. Private settlement station mismatch audit.
4. Paper-PnL replay by timestamp.
5. Edge decay by time of day.
6. False-positive city report.
7. Light Polymarket poller for overlapping/global targets.
8. Full Polymarket archive after model coverage expands.

## First Implementation Branch

Recommended branch:

```text
feat/station-rule-nowcast-core
```

First milestone:

1. Station/rule table for current cities.
2. Chicago station mismatch audit.
3. ASOS/METAR current + historical fetcher.
4. Feature builder for high-so-far / low-so-far / temp trend.
5. High-temp nowcast smoke report for current cities.
6. No market APIs in mainline.

Definition of done:

- Ruff and pytest pass.
- Station/rule table exists and is tested.
- ASOS/METAR parser has fixture tests.
- No-leak timestamp tests pass.
- Chicago station conclusion documented.
- Bobby receives the exact station/rule output for private PnL replay.

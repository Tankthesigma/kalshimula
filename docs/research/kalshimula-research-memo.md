# Kalshimula Research Memo — Weather-Market Decision-Support Architecture

**Date:** May 24, 2026
**Scope:** Decision-support architecture only. No auto-trading, no order placement, no execution logic. Mainline stays market-data-free. Private audit lanes may consume read-only public/authorized market data for paper validation.
**Status of every finding below:** paper-trading / decision-support hypothesis until validated on forward live data.

---

## 0. Reading Guide

This memo answers a single question: what is the most defensible path from where Kalshimula is today (calibrated weather model, source breakout, gfs_ens beating openmeteo_naive in no-leak walk-forward) to money-useful, well-calibrated probabilities for Kalshi and Polymarket weather contracts. The current results are real but small. The biggest immediate vulnerability is that the morning-only forecast does not consume the intraday signal the market itself trades on after about 10–11 AM local time. That is the place to attack first.

A consistent thread runs through the rest: **forecast skill ≠ market edge**. The literature on AI weather models is mostly about the first; the literature on prediction markets is mostly about the second; the actual P&L only happens when both are positive at the same time. Sections below try to keep that separation honest.

---

## 1. Executive Summary

**Recommended architecture (hybrid, phased):**

1. **Structured model stack** does all numeric probability work — never an LLM. Specifically: a station-level MOS/LAMP-style post-processor on top of multi-source NWP and AI-weather inputs, with a same-day nowcast layer that ingests live ASOS observations, plus a quantile/conformal calibration head that produces the bin probabilities Kalshi and Polymarket markets actually settle on.
2. **A local LLM analyst layer** sits on top — it does *not* generate probabilities. It reads structured model outputs, the day's observations, and market data, then writes the morning brief, the midday nowcast brief, the evening low-temp brief, skip reasons, and risk flags. This is the role LLMs are reliably good at; it is not the role they are reliably bad at.
3. **The private audit lane** is the only place market data and model output meet. Mainline stays pure weather → probability, never market → probability. The audit lane computes paper P&L, edge vs. market, calibration curves stratified by city/lead-time/price-bucket, and false-positive cities.

**Most important missing data, ranked:**
1. Live, station-level intraday observations (ASOS METAR + 1-min ASOS) tied to the *exact* Kalshi/Polymarket settlement station. This is the single biggest gap. Without it the model is structurally blind after about 10 AM local time.
2. NBM v5.0 probabilistic MaxT/MinT percentile guidance directly — currently you may be approximating it via Open-Meteo consensus.
3. LAMP hourly station guidance for short-lead temperature (this is exactly what professional aviation forecasters use; you are not).
4. Forward-archived Polymarket order book snapshots. Polymarket's `/prices-history` returns only 12+ hour granularity once a market is resolved, so any historical study you don't archive yourself is lost.
5. Source-provenance audit on Open-Meteo `hrrr` vs `gfs_ens` to confirm/deny the duplicate-source artifact.

**Best near-term edge path (this week):**
The current 1.015 MAE gfs_ens result is not a forecasting victory — it is a *market microstructure* observation: when gfs_ens diverges from consensus, the market is likely also reading consensus, so the divergence is the edge. The right next step is not to add another model; it is to **add ASOS-based observation-aware features and re-evaluate at multiple intraday timestamps**. That alone should produce a step change because the morning forecast is currently being scored against an afternoon-settled market that has seen 6+ hours of station temps the model hasn't.

**Highest-risk assumptions in the current artifact:**
- That `hrrr` and `gfs_ens` are independent sources (they are not, per current internal diagnostic — verify upstream).
- That walk-forward MAE on the closing high is a usable proxy for market P&L (it is a necessary but very insufficient condition — see Section 6 on price bucket and edge threshold).
- That Austin and Chicago wins in the diagnostic generalize (private audit already shows they may not).
- That settlement station = forecast city. Kalshi NYC settles on **KNYC (Central Park)**, not LaGuardia/JFK; Chicago settles on **KORD**, not Midway; many APIs default to the wrong station. This is the single largest non-modeling failure mode.

**LLM-predictor vs. structured-model question, short answer:**
Build the structured model. Do not have a local LLM predict temperatures. Use a local LLM as analyst/reranker/report-writer only. Reasoning detail in Section 8.

---

## 2. Literature & Source Review

### 2.1 Professional meteorology stack (what real desks use)

NWS forecasters do not stare at a single model. They use a calibrated, multi-source post-processed product, and for short-lead station-level temperature they use two specific ones:

**LAMP — Localized Aviation MOS Program.** LAMP is a statistical system that provides forecast guidance for sensible weather elements; it updates MOS on an hourly basis for most elements (every 15 minutes for ceiling height and visibility beginning with v2.6), runs on NCEP WCOSS, and provides guidance for over 2000 stations as well as gridded forecast guidance on the NBM CONUS 2.5-km grid out to 38 hours. As of the September 30, 2024 upgrade, LAMP serves as an important input to the NBM in the first 36 hours of the forecast. This is the closest off-the-shelf thing to "professional same-day station-level temperature guidance." Critically, it ingests recent observations as part of its statistical update — exactly the missing piece in the current Kalshimula architecture.

**NBM — National Blend of Models.** A nationally consistent and skillful suite of calibrated deterministic and probabilistic forecast guidance based on a blend of NWS and non-NWS NWP and post-processed model guidance. The operational NBM leverages two forms of bias correction methodologies (decaying average and quantile mapping) along with dynamic decaying MAEs and static expert weights applied on its individual model inputs. NBM gridded guidance is produced for CONUS (2.5 km), Alaska (3.0 km), Hawaii (2.5 km), Puerto Rico (1.25 km), Guam (2.5 km), Oceanic (10 km), and Global (50 km). It combines a wide suite of NWP — GFS, HRRR, RAP, GEFS, and international systems such as the ECMWF IFS — into a calibrated, high-resolution blend, with statistically sophisticated techniques such as MOS, quantile mapping, and ensemble weighting; forecasts are bias-corrected against the URMA analysis of record. NBM was upgraded to version 5.0 on May 5, 2026 — that is current as of this memo. NBM provides QMD-based probabilistic daytime maximum (MaxT) and nighttime minimum (MinT) temperatures in the form of percentiles and exceedance values. This is exactly the variable Kalshi settles on, in exactly the probabilistic form Kalshi market bins want.

**MOS heritage.** The "MOS" idea — regress observed station outcomes against model output features to learn station-specific bias and uncertainty — is from the 1970s. LAMP and NBM are its modern descendants. The Kalshimula plan should treat MOS-style post-processing as the *baseline*, not as an option.

### 2.2 AI weather models — what they prove and what they don't

The headline results are real:

- Pangu-Weather was the first AI model to cross the IFS threshold, achieving a 5-day Z500 RMSE of 296.7 compared to IFS's 333.7, an approximately 11% improvement, while running more than 10,000 times faster than the operational IFS on a single GPU. GraphCast extended the benchmark, outperforming IFS-HRES on 90% of 1,380 verification targets and producing a 10-day global forecast in under one minute on a single TPU. GenCast became the first AI ensemble system to outperform IFS's operational ensemble (ENS) on 97.2% of 1,320 probabilistic targets, completing a 15-day 50-member ensemble in approximately 8 minutes on a single TPU v5.
- The AIFS (ECMWF's machine-learned forecast model) has been fully operational at ECMWF since 25 February 2025. The current operational version, AIFS 1.1.0, was released on 27 August 2025 to correct a precipitation forecast issue in the initial version.
- The AIFS Single was made operational in February 2025, and the ensemble version (AIFS ENS) was made operational in July 2025.
- Aurora is a 1.3 billion parameter foundation model for high-resolution forecasting of weather and atmospheric processes — a 3D Swin Transformer with 3D Perceiver-based encoders and decoders, pretrained on multiple heterogeneous datasets, then fine-tuned in two stages: short-lead time fine-tuning of pretrained weights, then long-lead time rollout fine-tuning using LoRA.

What they prove: that data-driven models running on initial conditions can match or beat IFS HRES on aggregate global metrics out to medium range, at orders of magnitude lower inference cost.

What they **do not** prove for prediction-market work:
- **They do not prove station-level skill.** WB2 verification is at 1.5° resolution against ERA5, not at KNYC against the NWS CLI. For warm extremes, Pangu-Weather, GraphCast and FuXi tend to be more skillful than IFS HRES within 3 d lead time but become less skillful as lead time increases. At longer lead times, forecasts generated by data-driven models tend to be smoother and less skillful compared to those generated by physical models. Smoothing at lead time is the exact failure mode that would tank station-level high/low prediction, because it shaves the extremes off.
- **They do not prove they are independent from IFS.** Most are initialized from IFS HRES or trained on ERA5, both ECMWF products. Adding three AI models that all condition on IFS does not give you three independent sources — it gives you one source filtered three ways. This is consistent with the empirical finding that Open-Meteo's `hrrr` looks identical to `gfs_ens` in the current artifact.
- **They are still beaten by NWS-style postprocessing on station-specific bias.** Foundation models do not magic away the boundary-layer microclimate at the Central Park station vs. the grid cell.

**Implication for Kalshimula:** AI weather models belong in the input layer as additional source members, but they should be treated *like* GFS or IFS members — feed them into a station-specific post-processor, do not consume them raw.

### 2.3 Post-processing literature

The consistent finding across the recent post-processing literature is that **calibrated probability**, not raw deterministic accuracy, is what matters, and that neural-network post-processors only modestly beat well-tuned classical statistical post-processors (EMOS, MBM) on temperature. Permutation-invariant neural networks that treat ensemble forecasts as a set of unordered member forecasts and learn link functions invariant to permutations of member ordering produce state-of-the-art prediction quality for surface temperature post-processing. For T850, DNN post-processors using only a small fraction of ensemble members outperform the full raw ensemble CRPS; the major source of improvement is a reduction in extreme values (outliers), which are indicative of forecast busts. That last point matters for the Kalshi tail bins (the outer "above X" and "below Y" buckets), where forecast busts are exactly the source of mispricing.

### 2.4 Kalshi & Polymarket — what the contracts actually settle on

**Kalshi (US, CFTC-regulated):**
- All weather market contracts settle based on the final climate report issued by the National Weather Service (NWS), typically released the following morning. Contract prices for temperature ranges shift as new weather data — forecasts, model runs, satellite images, observations — becomes available. The market settles based on the high temperature recorded in the final NWS Daily Climate Report.
- Kalshi typically uses the final NWS Climatological Report (CLI) for the official high temperature that determines the market outcome. Final CLIs record data for a full day, usually from 12:00 AM to 11:59 PM Local Standard Time (LST) for the specific city. The official 24-hour climate reporting period for CLI often remains based on LST, not DST. **This is a hidden trap**: during DST months, the settlement window is **shifted by one hour** from the wall clock most data sources display. A "daily high" market for Houston on June 5 is really a high computed from 1 AM to 12:59 AM Central Daylight Time, because LST = CST is what CLI uses.
- Contracts resolve based on the highest temperature recorded at a specific NWS station — KNYC for NYC (Central Park), avoiding rounding errors from Celsius conversions. Bet limits up to $25,000 per contract.
- Kalshi offers daily weather markets across multiple cities with 6 brackets — the middle 4 are 2°F wide, the 2 edges contain everything under or over the listed temperatures. Markets launch at 10 AM the previous day.
- Kalshi runs structured multi-bracket markets for more than a dozen American cities including daily high and low temperatures, monthly snowfall, and monthly rainfall along with longer-horizon contracts.
- For monthly precipitation markets the underlying is the total monthly precipitation (in inches) recorded at the specified NWS weather station; expiration time is 10:00 AM ET; and the latest Expiration Date is 15 days after the end of the month, accelerated if the monthly total is released early.

**Kalshi API** (`https://external-api.kalshi.com/trade-api/v2`):
- Order book endpoint returns yes bids and no bids only (no asks). In binary markets, a bid for yes at price X is equivalent to an ask for no at price (100-X); a yes bid at 7¢ is the same as a no ask at 93¢, with identical contract sizes.
- Historical market candlesticks are available, with valid time period lengths of 1 minute, 60 minutes, or 1440 minutes (1 day). Each candle has yes_bid, yes_ask, price (open/low/high/close/mean/previous), volume, and open_interest.
- A November 27, 2025 release added `GET /markets/candlesticks` for batch retrieval of up to 10,000 candlesticks total across multiple markets in a single call. This is the right endpoint for the private audit pipeline — one call replaces N.

**Polymarket (decentralized, USDC, global cities):**
- The Weather category hosts 115 markets covering subcategories like Temperature, Pandemics, and Global. Outcomes priced 0–100¢; "Highest temperature in London on May 24?" and "Highest temperature in Seoul on May 24?" are examples of currently-listed markets.
- Polymarket lists temperature markets globally — examples include "Lowest temperature in Tokyo on May 24?" and "Lowest temperature in Shanghai on May 24?".
- Architecture separates Gamma API (market metadata/discovery), CLOB API (central limit order book — current prices, order book snapshots, historical price information), and Data API (user-specific data). Outcome token prices are essentially probabilities — a price of 0.65 for "Yes" means the market believes there's a 65% chance of "Yes".
- Gamma API is at gamma-api.polymarket.com — events, markets, tags, series, sports data, no auth required. Price history is via CLOB API's /prices-history endpoint with token_id and time range. General limit 4,000 requests per 10 seconds; /events 500/10s, /markets 300/10s.

**Polymarket data limitation — this is the most important single thing for archiving:**
- The /prices-history endpoint only returns historical data at 12+ hour granularity for resolved/closed markets, even for extremely high-volume events where price fluctuations definitely occurred at finer intervals. **Conclusion: any sub-12-hour Polymarket price history not archived before settlement is gone**. The private audit lane must run a persistent WebSocket archiver for every active weather market starting Day 1; there is no retroactive remedy.

### 2.5 Data infrastructure

- Open-Meteo Historical Forecast API is a continuous hourly timeseries built by stitching the first hours of each successive model run, with coverage from around 2021. Previous Runs API archives the same models at a fixed lead-time offset (1–7 days), with data from January 2024 (GFS from March 2021, JMA from 2018).
- Open-Meteo Single Runs API allows retrieval of any individual archived model run by initialization time — ECMWF IFS HRES from March 14, 2024 (IFS Cycle 49R1 hindcasts), all other models from September 2025.
- IEM (Iowa Environmental Mesonet) provides one-minute ASOS data from many US sites back to 2000 (sourced from NCEI's archive but in a more usable format).
- MADIS handles 1-minute ASOS data; the live MADIS feed processes current and previous hour's data every 5 minutes, with data arriving on a continuous asynchronous schedule. This is the live nowcast channel.

### 2.6 Diurnal cycle physics (necessary because the market is timed by it)

- Daily maximum temperature generally occurs between 2 PM and 5 PM and then continually decreases until sunrise the next day. The angle of the Sun to the surface increases until around noon when the angle is largest (sunlight most direct).
- From April to October, when incoming shortwave radiation dominates over longwave cooling, maximum temperature and the diurnal ranges of temperature and relative humidity increase with decreasing opaque cloud cover, while minimum temperature is almost independent of cloud. During the winter period, both maximum and minimum temperature fall with decreasing cloud, as longwave cooling dominates over the net shortwave flux. A few hours after sunrise, there is a transition when the nighttime stable layer is eroded by surface heating.
- Minimum daily temperature generally occurs substantially after midnight, during early morning in the hour around dawn, since heat is lost all night long. Peak daily temperature generally occurs after noon, as air keeps absorbing net heat from morning through noon and some time thereafter.

**Operational implication:** The morning Kalshimula run timed at 4–6 AM cannot see the high (which has not happened yet); it also probably cannot see the low (which sometimes happens right at sunrise, sometimes earlier). The clean prediction windows are:

| Market | Best prediction window | Why |
|---|---|---|
| Daily high | 04:00, 07:00, 10:00, 13:00, 15:00 local | After 13:00 you start observing the realized high candidate; after 15:00 you're mostly settled |
| Daily low | previous 18:00, previous 21:00, 00:00, 04:00, 06:00 local | The low candidate appears between midnight and sunrise; pre-dawn observation is highly diagnostic |
| Hourly temp | T-6h, T-3h, T-1h | HRRR/RAP/LAMP dominate; same-day observations dominate beyond the model |

---

## 3. Data-Source Priority List

Ranking by **(usefulness for Kalshi/Polymarket weather edge) × (latency / reliability)**, divided by implementation friction. The market-data items are private-audit only and never touch mainline.

### Tier 0 — must-have, build this first
| Source | Use | Cost | Latency | Notes |
|---|---|---|---|---|
| **IEM ASOS METAR archive + live** | Live obs nowcast layer; current high/low so far, dew point, wind, clouds. Free HTTP GET API at mesonet.agron.iastate.edu/cgi-bin/request/asos.py | Free | ~5–15 min | The single biggest current gap |
| **NWS CLI Daily Climate Reports** | Settlement ground truth for Kalshi. Must match the **exact** station Kalshi uses (KNYC, KORD, KMIA, KAUS, KLAX, KDEN, KPHL, etc.) | Free | Next morning | Use this, not paper forecasts, as the training label |
| **NOAA NBM v5.0 on AWS** | NOAA National Blend of Models registry of open data on AWS — primary calibrated probabilistic baseline. Provides MaxT/MinT percentile guidance directly | Free | ~3 h after run | Currently approximated by Open-Meteo consensus; should be ingested directly |
| **NOAA LAMP station guidance** | Hourly station-level forecast guidance — your 1–25h nowcast spine. Updates hourly for most elements (every 15 min for C&V), provides guidance for over 2000 stations | Free | ~30 min after run | The closest off-the-shelf "professional desk" product for hourly temp |
| **HRRR (NOAA AWS bucket)** | 3-km resolution, hourly updated, cloud-resolving, convection-allowing. 0–18h spine for hourly markets | Free | ~1 h after run | Currently in Open-Meteo but should be verified as independent of GFS — current artifact suggests it isn't |

### Tier 1 — high-leverage, build next
| Source | Use | Cost | Notes |
|---|---|---|---|
| **ECMWF IFS HRES + ENS open data** | Fully open under CC-BY-4.0 from 1 October 2025, full resolution, no data cost. The strongest independent source vs. NOAA models | Free | 9 km native or 0.25° open |
| **ECMWF AIFS open data** | Real-time open data released as soon as the forecast is produced; 1-hour delay was removed following operational implementation of AIFS Single v1 on 25 February 2025 | Free | Treat as another member, not as independent of IFS |
| **NWS station metadata** | Station ID → lat/lon, elevation, equipment quirks, sensor history | Free | The KAUS = Austin-Bergstrom, KMIA = Miami Intl, etc. mapping table is the contract-station integrity layer |
| **Open-Meteo Single Runs API** | Access any individual archived model run by initialization time — ECMWF IFS from March 2024, all other models from September 2025 | Free up to 10k/day | Critical for no-look-ahead backtesting; **this is what should anchor your reforecast training set** |
| **ERA5 reanalysis** | 0.25° (~25 km) from 1940, ERA5-Land at 0.1° (~9 km) from 1950 | Free | Climatological baseline + ML pretraining for any custom model |

### Tier 2 — for global expansion / Polymarket
| Source | Use | Notes |
|---|---|---|
| **METAR worldwide via IEM** | The IEM maintains an ever-growing archive of automated airport weather observations from around the world. Same API, different `network=` param | Coverage is excellent for major airports globally; thin for inland Africa, parts of Central Asia |
| **JMA AMeDAS, ECMWF Set IX, KMA, BoM** | National-service station and model data | Most accessible via Open-Meteo's national-service feeds |
| **Polymarket Gamma + CLOB** | Market metadata, current order books, live WebSocket | Gamma is fully public; CLOB uses wallet-based authentication for trading but read access does not require it |
| **Kalshi public API** | Public market data endpoints at https://external-api.kalshi.com/trade-api/v2 without authentication for series/events/markets/orderbook | Read-only mode for private audit. Mainline never calls this |

### Tier 3 — premium / experimental
| Source | Use | Notes |
|---|---|---|
| **GenCast / GraphCast / Aurora weights** | Run locally on GPU for additional ensemble members | Useful primarily as an *additional postprocessing input*, not as the final prediction. Watch for IFS dependency |
| **Commercial vendors (DTN, AccuWeather Enterprise, Tomorrow.io)** | Pre-bias-corrected station forecasts | Only buy if a clean A/B vs. NBM+LAMP+ASOS shows real PnL lift on paper |
| **GOES-R satellite / MRMS radar** | Cloud/precip nowcast features (>6h horizon, hourly markets) | Adds material lift on rain markets, marginal lift on temp |

---

## 4. Model Architecture Recommendation (Phased)

### Architecture overview

```
                ┌─────────────────────────────────────────────────────┐
                │  MAINLINE (market-data-free)                        │
                │                                                     │
   NWP +AI ─┐   │   ┌──────────────┐    ┌──────────────────┐         │
   sources  ├──►│──►│ Source-level │───►│ Station MOS/LAMP │──┐      │
            │   │   │ ingest +     │    │ post-processor   │  │      │
   ASOS    ─┤   │   │ provenance   │    │ (per city/season)│  │      │
   live obs │   │   └──────────────┘    └──────────────────┘  │      │
            │   │           │                                  │      │
            │   │           ▼                                  ▼      │
            │   │   ┌──────────────┐    ┌──────────────────────────┐ │
            │   │   │  Nowcast     │───►│ Calibrated prob model    │ │
            │   │   │  layer       │    │ (quantile/conformal)     │ │
            │   │   │  (live obs)  │    │ → bin probabilities      │ │
            │   │   └──────────────┘    └────────────┬─────────────┘ │
            │   │                                    │               │
            │   │                                    ▼               │
            │   │                          ┌─────────────────────┐   │
            │   │                          │ Local LLM analyst   │   │
            │   │                          │ (report writer +    │   │
            │   │                          │  risk flagger,      │   │
            │   │                          │  no numbers)        │   │
            │   │                          └─────────┬───────────┘   │
            │   └────────────────────────────────────┼───────────────┘
            │                                        │
            │                                        ▼
            │                       ┌──────────────────────────────┐
            │                       │ PRIVATE AUDIT LANE           │
            │                       │ (read-only market data only) │
            │                       │ Kalshi/Polymarket prices ──► │
            │                       │ → edge, paper P&L, calib     │
            │                       │   curves stratified by city, │
            │                       │   season, lead, bucket       │
            │                       └──────────────────────────────┘
            │
            └── Note: nothing here is order placement, allocation, or
                bankroll-management code. Audit produces *labels* on
                model outputs ("model said X, market said Y, settled Z").
```

### Phase 1 — Live observation & nowcast layer *(weeks 1–3)*

This is the highest-ROI item. Build before doing anything else.

- Pull IEM ASOS METAR archive for every Kalshi settlement station back to at least 2018.
- Establish a per-station live ingest, fetch every 10 minutes during market hours.
- Engineer the observation features the diurnal-cycle physics demands:
  - `high_so_far`, `low_so_far`, `temp_at_t`
  - `dew_point`, `temp_minus_dewpoint` (sets the radiative-cooling floor for lows)
  - `cloud_cover_oktas`, `wind_speed_at_t`, `pressure_trend_3h`
  - `solar_zenith`, `hours_until_solar_noon`, `hours_since_sunrise`
  - `radiative_cooling_index` for lows (clear+calm+low dewpoint = strong cooling)
  - `1h_temp_trend`, `3h_temp_trend`
- Add a same-day re-prediction step at 04:00, 07:00, 10:00, 13:00, 15:00 local — these are different *models* trained on those exact time-slices of feature availability, not the same model called five times. (Treating them as the same model leaks information.)

**Kill criterion for Phase 1:** if the 13:00 nowcast is not at least 0.3 °F MAE better than the 04:00 morning prediction on the held-out 2024–2025 season, the live-obs feature engineering is wrong — fix before moving on.

### Phase 2 — High/low temperature probability model *(weeks 3–6)*

This is the structured-model heart of Kalshimula. **Recommended baseline:**

- **Inputs**: NBM percentile guidance (when ingested directly) + GFS ensemble + EC ENS + AIFS ENS + ICON ENS + GEM ENS + GraphCast/Pangu/Aurora deterministic members + LAMP station guidance + Phase-1 live-obs features + station climatology (per-station per-day-of-year mean/std).
- **Post-processor**: per-station, per-season gradient boosting (LightGBM, with monotone constraints where they apply — e.g. `temp_at_15:00` should be monotone in eventual high). Output is mean + standard deviation of a Normal or Student-t.
- **Calibration head**: per-station per-season quantile regression *or* conformal prediction wrap. Conformal is strongly recommended for the bin probabilities Kalshi uses, because it gives *guaranteed marginal coverage* on the training distribution, which matters more for market calibration than raw CRPS optimization. Conformal/quantile post-processing produces state-of-the-art prediction quality for surface temperature when calibration and sharpness are evaluated rigorously.
- **Output to mainline**: bin probabilities for the Kalshi 6-bracket layout and equivalent for Polymarket multi-outcome layout.

**Why not just deep learning end-to-end here:** Per the post-processing literature, well-tuned classical/GBM methods beat or tie neural networks on per-station temperature when the dataset is the size you have (a few years × a few hundred stations × daily, i.e. tens of thousands of rows). TabPFNv2 is a foundation model that yields dominant performance for datasets with up to 10,000 samples and 500 features — try TabPFN as a parallel head and let it duke it out with LightGBM. Production picks whichever wins on out-of-sample CRPS *and* paper PnL.

**Kill criterion for Phase 2:** if the calibrated bin probabilities do not have Brier score better than (i) Open-Meteo consensus, (ii) NBM percentiles, and (iii) climatology, in walk-forward on a held-out 2024–2025 season, the model is not yet ready for the private audit lane.

### Phase 3 — Hourly temperature model *(weeks 6–8)*

Hourly markets are likely *more* efficient than daily markets because the underlying signal (HRRR/RAP + recent obs) is more standardized. Edge here probably comes from:
- The ~1 hr after a sharp frontal passage when the market has not yet repriced.
- Stations with persistent micro-bias that the public forecast doesn't correct.
- Sunset/sunrise transitions where bulk cooling/heating rate is highly station-specific.

Architecture: HRRR/RAP + LAMP + most recent 6h of ASOS observations + diurnal-cycle features → per-station per-hour-of-day quantile regression. LAMP provides hourly station guidance and updates every 15 minutes for some elements, so the natural cadence is 15-minute refresh.

**Honest expected outcome:** the hourly product is the most likely to *not* produce paper PnL because the markets are thin and the public LAMP product is already exploiting most of the structural edge. Plan it but expect to use it primarily as an *input* to the daily-high model rather than a standalone product.

### Phase 4 — Cross-market Kalshi/Polymarket private audit *(weeks 4–8, in parallel with Phase 2/3)*

- Persistent WebSocket archiver for every active Kalshi weather market and every Polymarket weather/climate market — sub-second snapshots, dumped to Parquet. **Start this Day 1**; you can never recover sub-12h Polymarket history retroactively.
- Read-only Kalshi candlestick puller for historical price series (1-min granularity available).
- Per-market post-mortem: for each settled market, store (model_prob_at_t for each t, market_prob_at_t for each t, settled_outcome). This is the single dataset the audit lane needs.
- Cross-platform consistency check: for shared cities (currently very few — Polymarket has more international, less US), compare same-day implied probabilities. If they diverge, that's *information* about which side has stale liquidity.

### Phase 5 — Local LLM weather-desk analyst *(weeks 8–12)*

The LLM never sees `np.array` of probabilities and outputs another. It sees a structured "brief packet" and writes prose. Specifically:

**Inputs to LLM** (all formatted as JSON or markdown):
- The structured model's bin probabilities for today's markets in each city.
- A "facts sheet" of the strongest pro-Yes and pro-No observations (high_so_far, dew_point, cloud_cover, expected heating curve from LAMP, NBM 90th percentile, etc.).
- Yesterday's analyst report and yesterday's settlement (for continuity).
- Top diagnostic flags ("`hrrr` and `gfs_ens` agree within 0.2°", "live obs already higher than 60% of the model bin's lower edge", "low temperature already realized at 04:23 local").

**Outputs from LLM**:
- Morning brief (one paragraph per city).
- Midday nowcast brief (what's changed; which bins moved; which moved for *physical* vs. *model* reasons).
- Evening low-temp brief.
- Skip reasons per market ("station mismatch suspected"; "current LST handling ambiguous because of DST transition"; "cloud cover at 1500 wildly different from model").
- Confidence label (low/med/high) — categorical, not numeric.

**Model choice:** start with Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct, deployed locally with vLLM. Fine-tuning is **not** required to get the report-writer to be useful; a good system prompt + few-shot is enough for v0. Fine-tuning becomes worth it (Phase 6) only when you have ≥1000 expert-edited reports as training pairs.

### Phase 6 — Optional fine-tuned / foundation model *(month 3+)*

Only worth it when both are true:
1. The Phase 2 structured model has plateaued and Brier/PnL improvements are diminishing.
2. You have a clear hypothesis for what additional capacity could capture — e.g., joint cross-city features (heat dome correlation across Phoenix/LA/Vegas/Houston/Dallas/Austin) that single-station GBMs don't see.

In that case, candidates are:
- **Aurora fine-tune** for high-resolution regional weather, then post-process to stations. Aurora can be fine-tuned for diverse applications at modest expense — 4–8 weeks with a small engineering team per task. The fine-tune dataset is your station-level ASOS observations.
- **Custom GNN** over the station network, treating Kalshi/Polymarket cities as nodes with shared environmental drivers.

Neither of these should be done before Phase 2 has been pushed to its limit.

---

## 5. Experiment Design (No-Leak, Timestamped)

### 5.1 General rules

1. **Every feature has a timestamp.** Train/inference uses only features knowable *at or before* the prediction time. No `mean_of_today_observations` features anywhere. A practical pattern: a `as_of_ts` column on the feature store and a query that *forbids* joining anything with `valid_ts > as_of_ts`.
2. **Model runs are dated by ingest time, not init time.** A 00z GFS run is not "knowable at 00z"; it's knowable at the actual NCEP availability time of about 04–06 UTC. Use Open-Meteo's documented availability schedule.
3. **Cross-validation is walk-forward by *calendar day*, not by random split.** Same-day rows from the same station are highly correlated.
4. **Use Open-Meteo Single Runs API for the reforecast training set** to avoid the stitching artifact in the Historical Forecast API.
5. **Two held-out splits**: most-recent 90 days (paper-trade evaluation) and a *seasonal* split (held-out winter to test summer-trained model, etc.).

### 5.2 Prediction-time grid

**For daily highs:** predict at 04, 07, 10, 13, 15 local.
- 04: morning baseline; you have all overnight obs, last night's NBM/EC/GFS run.
- 07: morning rush; new 12z GFS run starting to be available.
- 10: market launches; previous day's settlement known; new HRRR/RAP runs available.
- 13: critical — observations from morning available; afternoon heating curve still ahead.
- 15: most days, peak temperature has now occurred or is occurring; this is the strongest theoretical edge window.

**For daily lows:** predict at 18 (previous evening), 21, 00, 04, 06 local.
- 18: previous evening, before nocturnal cooling onset.
- 21: cooling rate now observable.
- 00: midnight check.
- 04: critical — radiative cooling has done most of its work; dewpoint floor visible.
- 06: most days, low has occurred; the only risk is a post-sunrise undercut, which is rare except for clear-air cold-air drainage cities.

**For hourly markets:** predict at T-6, T-3, T-1.

### 5.3 Evaluation stratification

Always report metrics stratified by:
- City / station.
- Season (DJF, MAM, JJA, SON).
- Lead time (hours from prediction to settlement).
- Market type (high / low / hourly / rain / snow).
- Side (Yes/No).
- Price bucket at prediction time (0.05–0.95 / 0.10–0.90 / 0.15–0.85 / 0.20–0.80 / 0.30–0.70).
- Edge threshold (0.02 / 0.03 / 0.05 / 0.08 / 0.10 / 0.15).
- Time of day at prediction.

### 5.4 Metrics

- **Forecast metrics** (mainline): Brier score, log loss, expected calibration error (ECE), MAE on point forecast, interval coverage at 50%/80%/90%, CRPS where you produce a full distribution.
- **Market metrics** (private audit): paper PnL net of estimated fees and slippage, hit rate by edge threshold, Sharpe of daily PnL, max drawdown, false-positive city rate (cities where edge looked positive in cross-val but flipped sign in held-out).

### 5.5 Kill criteria

Hard stops on the system that would otherwise overfit:
- **Calibration drift kill**: if ECE on the last 30 forward days exceeds 0.05 for any market type, stop trading that market type in audit until refit.
- **Market-data leakage kill**: if any feature can be predicted by the day's *prior* market price with R² > 0.5 (excluding climatology), it's almost certainly leaking the market — drop it.
- **Source-collapse kill**: if the model's effective number of independent inputs (1 / sum of variance-weighted correlation squared) drops below 3, you've stacked correlated models — pause and audit provenance. This is exactly the `hrrr` ≡ `gfs_ens` failure mode.
- **Walk-forward sign-flip kill**: any city where cross-val PnL was positive but held-out PnL is negative gets paused for that season.

---

## 6. Market / PnL Research Plan (Private Audit)

### 6.1 The core question the audit answers

**Is `model_prob - market_prob` predictive of `settled_outcome - market_prob`?** That's it. Everything else is a slice of that question.

The audit lane builds a tall table:
```
market_id, ts, city, station, market_type, lead_time_h,
model_prob, market_prob, model_minus_market,
yes_price, no_price, yes_bid, yes_ask,
fees_estimated, slippage_estimated,
settled_outcome, settled_temp
```
…and computes paper PnL grouped every way Section 5.3 lists.

### 6.2 Key sub-questions

1. **Direction matters more than magnitude at first.** Is `sign(model_minus_market)` predictive of `sign(settled_outcome - market_prob)`? If yes, the model has *some* edge. If no, more sophisticated PnL analysis is wasted effort.
2. **YES vs NO asymmetry.** Buying Yes and buying No are not identical economically because fees and stale-quote risk differ. Run both as separate strategies in the audit and only conclude there's edge if at least one side is robust across stratifications.
3. **Price-bucket edge.** Expect strongest edge in the 0.20–0.80 range — outside that the percentage edges look big but the absolute edges in cents are small and fees dominate. Test all of [0.05–0.95], [0.10–0.90], [0.15–0.85], [0.20–0.80], [0.30–0.70] separately.
4. **Edge threshold sweep.** At thresholds of 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, how does paper PnL behave? Expect a typical pattern: low thresholds = lots of trades, near-zero per-trade PnL, fees eat everything; high thresholds = few trades, high variance. The "sweet spot" is usually 0.05–0.08 for liquid markets and 0.10+ for illiquid.
5. **Fees + slippage modeling.** Kalshi fees vary by market and contract price (small per-trade fee + settlement fees). Slippage on weather markets at peak hours can be 1–3¢ on the bracket bins; estimate from the order book itself.
6. **Stale price risk.** A "good edge" at 4 AM may evaporate by 10 AM when the market reprices. Always score edge against *current* book, not last-trade price, and time-decay model probabilities if obs come in.
7. **Intraday market efficiency.** Plot model edge vs. time-of-day. If edge consistently shrinks toward zero by mid-afternoon, the market is doing its job and the structural edge is in the 4–10 AM window — i.e., predicting at a time before the market has the same information. If edge stays positive into mid-afternoon, you have something the market is not pricing.
8. **False-positive cities.** Austin and Chicago were already flagged. For each candidate city, require positive paper PnL across at least two non-overlapping held-out periods *and* two of {high market, low market, hourly market} before promoting to production lane.
9. **Cross-platform differences.** A city listed on both Kalshi and Polymarket may show divergent implied probabilities. The audit should log both and check whether convergence trades (one side without the other) would have been profitable — *for research*, no order routing.

### 6.3 What the audit does **not** do

- It does not place orders.
- It does not allocate capital.
- It does not run a Kelly sizer or any sizing function. (Sizing is a P&L policy decision, not a research output.)
- It does not feed market data into mainline. The audit is a one-way valve.

---

## 7. City Expansion Strategy

Tiered by how easy the data, station, and settlement story are. Within each tier, rank by liquidity × calibration tractability.

### Tier 1A — current US cities, refine first
NYC (KNYC, Central Park), Chicago (KORD), Miami (KMIA), Los Angeles (KLAX or downtown depending on contract), Austin (KAUS), Houston (KIAH or KHOU — confirm), Phoenix (KPHX), Philadelphia (KPHL), Boston (KBOS), Denver (KDEN). These are the cities with the most history and the most Kalshi liquidity.

### Tier 1B — stable maritime / mild-climate US (high calibration tractability)
San Francisco (KSFO/KSFOC1 — note micro-climate split), San Diego (KSAN), Seattle (KSEA), Portland (KPDX). Low diurnal range = low variance = tighter bins = harder market but easier model.

### Tier 2 — high-variance US cities (high-edge potential, harder calibration)
**Desert (large diurnal range, strong radiative effects):** Phoenix, Las Vegas (KLAS), Tucson (KTUS), Albuquerque (KABQ). Maxes are easier; mins are extremely sensitive to cloud cover late in the night.
**Humid / thunderstorm-prone:** Houston, New Orleans (KMSY), Atlanta (KATL), Tampa (KTPA). Cloud and storm timing destroys MAE; not great for the model but liquid markets.
**Mountain / elevation-sensitive:** Denver, Salt Lake City (KSLC), Reno (KRNO). Microclimate inversions; need a *station-specific* MOS-style correction more than anywhere else.

### Tier 3 — Polymarket international cities
Per Polymarket category coverage examples: London and Seoul (highest temperature) on May 24, Tokyo and Shanghai (lowest temperature) on May 24. Other likely candidates based on standard global METAR coverage: Paris, Berlin, Madrid, Rome, Amsterdam, Singapore, Hong Kong, Dubai, Mexico City, São Paulo, Sydney, Mumbai, Delhi.

Add in this order, easiest to hardest:

| City | METAR station | Data tractability | Settlement risk |
|---|---|---|---|
| London | EGLL (Heathrow) or EGWU | Excellent — IFS native | Confirm Polymarket station before any audit |
| Tokyo | RJTT (Haneda) or RJAA (Narita) | Excellent — JMA AMeDAS + Open-Meteo | Confirm which station |
| Singapore | WSSS (Changi) | Excellent | Tight diurnal range, tight bins |
| Paris | LFPG or LFPO | Excellent | Climate similar to mid-Atlantic US |
| Berlin | EDDB or EDDT | Excellent | Same |
| Sydney | YSSY | Good (BoM) | Southern Hemisphere season inverted — re-train per-season |
| Shanghai | ZSPD (Pudong) or ZSSS (Hongqiao) | Good — CMA via Open-Meteo | Watch which station settles |
| Mexico City | MMMX | Good | High elevation = unusual radiation balance |
| Dubai | OMDB | Good | Extreme summer; small daily variation makes bins narrow |
| Mumbai | VABB | Moderate — monsoon discontinuities | Settlement during monsoon is volatile |
| Delhi | VIDP | Moderate — high pollution affects obs | Use METAR cautiously |
| São Paulo | SBSP or SBGR | Moderate | Less Open-Meteo HRRR coverage |

### Tier 4 — hourly-friendly cities
Hourly markets reward stable, predictable diurnal cycles with minimal weather noise. Best candidates: San Diego, Honolulu (PHNL), Miami in winter, Singapore. Worst candidates for hourly: anywhere with regular afternoon thunderstorms or sea-breeze front passages.

### Data gating before promoting a city to production
1. ≥3 years of ASOS history at the *exact* settlement station.
2. NBM probabilistic guidance available (CONUS) **or** demonstrated equivalent post-processing on regional NWP (international).
3. Station identity verified against contract terms — text scrape + manual review.
4. Two non-overlapping held-out seasons with positive paper edge.
5. Audit-lane false-positive check passed.

---

## 8. LLM vs. Structured Model — Direct Answer

**Should we train a local 2B/3B/7B model to be a meteorologist?**
No, not in the sense of having it output numeric probabilities or temperatures. The technical literature is consistent: LLMs are *not* calibrated numerical predictors out of the box, they tokenize numbers in ways that destroy locality, and on tabular numeric prediction tasks small/medium specialized models (XGBoost, LightGBM, TabPFN, neural post-processors) dominate them at a fraction of the inference cost. TabPFN outperforms ensembles of the strongest baselines (including gradient-boosted decision trees) tuned for 4 hours, in 2.8 seconds, on datasets up to 10,000 samples and 500 features. A 7B LLM is not going to beat that on a numeric regression target that already has well-tuned classical methods.

**Should it predict temperatures directly?**
No. Every wrapper around an LLM that asks "what's the probability the high in NYC tomorrow is 72°F?" is making the LLM do arithmetic and logic the LLM is structurally bad at, when the same prompt re-engineered as "explain why the structured model says 38%" gets you something useful that the structured model alone cannot produce.

**Should it be an analyst / reranker?**
Yes. The role is:
- **Explainer**: take the structured model's bin probabilities and the strongest pro/con observations and write a one-paragraph rationale, with named flags (e.g., "model is high-confidence but NBM 50th percentile disagrees with HRRR by 4°F — note this").
- **Skip-flagger**: detect contexts the structured model is unreliable in (DST transition, station-mismatch suspicion, low data coverage, missing model run, etc.) and write a skip-reason.
- **Cross-source narrative**: when GFS, EC, and AIFS agree but HRRR and the live obs disagree, the LLM is good at writing a paragraph about why. The structured model is bad at that.
- **Optionally, a reranker over priority queue**: given N candidate markets the model is most confident in, the LLM gets the briefs and labels them "lean-yes / lean-no / skip / monitor", *informing* the audit lane's prioritization without ever sizing.

**Training data for fine-tuning (if you go there):**
- ≥1000 morning briefs hand-edited by a (human or expert-pseudo) analyst, paired with the same input "facts sheet".
- ≥500 skip-decisions with annotated reasons.
- Negative examples: 100+ cases where the model was wrong, with post-hoc explanation of what the analyst should have flagged at decision time.

Until you have that dataset, a strong system prompt + 5–10 few-shot examples on Qwen2.5-7B is enough.

**Benchmarks that would prove the LLM is doing useful work:**
1. Audit-lane skip-decision precision — does following the LLM's skip-flags reduce false-positive rate without losing too much PnL?
2. Brief readability rated by a human (you, or another reviewer) on a fixed rubric: "Did the brief surface the strongest pro/con? Did it correctly call the risk flag?"
3. **Strict no-leak check**: the LLM brief never moves probabilities. Validate by running mainline with and without the LLM in the loop — bin probabilities must be byte-identical.

**Failure modes to expect:**
- Hallucinated calibration ("the model is 90% confident" when it isn't). Mitigate with structured input only, no free-text reasoning over numbers.
- Inversion of pro/con under pressure (long context, ambiguous obs). Mitigate with explicit "pro_yes" / "pro_no" fields in input.
- DST/time-zone errors. Mitigate by passing only ISO 8601 timestamps with offset, never human-readable times.
- Forgetting yesterday's settlement. Mitigate by passing yesterday's brief and outcome explicitly.

---

## 9. Implementation Blueprint

A repository-level proposal. New modules in **bold**, existing modules unchanged unless noted.

### 9.1 New modules to add (mainline)

```
kalshimula/
  sources/
    open_meteo_historical.py       # existing
    open_meteo_single_run.py       # NEW — anchor for no-look-ahead backtest
    ecmwf_open_data.py             # NEW — direct IFS+AIFS pull
    nbm_aws.py                     # NEW — NBM v5.0 from AWS bucket
    lamp.py                        # NEW — NCEP LAMP station guidance
    asos_iem.py                    # NEW — METAR + 1-min ASOS via IEM
    asos_live.py                   # NEW — live ingest, 10-min cadence
  features/
    diurnal.py                     # NEW — solar zenith, hours since sunrise
    nowcast_obs.py                 # NEW — high_so_far, dew floor, cooling rate
    source_provenance.py           # NEW — detect duplicated members (hrrr≡gfs case)
  models/
    station_mos.py                 # NEW — LightGBM per station/season
    quantile_head.py               # NEW — conformal-wrapped calibration
    tabpfn_head.py                 # NEW — parallel head, A/B vs LightGBM
  inference/
    morning.py                     # existing, refactored to schedule
    midday.py                      # NEW — 10/13/15 local nowcast runs
    evening_low.py                 # NEW — 18/21/00/04/06 low runs
    hourly.py                      # NEW — T-6/T-3/T-1 hourly runs
  reports/
    brief_packet.py                # NEW — structured "facts sheet" for LLM
    llm_analyst.py                 # NEW — local vLLM call, prose only
  diagnostics/
    walk_forward.py                # extend with stratification
    source_independence.py         # NEW — variance-weighted correlation check
```

### 9.2 New modules in the private audit lane (separate repo or top-level `audit/` that is gitignored or in private)

```
audit/
  market_data/
    kalshi_read.py                 # read-only candlesticks + orderbook
    polymarket_gamma.py            # read-only metadata
    polymarket_clob_read.py        # read-only prices-history + orderbook
    polymarket_ws_archiver.py      # persistent WebSocket archiver (START DAY 1)
  pnl/
    paper_pnl.py                   # tall table per Section 6.1
    edge_stratify.py               # all the slicings of Section 5.3
    fees_slippage.py               # per-platform fee modeling
  reports/
    audit_brief.py                 # daily P&L stratified report
```

### 9.3 Data schemas

**Mainline `predictions` table:**
```
model_version, run_ts, prediction_ts, valid_for_date, city, station,
market_type ('high'|'low'|'hourly'|'rain'|'snow'),
bin_lower, bin_upper, prob, calibrated_prob,
mean_forecast, std_forecast, q05, q25, q50, q75, q95,
sources_used (list), source_independence_score,
nowcast_features_hash, skip_reason (nullable)
```

**Audit lane `market_snapshots` table:**
```
ts, platform, market_id, ticker, station,
yes_bid, yes_ask, no_bid, no_ask, last_price,
yes_volume, no_volume, open_interest,
implied_prob_yes, implied_prob_no
```

**Audit lane `paper_trades` table** (research labels, never instructions):
```
prediction_ts, market_id, side ('yes'|'no'),
model_prob, market_prob, edge,
hypothetical_entry_price, hypothetical_size,
settled_outcome, hypothetical_pnl, hypothetical_pnl_net_fees,
strategy_label
```

### 9.4 CLI tools

```
kalshimula predict --city NYC --market high --as-of "2026-05-24T15:00:00-05:00"
kalshimula brief   --city NYC --market high --window midday
kalshimula source-audit                       # checks for duplicate sources
kalshimula calibrate --season 2024            # refit per-station MOS
audit pnl --since 2026-01-01 --by city,bucket # private only
audit archive-snapshot                         # cron, every 10 sec
```

### 9.5 Tests

- **Time-leak test**: for every feature builder, assert that feature computed at `as_of=T` is byte-identical to feature computed at `as_of=T+1` using only data with `valid_ts ≤ T`.
- **Station integrity test**: for every Kalshi market, assert the station code in the contract matches the station the model uses.
- **DST test**: every model + report on a DST transition day must explicitly handle LST vs. DST and not silently use local civil time as the settlement window.
- **Provenance test**: for each prediction, the sources-used list must have effective independence ≥ a configurable threshold.
- **No-market-data test (mainline)**: any import that pulls from `audit/` must error. This is enforced in CI.

### 9.6 What stays out of mainline

- Anything in `audit/`.
- Anything that mentions an order, a position, a wallet, a fee, a size, a bankroll, or a P&L.
- Any function that converts a market price into a probability prior. (The audit lane may do this; mainline never does.)

### 9.7 What can be safely committed publicly

- Everything in `sources/`, `features/`, `models/`, `inference/`, `reports/`, `diagnostics/`.
- Configuration, station tables, climatology, calibration coefficients (no market data embedded).
- The `kalshimula predict` CLI and the LLM analyst module.

What should **not** be committed publicly even if it's data-driven:
- Anything in `audit/`.
- Per-city "edge windows" found via market-data backtest. These are commercially sensitive and also at risk of being treated as financial advice.

---

## 10. Final Recommendation

**This week:**
1. Stand up the IEM ASOS live + archive ingest for every current Kalshi settlement station. This is the single biggest missing piece. Aim for ≤15-minute observation latency for each city.
2. Build the source-provenance audit script and confirm/refute the `hrrr ≡ gfs_ens` duplication finding. Either fix the data feed or document it as a known overlap.
3. Add the prediction time grid (04/07/10/13/15 local for highs, evening + dawn for lows) and re-run walk-forward at each grid point separately. Expect 13:00 and 15:00 predictions to dominate.
4. Start the Polymarket WebSocket archiver in the audit lane. Every minute of delay is sub-12h data lost forever.

**Next month:**
5. Direct ingestion of NBM v5.0 probabilistic MaxT/MinT percentiles from the NOAA AWS bucket; replace the Open-Meteo consensus approximation.
6. LAMP station guidance ingest for all current cities.
7. Per-station per-season LightGBM MOS post-processor on top of multi-source NWP + AI-weather + LAMP + live obs.
8. Conformal calibration head producing Kalshi-bin probabilities.
9. Audit-lane paper PnL with stratification per Section 5.3, plus the time-of-day edge decay plot. Expect to learn whether structural edge lives in the morning window or in the afternoon nowcast window.
10. Begin Phase 5 LLM analyst with off-the-shelf Qwen/Llama; no fine-tuning yet.

**Requires paid data or major compute:**
- ECMWF AIFS at native resolution streaming in real time (currently free open-data; full dissemination has SLA tiers).
- Aurora fine-tune for high-resolution regional forecasting — GPU-hours, not dollars, but they add up.
- Persistent NWP archive of multiple model raw outputs (NBM, HRRR, LAMP) for ML training — storage costs scale; estimate ~1–5 TB/year if you keep grib2 native.

**What to avoid:**
- Having a local LLM output bin probabilities directly. (Section 8.)
- Treating WeatherBench 2 wins as evidence the AI weather models can replace a station-specific MOS step. They cannot, yet.
- Building a fine-tuned foundation model before the structured model has plateaued.
- Trusting that Open-Meteo's per-source names map cleanly to independent NWP runs. Verify provenance per source per run.
- Letting any market-data signal seep into mainline forecasting. The cost-benefit is bad: short-term backtest lift, long-term overfitting to market microstructure, and a system that can't claim its probabilities are "weather forecasts" anymore.

**What would make the system meaningfully closer to a professional meteorologist's workflow:**
- Direct LAMP and NBM ingest. (NWS desks literally start their day with these.)
- Per-station seasonal MOS coefficients refreshed regularly.
- A live-obs nowcast layer that updates the forecast as observations come in.
- A skip-decision layer that knows when *not* to predict (frontal passage, sea breeze, DST transition, station outage).
- A morning + midday + evening brief written by an analyst layer.

Items 1, 3, and 4 are all in the proposed Phase 1+2; items 2 and 5 are in Phase 5. Once those are in place, the gap between Kalshimula and "professional weather desk" narrows to two things: (a) human experience reading synoptic charts (which a foundation model fine-tune in Phase 6 can partially fill in), and (b) live access to internal NWS regional discussions (which are public and can be ingested).

---

## Appendix A — Source Map

Primary citations used in this memo:

**Meteorology / NWS infrastructure**
- NOAA MDL Virtual Lab — LAMP: https://vlab.noaa.gov/web/mdl/lamp
- NOAA MDL — LAMP & GLMP September 2024 upgrade: https://vlab.noaa.gov/web/mdl/-/mdl-lamp-and-gridded-lamp-glmp-upgraded-at-nws
- NOAA MDL — NBM v5.0 (May 2026): https://vlab.noaa.gov/web/mdl/nbm
- NOAA MDL — NBM v4.3 (May 2025): https://vlab.noaa.gov/web/mdl/-/nbm-upgraded-to-version-4.3
- NOAA NBM AWS Open Data: https://registry.opendata.aws/noaa-nbm/
- NOAA NOMADS NBM description: https://nomads.ncep.noaa.gov/txt_descriptions/BLEND_txt.html
- HRRR overview (NOAA GSL): https://rapidrefresh.noaa.gov/hrrr/
- HRRR AWS Open Data: https://registry.opendata.aws/noaa-hrrr-pds/
- HRRR Part I (J. Wea. Forecasting 2022): https://journals.ametsoc.org/view/journals/wefo/37/8/WAF-D-21-0151.1.xml

**Observation infrastructure**
- IEM ASOS network overview: https://mesonet.agron.iastate.edu/ASOS/
- IEM 1-minute ASOS download: https://mesonet.agron.iastate.edu/request/asos/1min.phtml
- IEM ASOS/METAR CGI API: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help
- IEM API documentation: https://mesonet.agron.iastate.edu/api/
- MADIS 1-minute ASOS: https://madis.ncep.noaa.gov/madis_OMO.shtml

**ECMWF / AI weather**
- ECMWF open data transition (Oct 2025): https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwf-achieve-fully-open-data-status-2025
- AIFS 1.1.0 update (arXiv 2509.18994): https://arxiv.org/html/2509.18994v1
- AIFS Single v1 operational (ECMWF Confluence): https://confluence.ecmwf.int/display/fcst/implementation+oF+Aifs+Single+v1
- AIFS open data: https://ecmwf.int/en/forecasts/dataset/aifs-machine-learning-data
- Aurora (Microsoft Research): https://www.microsoft.com/en-us/research/project/aurora-forecasting/
- Aurora intro: https://microsoft.github.io/aurora/intro.html
- WeatherBench 2 (arXiv 2308.15560): https://arxiv.org/pdf/2308.15560
- WeatherBench 2 hydroclimatic extension (GMD 2025): https://gmd.copernicus.org/articles/18/5781/2025/
- Pangu / GraphCast extreme comparison (GMD 2024): https://gmd.copernicus.org/articles/17/7915/2024/

**Post-processing literature**
- ANET2 ensemble post-processing (QJRMS 2024): https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4809
- ML post-processing of near-surface temperature (QJRMS 2024): https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4613
- Permutation-invariant NN post-processing (AIES 2024): https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0070.1.xml

**Tabular ML**
- TabPFN (Nature 2025): https://www.nature.com/articles/s41586-024-08328-6
- TabPFN time-series (arXiv 2501.02945): https://arxiv.org/html/2501.02945v2

**Open-Meteo**
- Features: https://open-meteo.com/en/features
- Historical Forecast API: https://open-meteo.com/en/docs/historical-forecast-api
- Single Runs API: https://open-meteo.com/en/docs/single-runs-api
- Previous Runs API: https://open-meteo.com/en/docs/previous-runs-api
- Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api
- Ensemble API: https://open-meteo.com/en/docs/ensemble-api
- ECMWF API: https://open-meteo.com/en/docs/ecmwf-api
- Self-host open-data: https://github.com/open-meteo/open-data

**Kalshi**
- Weather markets help (settlement details): https://help.kalshi.com/markets/popular-markets/weather-markets
- Kalshi API quick start: https://docs.kalshi.com/getting_started/quick_start_market_data
- Markets / candlesticks / orderbook reference: https://docs.kalshi.com/api-reference/market/get-market-orderbook, https://docs.kalshi.com/api-reference/market/get-market-candlesticks, https://docs.kalshi.com/api-reference/market/batch-get-market-candlesticks
- Historical candlesticks: https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks
- Kalshi API changelog: https://docs.kalshi.com/changelog
- "Trading the weather" (Kalshi News): https://news.kalshi.com/p/trading-the-weather
- Kalshi monthly rainfall contract terms (CFTC): https://kalshi-public-docs.s3.amazonaws.com/contract_terms/RAINM.pdf
- LST vs DST settlement window write-up: https://wethr.net/edu/trading-guide

**Polymarket**
- Polymarket Weather: https://polymarket.com/weather
- Polymarket Low Temp: https://polymarket.com/weather/low-temperature
- Polymarket Climate & Science (weather): https://polymarket.com/climate-science/weather
- Polymarket developer docs overview: https://docs.polymarket.com/developers/gamma-markets-api/overview
- Polymarket CLOB intro: https://docs.polymarket.com/developers/CLOB/introduction
- Gamma API guide (third party): https://agentbets.ai/guides/polymarket-gamma-api-guide/
- Polymarket API architecture summary: https://medium.com/@gwrx2005/the-polymarket-api-architecture-endpoints-and-use-cases-f1d88fa6c1bf
- prices-history 12h granularity issue: https://github.com/Polymarket/py-clob-client/issues/216

**Diurnal cycle physics**
- Diurnal temperature variation (Wikipedia): https://en.wikipedia.org/wiki/Diurnal_temperature_variation
- Daily Temperature Variations & Solar Radiation (Vermont State Univ.): https://apollo.nvu.vsc.edu/classes/met130/notes/chapter3/daily_trend3.html
- "Diurnal Cycle" (ResearchGate excerpt): https://www.researchgate.net/publication/266393516_Diurnal_Cycle

---

## Appendix B — Open Questions / Things Not Resolved in This Memo

1. **Exact station list per current Kalshi market**, including station changes over time. The contract docs say "as specified by the Exchange" — pull the live list from `/series` and `/markets` and snapshot it.
2. **Polymarket settlement source per international market**. London settles via what? Tokyo settles via what? Confirm individually from each market's rules text before any city is promoted to audit.
3. **Whether OpenMeteo's `hrrr` channel is HRRR raw or HRRR via NBM intermediate**. The current internal artifact suggests it's correlated to GFS — needs upstream confirmation from Open-Meteo's source documentation.
4. **Whether the `gfs_ens` MAE win replicates on a 2026 season**. The 1.015 vs. 1.222 result is on the current artifact; lock down a 2026-only walk-forward as a confirmation set.
5. **Whether monthly precipitation markets are calibratable at all**. The settlement is on the *first* monthly CLI containing all daily precip data for the month — a 15-day expiration window means liquidity dries up close to settlement. Market efficiency may be high on these.
6. **Hurricanes**. Not modeled here. Hurricane markets are a *very* different beast — they involve track + intensity + landfall geography uncertainty that small-station models can't capture. Treat as a separate Phase 7 problem. Aurora and GenCast both have demonstrated tropical-cyclone skill; that's the right starting point if you take it on.

---

*End of memo. No part of this is trade execution code or instruction. All findings are research-stage and require forward live-data validation before being treated as money-useful.*


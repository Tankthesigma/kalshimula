# Kalshimula Weather Market Edge System

## Executive summary

The strongest evidence supports a hybrid system in which **structured weather models do the numeric prediction** and a **local LLM does explanation, triage, and report writing**. MOS is explicitly a site-specific statistical post-processing framework that converts raw model output into more useful station guidance, LAMP explicitly updates MOS using recent observations and short-range model information, and NBM is explicitly a calibrated blend designed to give forecasters a consistent starting point rather than a final answer. That is the right template for Kalshimula: raw models and AI forecasts at the bottom, station-level post-processing in the middle, live observation nowcasting on top, and market-data validation in a separate private audit lane. citeturn38view0turn35view0turn36view1turn37view1turn37view0turn32view1

The **best near-term edge path** is not “train an LLM meteorologist.” It is to close the exact gap you identified: same-day live nowcasting after the market has already seen intraday temperatures. For U.S. work, that means ingesting ASOS and METAR observations in real time, using one-minute and special observations where available, then combining them with LAMP, NBM, HRRR, RAP, and your existing calibrated source stack to model the **remaining heating or cooling to settlement**. NOAA and NCEI document that ASOS runs continuously, produces hourly and special observations, and also has finer 1-minute and 5-minute archives; LAMP is specifically built to bridge the gap between observations and MOS; NBM is updated hourly and is already a calibrated blend. citeturn32view3turn17search9turn17search15turn35view0turn37view1turn37view2

The **most important missing data** are not glamorous. They are: a point-in-time station observation store; station metadata and quirks; one-minute U.S. ASOS history; a point-in-time archive of short-range guidance used as of each decision timestamp; a rules-normalized settlement map for every market; and, in the private audit lane, full time-stamped market snapshots with price, spread, depth, and rule text. Kalshi resolves daily weather markets from the final NWS Daily Climate Report and explicitly notes DST and rare METAR-versus-report consistency delays. Polymarket’s weather markets can use different station mappings and resolution sources, including airport-specific Wunderground pages and whole-degree Celsius rounding in current international examples. That rules layer is a first-order modeling input, not back-office detail. citeturn7view0turn24view0turn23view7

The **highest-risk assumptions** are: first, assuming “best MAE source” equals “best market edge source”; second, assuming settlement temperature equals the most obvious public weather-app number; third, assuming hourly markets share the same source logic as daily high and low markets; and fourth, treating two nominally different sources as independent before provenance is verified. Kalshi’s own rules show settlement-specific nuances, and Polymarket’s international weather rules already show source and rounding heterogeneity. Your own prompt-level evidence about GFS-ENS being useful partly because it differs from consensus fits that reality. citeturn7view0turn24view0

My bottom-line recommendation is: **build a structured model now, not an LLM predictor**. Use the local LLM later as a weather-desk analyst and reranker that writes morning, midday, and evening low reports from structured outputs. If you ever train a local 7B-class model, train it against report-quality and ranking labels, not raw temperature prediction. The expected professionalization path is: official observations and settlement rules, MOS/LAMP-style station post-processing, NBM-style calibrated blending, timestamp-clean live nowcasting, then private market validation. citeturn32view1turn35view0turn36view1turn37view1turn37view0turn40view0

## What the evidence says

### Professional meteorology already gives you the right template

The AMS describes modern forecasting as a synthesis of observations, NWP output, scientific theory, and experience, updated collaboratively as new information arrives. That is exactly the opposite of a one-shot text model guessing a number. The same AMS statement also notes that short-term forecasting increasingly uses rapidly updating numerical models plus statistical and AI tools that blend observations with NWP outputs, while beyond a few hours NWP remains the dominant base and is improved by statistical bias correction, model blending, ensembles, and machine learning. citeturn32view1

MOS matters because it is the canonical example of **site-specific post-processing**. NOAA’s MOS explainer describes MOS as a statistical post-processing scheme applied to NWP output that uses the historical record at forecast points, corrects systematic model biases, quantifies uncertainty, and transforms raw model output into sensible local forecast elements such as temperature, cloud, visibility, and thunderstorm probability. It is explicitly station-oriented and observation-aware. citeturn38view0turn38view1

LAMP matters because it is the canonical example of **same-day observation updating**. NOAA’s LAMP material describes it as a statistical system using observations, MOS output, and model output to provide guidance, explicitly bridging the gap between observations and MOS forecasts. The current NWS presentation states that LAMP runs every hour, covers roughly 1 to 38 hours for most elements, runs every 15 minutes out to 3 hours for ceiling and visibility, supports NBM, and in its “meld” form combines observations, Base LAMP, and HRRR MOS. That is a direct blueprint for Kalshimula’s nowcast layer. citeturn35view0turn36view0turn36view1

NBM matters because it is the operational answer to “how do I blend many imperfect models without handwaving?” NOAA and NCEP describe NBM as a nationally consistent, calibrated deterministic and probabilistic blend of NWS and non-NWS models plus post-processed guidance. NCEP’s current text description says it uses decaying-average and quantile-mapping bias corrections together with dynamic MAEs and expert weights; NOAA’s NWS article explains that it ingests many model systems, bias-corrects them, chooses weights, and provides a highly accurate, calibrated, and consistent starting point for forecasts. citeturn37view1turn37view0

### How forecasters actually think about daily highs and lows

The strongest practical material I found for station max and min forecasting is the Air Force meteorological techniques handbook. It recommends starting from climatology and persistence, then using upstream stations in the expected air mass, elevation corrections, cloud-cover effects on insolation, and diurnal temperature curves to make the final maximum-temperature forecast. It also describes Skew-T-based max-temperature methods and explicitly notes that if a morning radiation inversion is present, the warmest point in the inversion can become the starting point for the day’s max calculation. citeturn34view0turn34view1turn33view0

For minimum temperatures, the same handbook recommends using the moist adiabat through the 850 mb dew point to estimate the expected minimum when the air mass is stable, warns that the *second* morning after a cold frontal passage is often colder than the first, and presents the McKenzie method, which uses the day’s maximum temperature and dew point at time of max with corrections for overnight conditions such as calm and clear versus cloudy and windy regimes. It also recommends plotting hourly temperature and dew point histories to establish station-specific diurnal trend curves and using average diurnal variation as a practical rule of thumb. citeturn34view2turn33view1

NOAA materials on nighttime cooling and frost align with that operational logic. NWS educational pages explain that the ground cools after sunset because it loses more radiation than it gains, and frost-risk guidance emphasizes that clear skies and light winds are the classic setup for strong radiational cooling. NOAA research on the diurnal cycle states that surface temperature typically peaks a few hours after local noon. Put together, those sources imply the two settlement dangers you care about most: **late-day upside extension for highs under strong mixing and sunshine**, and **overnight undercut risk for lows under clear, calm, dry setups that keep cooling toward daybreak**. citeturn4search0turn4search11turn5search0

### What modern AI weather models prove and what they do not prove

GraphCast, GenCast, Pangu-Weather, AIFS, and Aurora all matter, but not for the reason most market builders first assume. GraphCast shows that global 0.25° medium-range deterministic forecasting can be very strong and very fast, with DeepMind reporting better performance than leading operational deterministic systems on most verification targets and forecasts in under a minute. GenCast extends that result to probabilistic 15-day global forecasting and, in the Nature paper, beats ECMWF ENS on most benchmarked targets. Pangu-Weather shows that AI can beat operational IFS on reanalysis-based deterministic tests and run far faster, but the paper also says real-world forecast-system performance still needs further investigation and notes limits around omitted variables and smoothness. Aurora shows that a large foundation model can be fine-tuned across several Earth-system tasks and beat strong operational baselines in those domains, while still requiring substantial training resources. citeturn41view0turn40view2turn41view1turn40view3

What these systems **do not** prove is that raw global AI output is ready to win station-level temperature markets. They mostly operate on grid scales, benchmark against reanalysis or global forecast targets, and do not natively solve station instrumentation issues, urban heat island quirks, airport siting, local sea-breeze timing, station rounding, or market-specific settlement windows. The most important recent paper for your use case is the 2025 post-processing study showing that statistical post-processing can improve AIFS just as it improves traditional NWP, and that blending AIFS with NWP models can improve skill even when AIFS alone is not best. That is a direct argument for **using AI weather models as blend components inside a post-processing stack**, not as stand-alone market predictors. citeturn40view0turn41view2turn41view1turn41view0

### Market structure matters at least as much as forecast accuracy

Kalshi’s weather help center states that its weather markets settle from the **final NWS Daily Climate Report**, usually the next morning, and it explicitly warns that market determination may be delayed if the final climate-report high is inconsistent with 6-hour or 24-hour METAR highs or if the final value is lower than a previous preliminary report. It also explicitly states that during Daylight Saving Time the NWS climate reports use **local standard time**, so the daily window is effectively 1:00 AM through 12:59 AM local clock time the next day, not a midnight-to-midnight DST window. That alone is enough to make careless backtests wrong. citeturn7view0

Kalshi’s public market-data surface is good enough for a private audit lane. Its docs expose markets, order books, trades, live candlesticks at 1-minute, 1-hour, and 1-day intervals, and historical candlesticks for archived markets. Kalshi also documents real-time WebSocket channels, though the order book update channel is documented as authentication-required, while the public REST market-data quickstart documents unauthenticated access to core market data. citeturn7view1turn7view2turn7view3turn7view4turn8search0turn8search1turn8search8

Polymarket has a broader international weather footprint right now. Its current weather pages show active markets in cities such as London, Shanghai, Beijing, Madrid, Singapore, Tokyo, Seoul, Hong Kong, and others, and its docs make clear that Gamma and Data APIs are public, while CLOB public endpoints expose order books, prices, and price history. The public market WebSocket exposes order book, price, and trade updates. Current international weather-market rule pages also show that resolution can be **airport-station specific**, use **Wunderground history pages** as the resolution source, and round to **whole degrees Celsius**. That is not a small implementation detail. It means global expansion requires a serious rule-normalization and station-mapping layer before any claim of cross-city generalization is trustworthy. citeturn23view0turn23view1turn23view4turn23view5turn23view7turn24view0turn42search0turn42search4turn42search8

### Primary source URLs

```text
https://www.weather.gov/documentation/services-web-api
https://aviationweather.gov/data/api/
https://www.ncei.noaa.gov/products/land-based-station/automated-surface-weather-observing-systems
https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
https://mesonet.agron.iastate.edu/api/
https://mesonet.agron.iastate.edu/request/asos/1min.phtml
https://madis.ncep.noaa.gov/
https://open-meteo.com/en/docs
https://open-meteo.com/en/docs/historical-forecast-api
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
https://www.ecmwf.int/en/forecasts/datasets/open-data
https://www.ecmwf.int/en/forecasts/dataset/aifs-machine-learning-data
https://arxiv.org/abs/2212.12794
https://www.nature.com/articles/s41586-024-08252-9
https://www.nature.com/articles/s41586-023-06185-3
https://www.nature.com/articles/s41586-025-09005-y
https://docs.kalshi.com/
https://help.kalshi.com/en/articles/13823837-weather-markets
https://docs.polymarket.com/
https://polymarket.com/weather
```

## Data stack priorities

The right ranking is driven by one rule: **buy or build whatever gets you closest to the actual settlement variable at the actual decision timestamp**.

1. **Official settlement-adjacent observations first.** For U.S. daily high and low work, ASOS and METAR observations are the core feed. ASOS runs continuously, produces hourly plus special observations, and NCEI archives 1-minute and 5-minute versions; AviationWeather’s METAR API and cache update as frequently as once a minute for current use. Usefulness: extremely high. Cost: free. Latency: very low live, moderate for historical 1-minute backfill. Reliability: highest for same-day nowcasts. Difficulty: moderate because of point-in-time storage and station quirks. citeturn32view3turn32view4turn17search9turn17search15

2. **Station metadata and station-history data immediately after that.** You need HOMR and ISD station metadata to normalize station identity, relocations, elevation, and network history. ISD provides global hourly and synoptic observations from more than 20,000 stations, and NCEI exposes station-history APIs. Usefulness: extremely high. Cost: free. Latency: low for metadata, moderate for historical data pulls. Reliability: high. Difficulty: low to moderate. citeturn17search3turn18search3turn18search0

3. **LAMP should be a first-class model input, not a nice-to-have.** LAMP exists specifically to update station guidance using observations and short-range model information, runs hourly, and supports NBM. For highs, lows, and especially hourly temperature markets, it is the closest public U.S. analogue to a professional same-day desk. Usefulness: extremely high for U.S. short horizons. Cost: free. Latency: very low. Reliability: high in its intended horizon. Difficulty: moderate mostly because of retrieval, parsing, and time alignment. citeturn35view0turn36view0turn36view1

4. **NBM should be the blend anchor.** NBM is already a calibrated deterministic and probabilistic blend, updated hourly, built to be a consistent starting point. Use it as an input feature family, not as the sole forecast. Usefulness: extremely high. Cost: free. Latency: low. Reliability: high. Difficulty: moderate. citeturn37view1turn37view2turn37view0

5. **HRRR and RAP are the highest-value raw short-range models.** HRRR is NOAA’s hourly updated 3 km, convection-allowing model with radar assimilation, and RAP is the broader hourly short-range analysis/forecast system. For 0 to 12 hour station nowcasts, these should outrank slower global raw models. Usefulness: very high. Cost: free. Latency: low. Reliability: high for short-range mesoscale structure, but still needs post-processing. Difficulty: moderate to high because of data volume and variable extraction. citeturn9search5turn9search8turn10search2turn10search13

6. **GFS, NAM, and ECMWF/IFS/AIFS are the next model layer.** GFS and NAM remain important background sources, while ECMWF open data now makes a subset of IFS and AIFS publicly available under CC BY 4.0 plus ECMWF terms. AIFS machine-learning data are available on a 0.25° grid, four runs per day, out to 15 days. Usefulness: high. Cost: mostly free at the open-data layer. Latency: low to moderate. Reliability: high as blend components, lower as direct station-settlement predictors. Difficulty: moderate. citeturn10search11turn10search1turn41view3turn41view2

7. **Open-Meteo stays valuable as a convenience and historical-forecast archive layer.** Open-Meteo exposes multiple models, including NOAA, ECMWF, and others, gives a best-match option, and archives historical forecasts from 2021 or 2022 onward depending on source availability. It is excellent for source-breakout experiments, rapid prototyping, and cross-model retrieval, but it is not the settlement authority and should not be treated as such. Usefulness: high for research velocity. Cost: low. Latency: low. Reliability: good as a convenience layer. Difficulty: low. citeturn16search0turn16search1turn16search4

8. **ERA5 and ERA5-Land are training infrastructure, not settlement truth.** ERA5 offers hourly global atmospheric reanalysis from 1940 onward, while ERA5-Land gives finer land-surface resolution at about 9 km. They are ideal for training regime classifiers, climatology baselines, and deep models, but they should not replace station observations in settlement-linked modeling. Usefulness: high for training and benchmarking. Cost: free. Latency: not live. Reliability: high for large-scale complete fields. Difficulty: moderate. citeturn21search1turn21search4turn21search12turn21search7

9. **Radar, MRMS, and GOES should be added when you start losing edge to clouds and convective timing.** MRMS updates every two minutes and explicitly fuses radar, satellite, surface obs, lightning, and models; GOES provides continuous imagery; NEXRAD Level II is available in near real time. These are most useful once the observation-plus-tabular nowcast layer is working and you want explicit cloud and precip-state features. Usefulness: medium now, very high later for convection-prone cities. Cost: free. Latency: very low. Reliability: high. Difficulty: high. citeturn20search5turn20search14turn20search7turn20search0

10. **Kalshi market metadata and candlesticks are mandatory for the private audit lane.** Kalshi’s public docs expose markets, order books, trades, and candlesticks, plus series metadata with settlement sources. Usefulness: extremely high for audit, zero for mainline forecasting. Cost: low. Latency: low. Reliability: high. Difficulty: low to moderate. citeturn7view1turn7view2turn7view3turn7view4turn26search4

11. **Polymarket Gamma plus CLOB plus market WebSocket are mandatory for global audit.** Gamma and Data APIs are public, CLOB public endpoints expose order books and price history, and the market WebSocket provides real-time order book and trade data. The documented history endpoint is for prices, not full depth, so if you care about historical book shape you should archive WebSocket snapshots and periodic /book pulls yourself. That is an inference from the current docs, which document current books and price history but not an official order book history endpoint. Usefulness: extremely high for audit. Cost: low. Latency: low. Reliability: good. Difficulty: moderate. citeturn23view0turn23view1turn42search0turn42search4turn42search8

12. **International station data sources come last only because the U.S. edge path is cleaner.** Globally, METAR coverage and ISD give you broad station reach, and Polymarket’s current markets are heavily airport-station-centric, which is good. But international rule heterogeneity, unit rounding, source differences, and local conventions mean global expansion is a rules-engine problem as much as a forecast problem. Usefulness: high for expansion. Cost: mostly free. Latency: low to moderate. Reliability: mixed by country and station. Difficulty: high. citeturn17search3turn32view4turn24view0turn23view7

## Recommended architecture

### Mainline weather core

The recommended mainline architecture is a **three-layer structured system**.

The first layer is a **source ingestion and normalization layer**. It pulls station observations, model guidance, and forecast archives into a point-in-time feature store with strict as-of semantics. Every feature must know its issue time, availability time, valid time, station, model, lead, run age, and provenance. Every source family that might be duplicated, such as your current HRRR-versus-GFS concern, should get a provenance-confidence flag so that independence assumptions are not silently baked into ensembles. This layer remains weather-only and can safely stay out of market execution scope. The scientific rationale is straightforward: both professional forecasting and operational post-processing systems start from careful observation and model synthesis, not from a flattened one-row CSV with hidden time leakage. citeturn32view1turn35view0turn37view1

The second layer is a **station-level post-processing and probabilistic distribution layer**. This is where you should turn raw sources into calibrated station outcomes. The model family I would prioritize is not a giant transformer at first. It is a hierarchy of **tabular models plus calibration**: city-specific and timestamp-specific gradient-boosted models or closely related tabular learners for the conditional mean, conditional quantiles, and market-bin probabilities, followed by isotonic or beta calibration and conformal interval checks. That recommendation follows from the structure of the problem: the key inputs are heterogeneous, sparse in places, timestamped, station-specific, and heavily driven by known physical covariates such as current temperature, dew point, cloud, wind, solar geometry, model spread, run age, and local climatology. MOS, LAMP, and NBM all point toward richer post-processing, not raw end-to-end text or image models. citeturn38view0turn35view0turn37view1turn40view0

The third layer is a **reporting and analyst layer**. This is where a local LLM belongs. Its inputs should be structured probabilities, calibrated intervals, source disagreements, live observations, station metadata, and prior diagnostics. Its outputs should be a morning call sheet, midday nowcast note, evening low-temperature watch, skip reasons, strongest pro factors, strongest con factors, and explicit confidence labels. It should never be the source of the actual numeric probability used for audit. That separation is critical because the evidence base favors structured post-processing for weather numbers and human-readable explanation for communication. citeturn32view1turn35view0turn37view0

### Live nowcast architecture

The nowcast layer should be modeled as **remaining-path estimation**, not just “rerun the morning model with current temp added.” Build separate targets for:

- remaining increase to daily high,
- remaining decrease to daily low,
- hourly temperature at 1, 2, 3, and 6 hours,
- probability that the final settlement bin changes from the current best guess.

The core features should include current station temperature, high-so-far, low-so-far, recent slope and curvature of temperature, dew point and dew-point depression, wind and gust history, cloud cover history, pressure trend, precipitation state, solar angle, time since sunrise or to sunset, model remaining heating or cooling from HRRR, RAP, LAMP, and NBM, plus station climatology and regime labels. This is exactly where same-day edge will come from in cities where late-morning and afternoon prices already reflect observed temperature trajectories. The meteorological support for these features is strong: LAMP exists to blend observations with MOS and short-range guidance; ASOS provides the relevant observation elements; nighttime cooling guidance emphasizes clouds and wind; and NOAA materials on apparent temperature variables confirm the importance of sun angle, cloud cover, humidity, and wind in surface thermal outcomes. citeturn35view0turn32view3turn4search0turn4search11turn5search16

For **high-temperature markets**, the mainline evaluation times should be exactly the ones you listed: 4 AM, 7 AM, 10 AM, noon, and 2 PM local. My expectation is that 4 AM and 7 AM will reward better post-processed model guidance, while 10 AM through 2 PM will only remain attractive if the nowcast layer materially improves on what the market already infers from observations. Because surface temperature typically peaks a few hours after local noon, the 10 AM to 2 PM window is where “remaining daytime heating” should become the central modeling target. citeturn5search0turn35view0

For **low-temperature markets**, split the day into two physically different problems. The first is prior-evening and overnight cooling. The second is dawn-risk management and post-sunrise protection against a hidden undercut before the official low is locked in. Under clear, calm conditions, temperatures can keep decaying through the night. The Air Force handbook’s minimum-temperature methods, especially its use of Tmax, dew point at Tmax, and overnight-condition corrections, are highly relevant here. This is why I would run low models at previous evening, midnight, 4 AM, sunrise, and 6 PM, but with separate model families for 6 PM versus pre-dawn inference. citeturn34view2turn4search0turn4search11

For **hourly temperature markets**, the right base is observations plus LAMP plus HRRR or RAP. At 1 to 3 hours, short-range station-updated guidance should dominate. At 6 hours, the plot becomes more market-dependent: if liquidity is low or the rule source is idiosyncratic, there may still be edge; if the market has deep liquidity and many participants see the same METAR and short-range model updates, the edge may collapse quickly. The right answer is empirical, but hourly should be treated as a distinct product, not as a side effect of the daily model. citeturn35view0turn32view4turn9search5turn9search8

### Global and Polymarket expansion

The global architecture should **not** be “clone Houston, swap in London.” Start with a separate global normalization stack. Polymarket’s current weather footprint is already international, and the current rule examples show airport-specific settlement sources and whole-degree Celsius resolution. That means the model target is not just temperature. It is **temperature after market-specific source choice, metric/imperial convention, station selection, and rounding rule**. citeturn23view7turn24view0

The easiest international cities are likely those with large, stable airport stations, rich METAR history, and relatively low convective randomness. Examples include London, Madrid, Paris, Tokyo, Singapore, and Dubai. The hardest are likely mountain or basin cities, strong sea-breeze cities, and tropical-convective cities where a one-hour cloudburst can cap the day’s maximum. Mexico City, Hong Kong, and many thunderstorm-prone Chinese or Southeast Asian cities will require stronger cloud and storm timing features than Los Angeles or Phoenix-style regimes. That ranking is an engineering inference, but it follows directly from the station-centric rule structure and the known importance of cloud, wind, and radiative effects in surface temperatures. citeturn24view0turn5search0turn4search0

### Local LLM weather-desk analyst

Do **not** train a local 2B, 3B, or 7B model to forecast temperature directly as the mainline numeric engine. The numeric part of this problem is a calibrated, station-level, timestamped probabilistic regression and classification problem. The evidence base for modern weather forecasting still points to structured models, post-processing, and ensemble blending. The best use of a small local LLM is as an **analyst and reranker** that consumes structured outputs, writes notes, flags regime risk, and suggests when to skip a market because rule ambiguity or market efficiency is too high. citeturn32view1turn35view0turn40view0

If you fine-tune later, the training set should be built from your own packets and audit labels: structured features, model distributions, final outcomes, whether a market would have cleared your decision thresholds, whether it would have failed due to liquidity or settlement ambiguity, and a human-written or templated best-practice note. The benchmark should not be “did the LLM guess the temperature closer than LightGBM?” The benchmark should be “does LLM-based ranking or skip-triage improve out-of-sample decision quality over deterministic rules, without hallucinating facts or changing calibrated probabilities?” That is the only benchmark that respects where LLMs are likely to help.

## Experiments and private audit

### Timestamp-clean forecast experiments

Your benchmark suite should be **as-of-timestamp or it does not count**. For every forecast row, store issue timestamp, feature availability timestamp, target settlement definition, and the exact market rule version applicable that day. The weather-side experiment matrix should include:

- highs at 4 AM, 7 AM, 10 AM, noon, 2 PM local;
- lows at previous evening, midnight, 4 AM, sunrise, 6 PM;
- hourly temperature at 1, 2, 3, and 6 hour horizons;
- city, station, season, market type, regime, and lead-time slices.

The primary scores should be Brier score, log loss, calibration or ECE, MAE for continuous intermediates, quantile coverage, and decision-support lift over baselines such as climatology, persistence, best single source, openmeteo_naive, raw NBM, and raw LAMP where applicable. LAMP and NBM are especially important baselines because they already encode the operational logic you are trying to replicate. citeturn35view0turn37view1turn37view0

The kill criteria should be strict. Kill a model or city if the apparent edge disappears after fees and slippage, if the gain vanishes in forward live paper data, if it is driven by one season or one city, if it requires corrected or unavailable data, or if calibration deteriorates materially once you move from retrospective archive runs to true point-in-time workflows. Also kill any experiment where a settlement-rule mismatch explains more of the apparent edge than the weather model does.

### Private market-data audit design

Keep market-data work in a **private read-only audit lane** and keep mainline weather-only. In that private lane, evaluate model probability against market probability on both YES and NO sides, across exactly the bucket and edge grids you listed: price buckets such as 0.05-0.95 through 0.30-0.70 and edge cutoffs such as 0.02, 0.03, 0.05, 0.08, 0.10, and 0.15. Use conservative fill assumptions based on documented order book access: for Kalshi, use current order books, trades, and candlesticks; for Polymarket, use current order books, price history, and market WebSocket feeds. citeturn7view2turn7view4turn8search13turn42search4turn42search0turn23view1

The private audit should explicitly separate **forecast edge** from **tradable edge**. A market can be miscalibrated in narrative terms but still not be worth touching after spread, fees, stale-book risk, and last-minute information absorption. For Kalshi, also respect the settlement nuances around final climate reports, DST windows, and rare METAR consistency checks. For Polymarket, archive rule text with every snapshot because rule and source heterogeneity is larger and can differ by city and contract family. citeturn7view0turn24view0

Because current Polymarket docs expose historical **price** data but not a documented full historical order book endpoint, you should archive your own event stream going forward if microstructure is part of the research question. The minimal archive is: event slug, market IDs, token IDs, rule text, settlement source URL, every price snapshot, every periodic /book snapshot, and every WebSocket delta with receive timestamps. That is the only way to answer later whether “stale price risk” or “book imbalance” contributed real edge. This is an inference from the currently documented public interfaces. citeturn42search0turn42search4turn23view1turn42search8

## City expansion and implementation roadmap

### City tiers

**Tier one should be current U.S. cities plus a few station-clean additions.** Start with the cities already in your artifact, then add only where settlement mapping is unambiguous and same-day observation feeds are clean. High-temperature-friendly cities are typically those where the local thermal regime is easier to model with sunshine, dry air, and less convective surprise. Desert and stable-coastal regimes often belong here. Thunderstorm-prone and lake-breeze-prone cities should be added more cautiously because a better weather model does not automatically convert to better market PnL if price discovery is already incorporating observed instability and mesoscale drift. This is consistent with your own finding that raw forecast accuracy and market mispricing are different objects.

**Tier two should be U.S. lows and hourly markets only after the nowcast layer works.** Lows are physically tractable but operationally fragile because undercut risk, cloud breaks, and wind decoupling can matter late. Hourly markets should only be promoted once you have a robust as-of station observation store and strict rule parser, especially if some hourly products use different source logic than daily high and low products.

**Tier three should be international airport-station cities on Polymarket.** Prioritize cities with strong airport metadata, rich METAR or ISD history, high liquidity, and clear rules. London, Madrid, Tokyo, Singapore, Paris, and Dubai are better first candidates than more topographically or convectively complex cities. Defer elevation-sensitive, basin, and monsoon-convective cities until you have cloud and precipitation nowcast features working well.

Before each new city tier is added, require four things: a validated settlement map, at least one year of usable historical observations, proof that nearest-grid or station interpolation is acceptable, and forward live paper evidence that the city’s edge survives market-data reality.

### Repository blueprint

A clean repository plan for Kalshimula should split into **mainline-safe weather modules** and **private audit modules**.

Mainline-safe modules should include: `obs_ingest`, `station_metadata`, `model_adapters`, `historical_forecast_archive`, `feature_store_asof`, `postprocess_high`, `postprocess_low`, `postprocess_hourly`, `calibration`, `regime_labels`, `report_packets`, and `desk_reports`. These modules should contain no market credentials, no order logic, and no execution capabilities.

Private audit modules should include: `kalshi_marketdata_readonly`, `polymarket_marketdata_readonly`, `rules_parser`, `market_snapshot_archive`, `paper_fill_simulator`, `fee_slippage_models`, and `audit_dashboards`. These should live in a separate private lane so the mainline remains weather-only and commit-safe.

The core schemas should include: observation facts, station metadata, model forecast members, model summary stats, feature rows with availability timestamps, calibrated distribution outputs, settlement-definition records, and in the private lane market snapshots and rule versions. The most important tests are not generic unit tests. They are **leakage tests**, **DST-window tests**, **station-mapping tests**, **duplicate-source provenance tests**, and **report-reproducibility tests**.

### What to build this week and next month

**This week**, build five things in order. First, a point-in-time station observation store for the current U.S. city set with one-minute backfill where available. Second, a rule-normalized settlement definition table for every active daily high and low market you care about. Third, a live nowcast feature builder that can regenerate features at 4 AM, 7 AM, 10 AM, noon, and 2 PM without leakage. Fourth, a first-pass high and low post-processing model using your current best sources plus LAMP and NBM. Fifth, a private audit recorder for price, spread, and depth snapshots. citeturn17search15turn7view0turn35view0turn37view1turn7view2turn42search4

**Over the next month**, add the remaining heating and cooling nowcast models, launch the hourly horizon benchmark, verify source provenance for every model family, stand up the private paper-audit dashboard, and start forward live paper tracking. Also add automated “skip reasons,” such as high settlement ambiguity, low liquidity, duplicate source provenance, or market already moved after decisive intraday observations.

**What requires paid data or major compute** should come later. Open NOAA, ECMWF, and Open-Meteo sources already cover most of the necessary raw inputs, and ECMWF open data is commercially usable with attribution. The first paid spend should be justified only after uplift is proven and should focus on cleaned high-frequency observations, resilient low-latency data delivery, or premium derived cloud and radar features, not on buying another global model feed just because it feels more institutional. Full custom foundation-model work is a major-compute project by any serious standard; Aurora’s published training scale alone shows how far that road goes. citeturn41view3turn16search0turn40view3

**What to avoid** is equally clear. Avoid direct LLM temperature prediction. Avoid evaluating on corrected data that was unavailable at decision time. Avoid treating consensus disagreement as a hard trade gate rather than a feature. Avoid city expansion before settlement normalization. Avoid assuming Polymarket and Kalshi represent the same target variable even when the market title sounds similar.

## Open questions and limitations

The most important unresolved item is **contract-family heterogeneity**, especially for hourly temperature products and some precip markets. Kalshi’s daily weather help clearly describes final NWS Daily Climate Report settlement for daily weather markets, but individual hourly and special-contract families need their own rule extraction and normalization before any broad generalization is safe. citeturn7view0turn27search7

The second unresolved item is **historical depth reconstruction** for Polymarket. The current official docs clearly provide public current order books, price history, and live market WebSocket data, but I do not see a documented public full order book history endpoint. That means any serious historical microstructure work should start archiving now rather than waiting for a future backfill that may not exist. This is an inference from the current public documentation, not a statement that no third-party archive exists anywhere. citeturn42search0turn42search4turn23view1turn42search8

The third unresolved item is **source provenance** inside your existing weather artifact. Because your prompt already flags possible HRRR versus GFS duplication, I would treat source-independence assumptions as untrusted until the raw provenance, run timing, and vendor transformation path are audited.

The practical final recommendation is therefore straightforward: **build the live observation nowcast layer first, wire it into station-level post-processing second, validate it against real market data in a private read-only audit lane third, and only then add an LLM analyst layer for ranking and communication**. That is the path most likely to make Kalshimula meaningfully closer to a professional weather desk while staying inside your safety and scope constraints.
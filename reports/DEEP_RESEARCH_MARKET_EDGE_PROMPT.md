# Deep Research Prompt: Weather Prediction Market Edge

You are an adversarial prediction-market microstructure researcher. Your job is to determine where, if anywhere, a small directional paper trader can find durable after-cost edge in daily weather prediction markets, primarily Kalshi temperature markets and secondarily comparable Polymarket markets.

The goal is not to produce a generic weather forecasting memo. The goal is to answer the money question: when does forecast skill turn into executable expected value after fees, spread, slippage, liquidity limits, capacity, settlement risk, and multiple-testing risk?

If the evidence says these markets are efficient and there is no durable small-trader edge after costs, say that plainly. Do not manufacture edges. A negative result is acceptable.

## Current Context

We already have:

- Weather model probabilities for daily high temperature bins.
- Kalshi market snapshots at decision times such as 7am, 10am, noon, 1pm, and 3pm.
- Settlement or NCEI daily high data.
- Station and settlement-rule tables.
- Paper-audit scoring with fees, spread/slippage assumptions, and forward logs.
- Evidence that forecast accuracy does not automatically imply market edge.
- Evidence that liquidity and timing matter: early books can be thin, while later books may already price intraday observations.

Kalshi is the primary venue because we operate there and have archived data. Polymarket is secondary and comparative. Treat Polymarket as an expansion venue, not a coequal focus, unless evidence shows it has clearly better small-trader opportunity.

## Hard Requirements

### 1. Quantify Or Discard

For every claimed mispricing, quantify all of the following in cents:

- Typical gross edge.
- Fee cost.
- Spread cost.
- Slippage cost.
- Net expected value after all costs.
- Sensitivity at low, medium, and high trade size.

If you cannot estimate magnitude, label the idea as an unverified hypothesis or discard it. Do not rank qualitative edges.

For the single strongest claimed edge, provide a worked example:

- Real or realistic contract.
- Model probability.
- Market price.
- Bid/ask or executable price.
- Fee at that price.
- Spread/slippage assumption.
- Net EV per $1 notional.
- Capacity estimate.
- Why the example is not cherry-picked.

### 2. Capacity Is A Hard Gate

For each candidate edge, estimate realistic deployable capacity:

- Contracts fillable per day at the edge price.
- Dollars at risk per day.
- Expected profit dollars per day.
- Expected profit dollars per season.
- How quickly the edge disappears as size increases.

Flag any positive-EV edge as IMMATERIAL if capacity is too small to matter. A positive edge that only supports 1-2 contracts is real but not operationally important unless it scales across many independent markets.

Choose and justify materiality thresholds for a small directional trader, such as minimum contracts/day, minimum expected dollars/day, and minimum expected dollars/season after all costs.

### 3. Recent Evidence And Persistence

For every claimed edge, identify:

- The losing counterparty or structural source of mispricing.
- Why that counterparty keeps being wrong.
- Evidence that the edge still exists in recent 2024-2026 Kalshi or Polymarket data.
- Whether the evidence is verified in these venues, only historically documented, or only observed in other markets.

If an edge is documented only in older studies or other markets, label it UNVERIFIED-HERE and rank it below recently verified edges.

### 4. Temporal Repricing And Information Timing

Investigate when the model can know something before the market prices it.

Analyze repricing latency for:

- New ASOS/METAR observations.
- NWS, NBM, LAMP, HRRR, GFS, or other forecast guidance updates.
- Sharp intraday moves such as fronts, sea breeze, rain, clouds, or sudden heating.
- Settlement-probability changes near the daily high or low.

For each timing claim, estimate:

- Lag window in minutes or hours.
- Typical gross edge during the lag.
- Liquidity available before the edge decays.
- Whether the edge survives fees, spread, and slippage.

Compare decision windows:

- Previous evening.
- Overnight.
- 7am local.
- 10am local.
- Noon to 1pm local.
- 3pm local.
- Pre-sunrise lows.
- Late-day lows.

Depth over breadth: go deep on the 2-3 most plausible small-trader edges rather than listing 15 vague biases.

### 5. Edge Mechanism Taxonomy

Assign every candidate edge to exactly one primary mechanism:

- Forecast-skill edge.
- Intraday-latency edge.
- Liquidity or microstructure edge.
- Tail-bin or probability-shape edge.
- Settlement or provenance edge.
- Cross-venue settlement mismatch.
- Behavioral or attention edge.

For each mechanism, give:

- The causal story.
- What data would prove it.
- What data would kill it.
- The strongest competing explanation.

Better forecast accuracy alone is not evidence of money edge. A candidate only matters if model-market disagreement predicts settlement after fees at executable prices.

Separate signal-generation edges, timing/entry edges, and risk-filter/avoidance edges. Do not rank defensive filters as profit engines unless they improve net EV.

### 6. Settlement And Cross-Venue Provenance

For Kalshi and Polymarket comparisons, first prove the underlying is identical:

- Station.
- Weather source.
- Unit.
- Rounding rule.
- Settlement window.
- Timezone and DST handling.
- Close time.
- Market wording.

If the underlying differs, label the apparent edge as FALSE ARB or DIFFERENT RISK, not a cross-venue opportunity.

Pay special attention to daily high and low markets, because high/low settlement windows, station choices, and DST rules can differ.

### 7. Validation And Multiple Testing

Assume every in-sample edge is overfit until it survives out-of-sample validation.

Specify a forward-validation protocol that survives multiple testing across cities, bins, sides, hours, days, and venues:

- Minimum independent forward observations before believing an edge.
- How to define independence when bins from the same city/day are correlated.
- Walk-forward or blocked out-of-sample design.
- Multiple-comparison control across hundreds of tested cells.
- Confidence intervals or bootstrap uncertainty for net EV.
- Day and city concentration checks.
- Freeze filters before forward testing.
- Explicit kill criteria.
- How to distinguish a lucky streak from a real edge at our sample sizes.

In-sample performance is hypothesis generation only. It is not evidence of a tradable edge.

### 8. Existing-Data Experiments

End with the first 3-5 concrete experiments we should run on our existing Kalshi paper-audit data.

For each experiment, specify:

- Exact input files or data tables.
- Required columns.
- Join keys.
- Decision-time labels.
- As-of timestamp rules.
- Leakage traps.
- Metric.
- Pass/fail threshold.
- Minimum N.
- Kill criterion.
- Whether it belongs in the mainline weather lane or private market-audit lane.

Use our existing data where possible:

- Model probabilities.
- Archived prices at 7am, 10am, noon, 1pm, and 3pm.
- Settlement or NCEI highs.
- Station-rule table.
- Observation coverage flags.
- Fee/spread/slippage assumptions.

### 9. Source Quality

Use this evidence hierarchy:

1. Primary exchange docs and live/archive API data.
2. Our archived Kalshi paper-audit data.
3. Peer-reviewed or current market microstructure papers.
4. Official weather and forecast-model documentation.
5. Credible practitioner writeups.
6. Social posts or anecdotes, clearly labeled as weak evidence.

Include source dates. Anything not recent enough for a live market claim must be labeled historical or background.

For each recommended API or data source, state:

- Exact fields to archive.
- Collection frequency.
- Retention risk.
- Whether the data can be reproduced later.
- Whether it is read-only and safe for paper auditing.

Unsupported claims must go into a hypothesis section, not the findings.

### 10. Implementation Contract

For the top 3 experiments, include an implementation plan:

- Dataset.
- Join keys.
- Timestamp and as-of rules.
- No-leak requirements.
- Output table schema.
- Metric calculation.
- Owner lane: mainline weather model or private market audit.

No-leak rules:

- Market prices must not enter mainline model features.
- Settlement or actuals must not enter pre-settlement predictions.
- Forecast guidance must satisfy available_ts <= as_of_ts.
- Observations must satisfy available_ts <= as_of_ts with reporting lag.
- Market scoring must use the price snapshot available at the decision time, not a future or fixed candle.

If the evidence is weak, the correct recommendation is "do not build yet; collect these data first."

## Required Output

### Executive Verdict

One line:

REALISTIC SMALL-TRADER EDGE: YES / NO / CONDITIONAL

Then explain in 3-5 bullets why.

### Ranked Edge Table

Provide a ranked table with these columns:

- Edge.
- Primary mechanism.
- Venue.
- Market type.
- When / decision time.
- Typical gross edge in cents.
- Fee cost in cents.
- Spread and slippage cost in cents.
- Net EV in cents.
- Capacity contracts/day.
- Capacity dollars/day.
- Expected profit dollars/day.
- Expected profit dollars/season.
- Liquidity/capacity status.
- Why not arbed / persistence.
- Losing counterparty.
- Small-trader capturable: yes/no.
- Recent evidence: verified-here / unverified-here / historical-only.
- Validation status.
- Kill criteria.
- Final rank: actionable / conditional / immaterial / discard.

### Risk Register

Include a risk register covering:

- Leakage.
- Thin liquidity.
- Stale prices.
- Settlement mismatch.
- Multiple testing.
- Market already priced the information.
- Capacity too small.
- Fee/spread kill.
- Station or timezone error.
- Model target mismatch.

For each risk, give mitigation and current evidence status.

### Experiments To Run Next

List the first 3-5 experiments we should run now, using our existing Kalshi audit data. Each experiment must include:

- Why it matters.
- Required data.
- Exact metric.
- Pass threshold.
- Fail threshold.
- Minimum sample size.
- Owner lane.
- Expected runtime or implementation difficulty.

### Build / Do-Not-Build Decision

Close with:

- What to build immediately.
- What to collect before building.
- What to reject.
- What to revisit only after more forward data.

If there is no durable, material edge after costs and capacity limits, say so directly.

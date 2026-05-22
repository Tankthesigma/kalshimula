# Phase 11 — Time-of-day P&L study

Same trading rule (edge≥0.05, dollar_risk_1, Kalshi fees, recal model prob)
evaluated at three different snapshot times per day.

- 04 UTC = midnight EDT target day start. No daytime info yet.
- 12 UTC = 8am EDT. Morning of target day; well before high forms in most cities.
- 20 UTC = 4pm EDT. Afternoon; high is forming for east cities, ongoing for west.

Market probability for each snapshot uses bid/ask midpoint when available,
otherwise last_close. (Phase 5 used last_close only.)

| hour UTC | n | win% | gross | net |
|---|---|---|---|---|
| 04 | 507 | 72.2% | $190.11 | $176.29 |
| 12 | 452 | 68.6% | $203.78 | $189.86 |
| 20 | 296 | 38.5% | $28.15 | $14.70 |

## Verdict — operationally critical

**Place bets in the morning, not afternoon.**

- 04 UTC (midnight EDT, no info) → win 72.2%, net $176.29 (n=507)
- 12 UTC (8am EDT, normal morning bet) → win 68.6%, net $189.86 (n=452) ← best total
- **20 UTC (4pm EDT, high forming) → win 38.5%, net $14.70 (n=296)** — edge collapses

By 4pm EDT the market has converged with the actual day's high. Three takeaways:
1. n drops 507 → 452 → 296 as the day progresses (model and market converge → fewer trades clear the 0.05 edge filter)
2. win rate plummets at 20 UTC (38.5%) — adverse-selection signal that we're trading against an informed market by afternoon
3. Net per trade: $0.35 (04 UTC) → $0.42 (12 UTC) → $0.05 (20 UTC). Edge per bet drops 88% by afternoon.

**Recommended operating window: 04-12 UTC** (midnight-to-morning EDT) for bet placement.

Note: 12 UTC slightly beats 04 UTC on total net because the 04 UTC snapshot has a few more
contracts at extreme prices (≤5c) that the dollar_risk_1 sizing scales up aggressively
and that don't always pay off. 04 UTC has better win rate but 12 UTC has better dollar-net.
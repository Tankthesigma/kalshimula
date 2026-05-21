# Night-ops runbook

Short operator guide for keeping the project healthy while bobby + codex
work autonomously. Read top-to-bottom once, then jump to whichever
"something looks wrong" section matches the symptom.

## Daily health check (60 seconds)

1. Open the Actions tab. Confirm latest CI is green and `Source Quality (live)`
   ran on Monday with no obvious-red rows in the run summary.
2. Open the agent Discord channel. Scroll to the most recent `@bobby` /
   `@codex` exchange. If neither agent has posted in >30 min during a known
   active run, see [bridge health](#bridge-health).
3. If the active collection runner is in flight, glance at codex's most
   recent status post -- should mention row count + per-city progress + any
   errors.csv entries.

## Reading the source-quality summary

`Source Quality (live)` posts to `$GITHUB_STEP_SUMMARY` and uploads two CSVs.
The summary shows one row per source. Columns:

| Column | What it means |
|---|---|
| `n` | rows probed for this source (about number of configured cities) |
| `ok_count` | fetcher call returned without raising |
| `error_count` | fetcher raised -- see error column in `smoke_all.csv` |
| `missing_high_count` | call succeeded but the source had no `high_f` for that day |
| `ok_rate` | `ok_count / n` |
| `missing_high_rate` | `missing_high_count / ok_count` (well-defined as 0 when `ok_count=0`) |

A healthy weekly summary on the default target (7 days ago) looks like:

```
source,ok_rate,missing_high_rate
asos,1.0,~0.0       # IEM is reliable for past days
ncei,1.0,~0.0       # Access Data Service is reliable >3d back
nws,1.0,1.0         # forecast API never returns historical, expected
power,1.0,~0.0      # NASA POWER is reliable >5d back
```

**`error_count > 0` for any source is the actionable signal.** Drift
(`missing_high_rate` shifting up) is a softer signal -- codex's note: this
workflow is intentionally non-blocking, so missing-high drift requires a
human looking at the per-city `smoke_all.csv` to decide whether it's a
real regression or just a publication delay.

## Source-specific triage

### NCEI red or 100% missing-high

- **Most likely cause**: API response shape changed.
- **Check**: open the most recent `tests/fixtures/ncei_*.json` against
  a fresh live response (`curl 'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00094728&dataTypes=TMAX&startDate=2025-01-01&endDate=2025-01-01&format=json&units=metric'`).
- **If fields renamed** (for example, `TMAX` changed): update
  `src/fetchers/ncei.py::_row_high_f`, re-capture fixtures, bump the
  pinned-shape tests in `tests/test_ncei_live_fixtures.py`.
- **If just a slow day**: NCEI publishes daily summaries 1-3 days after
  the date. Re-run the workflow with `target_date` 5+ days back.

### ASOS red or 100% missing-high

- **Most likely cause**: IEM endpoint shape change or station code
  convention change (the K-prefix bug).
- **Check**: `python -m src.fetchers.asos` won't run standalone; instead
  curl `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=KORD&data=tmpf&year1=2025&month1=1&day1=1&year2=2025&month2=1&day2=1&format=onlycomma`
  and inspect the `station` column.
- **K-prefix mismatch** (rows say `ORD`, code expects `KORD` or vice versa):
  see `_asos_station_key` in `src/fetchers/asos.py` and the
  `test_iem_strips_leading_k_in_csv_rows` test for the pattern.

### NWS red

- **Most likely cause**: bad User-Agent header, gridpoint URL changed, or
  the NWS forecast endpoint is rate-limiting.
- **Check**: `curl -H "User-Agent: weather-predictor-debug/1.0" https://api.weather.gov/points/39.8328,-104.6575`
  -- should return JSON with `properties.forecast`.
- **100% missing-high on historical target is normal**: NWS forecast API
  does not serve forecasts for past dates.

### POWER red

- **Most likely cause**: NASA POWER service outage or shape change.
- **Check**: `curl 'https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M_MAX&community=RE&latitude=40&longitude=-75&start=20250101&end=20250101&format=JSON'`.
- **Expected fill value**: `-999` or `-9999` for missing data; the parser
  treats both as missing.
- **100% missing-high on a target within last ~5 days is normal**: POWER
  daily aggregates lag.

## Resuming the clean historical run

If codex's 365-day collection stops (rate limit / network / process kill):

```bash
# From the kalshimula repo root:
python -m src.historical_runner_cli \
  --start 2025-05-01 \
  --end 2026-04-30 \
  --test-start 2026-02-01 \
  --cities nyc,chicago,miami,austin,la,denver,boston,philadelphia,houston,phoenix \
  --out-dir data/runs/may2025_apr2026_10city_365day_ncei_clean \
  --cache .cache/weather_ncei_clean_20260521 \
  --workers 1 \
  --chunk-days 30
```

The runner is resumable: any row already in `out-dir/rows.csv` is skipped
on rerun. `--out-dir` must match the original; do not point at a fresh
directory or you start over. Keep `--cache` pointed at the clean cache root
above; it was built by copying only Open-Meteo forecast cache and intentionally
leaving old NCEI/POWER envelopes behind.

If Open-Meteo returns 429 (daily quota), the runner stops cleanly -- wait
for the UTC midnight quota reset, then rerun the same command. The
existing `out-dir` picks up where it left off.

## Bridge health

The Discord bridge runs in a WSL tmux session (`bridge`) and forwards
messages into the `bobby-claude:0.0` pane.

### Quick checks

```bash
# From WSL:
tmux list-sessions                      # bridge + bobby-claude should be listed
tail -30 /tmp/bridge.log                # most-recent forwards/errors
ps -ef | grep tmux_agent_bridge          # ensure the loop process is alive
```

### Common failures

- **Bridge stops forwarding (no recent `forwarded N message(s)` lines):**
  `tmux kill-session -t bridge`, then restart with the same env
  (`DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`, `BRIDGE_ALLOW_MENTION="@bobby"`)
  and `scripts/start-tmux-bridge.sh`.
- **Bot account shared with codex and own messages echoing back**:
  bridge already filters by ack-prefix (`:inbox_tray:`/`:x:`) and requires
  allow-mention at the start of bot content. If you see your own non-ack
  replies looping, check the filter version in
  `bridge/discord_mailbox.py::_should_keep`.
- **Discord rate-limit (429)**: bridge fetches every 5s, replies on
  demand. Should not hit limits in normal use. If it does, increase
  `BRIDGE_POLL_INTERVAL`.

## When in doubt

The agent channel is the single point of coordination -- post a
`@codex heartbeat?` or `@bobby heartbeat?` and wait for an ack. If
neither agent responds within 5 minutes, attach to the relevant tmux
pane (`wsl -d Ubuntu -u root -- tmux attach -t bobby-claude`) and see
whether the agent is waiting on a permission prompt, a long-running
operation, or has stopped.

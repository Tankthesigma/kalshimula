# Model Audit Final Pass - 2026-05-25

Scope: mainline weather/model code after cross-review with Bobby's private
market lane. This report covers model correctness, no-leak behavior, station
metadata, candidate packet transforms, and lane separation. It does not include
private Kalshi API code or paper-PnL logic.

## Result

Mainline is clean after the fixes below. The current pushed head is `91bdf6c`.

## Fixes Landed In This Pass

1. **Shared GHCND station IDs**
   - `config/station_rule_table.csv` now carries `ghcnd_id` for every high/low
     station row.
   - `StationRule` exposes `ghcnd_id` and `ghcnd_bare`.
   - Tests enforce station-rule `settlement_station` and `ghcnd_id` match
     `config/stations.yaml`.
   - This lets Bobby/private audit remove its hidden GHCND map and use the
     shared weather/station source of truth.

2. **Lone-outlier NWS guidance no-leak hardening**
   - `src/models/lone_outlier_correction.py` no longer assumes a guidance CSV
     contains only rows available at packet time.
   - It now selects the latest guidance row with
     `available_ts_utc <= predictions_nowcast.as_of_ts_utc`, per
     city/market/station/date.
   - Regression test proves a future NWS guidance row is ignored for an older
     prediction slice.

## Prior Same-Day Fixes Still In Force

- Low-market prediction export fails closed. Mainline can emit low features and
  low guidance for research, but it will not relabel a high-temperature model
  packet as a low-temperature prediction.
- Forward-test ASOS settlement fallback fetches target date plus the next UTC
  day, then applies station LST filtering.
- Heat-regime correction subtracts the packet's existing bias shift so it does
  not double-count hot-regime residuals.
- Nowcast target-day filtering uses station LST settlement date, not UTC date.

## Cross-Lane Review

Bobby reviewed mainline `dc507a6` and passed:

- no market/API/order execution surface in `src/`;
- station-rule table matches verified settlement stations;
- low-market prediction export fail-closes;
- packet PMFs/schema are normalized;
- ASOS fallback uses two-day UTC fetch plus LST filtering.

Codex reviewed Bobby's private audit artifacts:

- official NCEI daily `TMAX` is now the realized-high source for private
  settlement-grade eval;
- hourly ASOS fallback is flagged and excluded from NCEI-graded verdicts;
- the noon-pinned exploratory per-slice tradeability issue is fixed by joining
  each slice to the nearest archived snapshot by that slice's `as_of_ts_utc`;
- core private `score_packets` money path remains archive-priced and was not
  affected by the exploratory issue.

## Checks

- `ruff check .` passed.
- `pytest` passed: 699 tests.

## Open Items

- Bobby is updating private `nowcast_edge_eval` to read `ghcnd_bare` from the
  shared station table and to use NCEI/settled realized highs for parity.
- Candidate modes remain candidate-only:
  - raw is the reference/default packet;
  - adjusted, heat-corrected, and lone-outlier modes require private forward
    validation before any promotion.
- Low-temperature markets still need a separately trained low model, explicit
  low settlement validation, calibration, and private paper-PnL audit.

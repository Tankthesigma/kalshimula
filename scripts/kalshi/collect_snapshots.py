"""Phase 11a: Collect Kalshi prices at multiple time-of-day snapshots.

For each market, query candlesticks at:
  - 04:00 UTC (overnight before target day; minimal info)
  - 12:00 UTC (morning, original snapshot)
  - 20:00 UTC (afternoon, US-east close, some/most of high formed)

Use 1-min candlesticks for the last hour before each snapshot, take
the last candle (yes_bid/yes_ask) within the window.

Output:
  reports/kalshi_edge/11_snapshots.csv
  Columns: city, target, ticker, snapshot_hour_utc, last_close, last_yes_bid,
           last_yes_ask, last_mid, last_volume, candle_count
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "kalshi"))
from auth import get  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"

with (REPO_ROOT / "scripts" / "kalshi" / "city_tickers.json").open() as f:
    CITY_TICKERS = json.load(f)

START = date(2026, 5, 1)
END = date(2026, 5, 21)
SNAPSHOT_HOURS_UTC = [4, 12, 20]


def snapshot_ts(target: date, hour_utc: int) -> int:
    dt = datetime(target.year, target.month, target.day, hour_utc, 0, 0, tzinfo=timezone.utc)
    return int(dt.timestamp())


def event_ticker(series_ticker: str, target: date) -> str:
    return f"{series_ticker}-{target.strftime('%y%b%d').upper()}"


def list_event_markets(series_ticker: str, target: date) -> list[dict]:
    et = event_ticker(series_ticker, target)
    cursor = None
    out: list[dict] = []
    while True:
        params = {"event_ticker": et, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        status, body = get("/markets", params)
        if status != 200 or not isinstance(body, dict):
            break
        ms = body.get("markets") or []
        out.extend(ms)
        cursor = body.get("cursor")
        if not cursor:
            break
    return out


def fetch_snapshot(series_ticker: str, ticker: str, snapshot_t: int) -> dict | None:
    """Last 1-min candle ending <= snapshot_t. None if nothing."""
    start = snapshot_t - 3600
    status, body = get(
        f"/series/{series_ticker}/markets/{ticker}/candlesticks",
        {"start_ts": start, "end_ts": snapshot_t, "period_interval": 1},
    )
    candles = (body or {}).get("candlesticks", []) if isinstance(body, dict) else []
    if not candles:
        # fallback 60-min over a wider window
        status, body = get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            {"start_ts": snapshot_t - 6 * 3600, "end_ts": snapshot_t, "period_interval": 60},
        )
        candles = (body or {}).get("candlesticks", []) if isinstance(body, dict) else []
    if not candles:
        return None
    last = candles[-1]
    price = last.get("price", {}) or {}
    yes_bid = last.get("yes_bid", {}) or {}
    yes_ask = last.get("yes_ask", {}) or {}

    def _f(v):
        try: return float(v)
        except (TypeError, ValueError): return None

    bid = _f(yes_bid.get("close_dollars"))
    ask = _f(yes_ask.get("close_dollars"))
    mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
    return {
        "candle_count": len(candles),
        "last_ts": last.get("end_period_ts"),
        "last_close": _f(price.get("close_dollars")),
        "last_yes_bid": bid,
        "last_yes_ask": ask,
        "last_mid": mid,
        "last_volume": _f(last.get("volume_fp")) or 0,
    }


def main():
    out_rows = []
    d = START
    while d <= END:
        for city, series_ticker in CITY_TICKERS.items():
            markets = list_event_markets(series_ticker, d)
            for m in markets:
                ticker = m.get("ticker", "")
                for hour in SNAPSHOT_HOURS_UTC:
                    st = snapshot_ts(d, hour)
                    snap = fetch_snapshot(series_ticker, ticker, st)
                    row = {
                        "city": city,
                        "target": d.isoformat(),
                        "ticker": ticker,
                        "snapshot_hour_utc": hour,
                        "candle_count": (snap or {}).get("candle_count", 0),
                        "last_ts": (snap or {}).get("last_ts"),
                        "last_close": (snap or {}).get("last_close"),
                        "last_yes_bid": (snap or {}).get("last_yes_bid"),
                        "last_yes_ask": (snap or {}).get("last_yes_ask"),
                        "last_mid": (snap or {}).get("last_mid"),
                        "last_volume": (snap or {}).get("last_volume", 0),
                    }
                    out_rows.append(row)
            print(f"  {city:14} {d}: {len(markets)} markets x {len(SNAPSHOT_HOURS_UTC)} snapshots")
            time.sleep(0.1)
        d += timedelta(days=1)
    with (OUT_DIR / "11_snapshots.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} snapshot rows")


if __name__ == "__main__":
    main()

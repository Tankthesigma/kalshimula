"""Phase 2: Pull Kalshi KXHIGH market data for May 1-21 2026.

Critical fix: use a "trader snapshot time" before close to grab market_prob.
We use target_date noon UTC as the snapshot (morning in US, well before settlement).
This represents the market_prob a trader could have acted on while making a
day-ahead bet.

GET-only.
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
from auth import get  # noqa

OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(REPO_ROOT / "scripts" / "kalshi" / "city_tickers.json") as f:
    CITY_TICKERS = json.load(f)

START = date(2026, 5, 1)
END = date(2026, 5, 21)


def event_ticker(series_ticker: str, target: date) -> str:
    return f"{series_ticker}-{target.strftime('%y%b%d').upper()}"


def trader_snapshot_ts(target: date) -> int:
    """Pick a 'morning of target day' snapshot ts in UTC = target_date 12:00 UTC.

    For US east coast that's 8am EDT — a realistic time for a morning bet.
    """
    dt = datetime(target.year, target.month, target.day, 12, 0, 0, tzinfo=timezone.utc)
    return int(dt.timestamp())


def collect_one(city: str, series_ticker: str, target: date) -> tuple[list[dict], list[dict], list[dict]]:
    event_t = event_ticker(series_ticker, target)
    markets_rows: list[dict] = []
    prices_rows: list[dict] = []
    errors: list[dict] = []
    cursor = None
    raw_markets = []
    while True:
        params = {"event_ticker": event_t, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        status, body = get("/markets", params)
        if status != 200 or not isinstance(body, dict):
            errors.append({"city": city, "target": target.isoformat(), "phase": "list", "error": f"HTTP {status}: {str(body)[:120]}"})
            return markets_rows, prices_rows, errors
        ms = body.get("markets") or []
        raw_markets.extend(ms)
        cursor = body.get("cursor")
        if not cursor:
            break
    if not raw_markets:
        errors.append({"city": city, "target": target.isoformat(), "phase": "list", "error": "no markets returned"})
        return markets_rows, prices_rows, errors

    snapshot_ts = trader_snapshot_ts(target)

    for m in raw_markets:
        ticker = m.get("ticker", "")
        strike_type = m.get("strike_type") or ""
        floor_s = m.get("floor_strike")
        cap_s = m.get("cap_strike")
        result = m.get("result")
        expiration_value = m.get("expiration_value")
        close_time = m.get("close_time")
        open_time = m.get("open_time")
        title = m.get("title")
        markets_rows.append({
            "city": city, "target": target.isoformat(),
            "event_ticker": event_t, "ticker": ticker,
            "strike_type": strike_type,
            "floor_strike": floor_s, "cap_strike": cap_s,
            "result": result, "expiration_value": expiration_value,
            "open_time": open_time, "close_time": close_time,
            "title": title,
        })

        # parse open/close
        try:
            open_dt = datetime.fromisoformat(open_time.replace("Z","+00:00"))
            close_dt = datetime.fromisoformat(close_time.replace("Z","+00:00"))
        except Exception as e:
            errors.append({"city": city, "target": target.isoformat(), "phase": "parse_time", "error": f"{ticker}: {e}"})
            continue
        open_ts = int(open_dt.timestamp())
        close_ts = int(close_dt.timestamp())

        # we want candles up to snapshot_ts (or earlier of snapshot_ts/close_ts)
        target_end_ts = min(snapshot_ts, close_ts)
        # if snapshot is before the market opened, fall back to first 6 hours after open
        if target_end_ts <= open_ts + 600:
            target_end_ts = open_ts + 6 * 3600
        # span: open_ts to target_end_ts
        # Use 1-min granularity for the LAST hour of the window so we get the
        # final snapshot price.
        last_hour_start = max(open_ts, target_end_ts - 3600)
        cs1, cb1 = get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            {"start_ts": last_hour_start, "end_ts": target_end_ts, "period_interval": 1},
        )
        candles = (cb1 or {}).get("candlesticks", []) if isinstance(cb1, dict) else []

        if not candles:
            # fall back to 1-hour over entire pre-snapshot window
            cs2, cb2 = get(
                f"/series/{series_ticker}/markets/{ticker}/candlesticks",
                {"start_ts": open_ts, "end_ts": target_end_ts, "period_interval": 60},
            )
            candles = (cb2 or {}).get("candlesticks", []) if isinstance(cb2, dict) else []

        if not candles:
            # try ALL the way to close as ultimate fallback (we already had this in v1)
            cs3, cb3 = get(
                f"/series/{series_ticker}/markets/{ticker}/candlesticks",
                {"start_ts": open_ts, "end_ts": close_ts, "period_interval": 60},
            )
            candles = (cb3 or {}).get("candlesticks", []) if isinstance(cb3, dict) else []
            # we know fallback prices may be near settlement, so flag them
            late_fallback = bool(candles)
        else:
            late_fallback = False

        if not candles:
            prices_rows.append({
                "city": city, "target": target.isoformat(), "ticker": ticker,
                "snapshot_ts": snapshot_ts, "candle_count": 0,
                "first_ts": None, "last_ts": None,
                "last_close": None, "last_yes_bid": None, "last_yes_ask": None,
                "last_mid": None, "last_volume": 0,
                "late_fallback": False,
            })
            continue

        # Filter to candles whose end <= target_end_ts when possible
        valid = [c for c in candles if c.get("end_period_ts", 0) <= target_end_ts]
        if not valid:
            valid = candles  # use whatever we got

        last_c = valid[-1]
        first_c = valid[0]
        price = last_c.get("price", {}) or {}
        yes_bid = last_c.get("yes_bid", {}) or {}
        yes_ask = last_c.get("yes_ask", {}) or {}

        def _fnum(v):
            try: return float(v)
            except (TypeError, ValueError): return None

        close_d = _fnum(price.get("close_dollars"))
        bid_d = _fnum(yes_bid.get("close_dollars"))
        ask_d = _fnum(yes_ask.get("close_dollars"))
        mid = None
        if bid_d is not None and ask_d is not None:
            mid = (bid_d + ask_d) / 2.0

        prices_rows.append({
            "city": city, "target": target.isoformat(), "ticker": ticker,
            "snapshot_ts": snapshot_ts, "candle_count": len(valid),
            "first_ts": first_c.get("end_period_ts"),
            "last_ts": last_c.get("end_period_ts"),
            "last_close": close_d, "last_yes_bid": bid_d, "last_yes_ask": ask_d,
            "last_mid": mid,
            "last_volume": _fnum(last_c.get("volume_fp")) or 0,
            "late_fallback": late_fallback,
        })

    return markets_rows, prices_rows, errors


def main() -> None:
    market_rows: list[dict] = []
    price_rows: list[dict] = []
    error_rows: list[dict] = []

    d = START
    while d <= END:
        for city, ticker in CITY_TICKERS.items():
            try:
                mr, pr, er = collect_one(city, ticker, d)
            except Exception as e:
                er = [{"city": city, "target": d.isoformat(), "phase": "exception", "error": str(e)[:200]}]
                mr, pr = [], []
            market_rows.extend(mr)
            price_rows.extend(pr)
            error_rows.extend(er)
            print(f"  {city:14} {d}: {len(mr)} markets, {len(pr)} priced, {len(er)} errors")
            time.sleep(0.15)
        d += timedelta(days=1)
        print(f"== finished {d - timedelta(days=1)}: cumulative markets={len(market_rows)}, prices={len(price_rows)}, errors={len(error_rows)}")

    if market_rows:
        with open(OUT_DIR / "02_kalshi_markets.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(market_rows[0].keys()))
            w.writeheader()
            w.writerows(market_rows)
    if price_rows:
        with open(OUT_DIR / "02_kalshi_prices.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(price_rows[0].keys()))
            w.writeheader()
            w.writerows(price_rows)
    if error_rows:
        with open(OUT_DIR / "02_collection_errors.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(error_rows[0].keys()))
            w.writeheader()
            w.writerows(error_rows)
    else:
        with open(OUT_DIR / "02_collection_errors.csv", "w", newline="") as f:
            f.write("city,target,phase,error\n")
    print(f"\nWrote {len(market_rows)} markets, {len(price_rows)} prices, {len(error_rows)} errors")


if __name__ == "__main__":
    main()

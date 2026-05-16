"""CLI: `python -m src.predict --city denver --date tomorrow`.

Milestone A: pulls every Open-Meteo source, pools members, renders an ASCII
histogram + point estimate + 80% CI. NWS comparison and bias correction come
in Milestone B.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

# Windows default console is cp1252 and chokes on box-drawing glyphs.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

from src.config import Station, get_station, load_stations
from src.fetchers.openmeteo import (
    SOURCES,
    ModelDailyHigh,
    fetch_source,
    members_dataframe,
)
from src.models.ensemble import NaiveForecast, naive_forecast_from_members


def _parse_date(s: str) -> date:
    s = s.strip().lower()
    today = date.today()
    if s in {"today", "t"}:
        return today
    if s in {"tomorrow", "tmrw"}:
        return today + timedelta(days=1)
    if s in {"yesterday", "y"}:
        return today - timedelta(days=1)
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fetch_all_parallel(
    station: Station, target: date, *, use_historical: bool
) -> list[ModelDailyHigh]:
    """Hit every Open-Meteo source concurrently. Failures degrade to empty."""
    results: list[ModelDailyHigh] = []
    with ThreadPoolExecutor(max_workers=len(SOURCES)) as ex:
        futures = {
            ex.submit(
                fetch_source,
                slug,
                lat=station.lat,
                lon=station.lon,
                target=target,
                use_historical=use_historical,
            ): slug
            for slug, *_ in SOURCES
        }
        for fut in as_completed(futures):
            slug = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"  ! {slug} failed: {e}", file=sys.stderr)
                results.append(
                    ModelDailyHigh(source=slug, target_date=target, members_f=[])
                )
    # Stable order by SOURCES definition for nicer output.
    order = [s[0] for s in SOURCES]
    results.sort(key=lambda r: order.index(r.source))
    return results


def _render_ascii_histogram(fc: NaiveForecast, width: int = 32) -> str:
    if not fc.bin_probs:
        return "  (no bins)"
    max_prob = max(fc.bin_probs.values())
    modal_bin = max(fc.bin_probs, key=fc.bin_probs.get)  # type: ignore[arg-type]
    lines: list[str] = []
    for b, p in fc.bin_probs.items():
        n_blocks = int(round((p / max_prob) * width))
        bar = "█" * n_blocks if n_blocks else "▏"
        marker = "  ← MODAL" if b == modal_bin else ""
        lines.append(f"  {b:>3}°F: {bar:<{width}} {p * 100:>4.1f}%{marker}")
    return "\n".join(lines)


def _render(station: Station, target: date, fc: NaiveForecast) -> str:
    header = (
        f"{station.name} | {target.strftime('%a %b %d, %Y')} "
        f"| Settlement: {station.nws_station} (LST UTC{station.lst_offset_hours:+d})"
    )
    rule = "═" * len(header)
    src_parts = [f"{s}({n})" for s, n in fc.per_source_counts.items() if n]
    src_line = "+".join(src_parts) + f" → {fc.n_members} members"
    body = _render_ascii_histogram(fc)
    return (
        f"\n{header}\n{rule}\n{body}\n\n"
        f"Point estimate: {fc.point_f:.1f}°F  "
        f"|  80% CI: [{fc.p10_f:.1f}°F, {fc.p90_f:.1f}°F]  "
        f"|  Median: {fc.p50_f:.1f}°F\n"
        f"Sources: {src_line}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="predict",
        description="Probabilistic daily-high prediction (Milestone A naive ensemble).",
    )
    parser.add_argument(
        "--city",
        required=True,
        help=f"City slug. One of: {sorted(load_stations().keys())}",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="YYYY-MM-DD, or `today`/`tomorrow`/`yesterday`.",
    )
    args = parser.parse_args(argv)

    station = get_station(args.city)
    target = _parse_date(args.date)
    use_historical = target < date.today() - timedelta(days=2)

    print(f"\nFetching {len(SOURCES)} Open-Meteo sources for "
          f"{station.name} on {target}...", file=sys.stderr)
    sources = _fetch_all_parallel(station, target, use_historical=use_historical)
    members = members_dataframe(sources)

    if members.empty:
        print("ERROR: every Open-Meteo source returned empty.", file=sys.stderr)
        return 1

    fc = naive_forecast_from_members(members)
    print(_render(station, target, fc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

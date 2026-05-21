"""CLI: `python -m src.predict --city denver --date tomorrow`.

Milestone A: pulls every Open-Meteo source, pools members, renders an ASCII
histogram + point estimate + 80% CI. NWS comparison and bias correction come
in Milestone B.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import Station, get_station, load_stations
from src.fetchers.openmeteo import (
    SOURCES,
    ModelDailyHigh,
    fetch_source,
    members_dataframe,
)
from src.models.bias import apply_bias_correction
from src.models.ensemble import NaiveForecast, naive_forecast_from_members
from src.models.intervals import apply_empirical_intervals

# Windows default console is cp1252 and chokes on box-drawing glyphs.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        with contextlib.suppress(Exception):
            _stream.reconfigure(encoding="utf-8")


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


def _load_selected_source(path: Path, city: str) -> str | None:
    """Return the selected source for a city from source_selection output."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {"city", "selected_source"} - fieldnames
        if missing:
            raise ValueError(f"selected sources CSV missing columns: {sorted(missing)}")

        city_key = city.strip().lower()
        for row in reader:
            if row.get("city", "").strip().lower() == city_key:
                selected = row.get("selected_source", "").strip()
                return selected or None
    return None


def _members_for_selected_source(
    members: pd.DataFrame, selected_source: str | None
) -> tuple[pd.DataFrame, bool]:
    """Filter members to one selected source when possible.

    ``openmeteo_naive`` is the historical pooled baseline, so live prediction
    represents it by keeping all member rows.
    """
    if not selected_source or selected_source == "openmeteo_naive":
        return members, False

    selected = members[members["source"] == selected_source]
    if selected.empty:
        return members, False
    return selected, True


def _prediction_source(selected_source: str | None, *, selected_applied: bool) -> str:
    if selected_applied and selected_source:
        return selected_source
    return "openmeteo_naive"


def _has_city_source(table: pd.DataFrame, *, city: str, source: str) -> bool:
    required = {"city", "source"}
    if required - set(table.columns):
        return False
    city_values = table["city"].astype(str).str.lower()
    source_values = table["source"].astype(str)
    return bool(((city_values == city.lower()) & (source_values == source)).any())


def _apply_prediction_artifacts(
    *,
    city: str,
    source: str,
    target: date,
    point_f: float,
    bias_table_path: Path | None = None,
    interval_table_path: Path | None = None,
) -> tuple[pd.Series, list[str]]:
    """Apply optional trained bias and interval artifacts to one prediction."""
    row = pd.DataFrame(
        [
            {
                "city": city,
                "source": source,
                "target_date": target.isoformat(),
                "point_f": point_f,
            }
        ]
    )
    warnings: list[str] = []

    if bias_table_path is not None:
        bias_table = pd.read_csv(bias_table_path)
        if _has_city_source(bias_table, city=city, source=source):
            row = apply_bias_correction(row, bias_table)
        else:
            warnings.append(
                f"no bias row for {city}/{source}; leaving point uncorrected"
            )

    if interval_table_path is not None:
        interval_table = pd.read_csv(interval_table_path)
        if _has_city_source(interval_table, city=city, source=source):
            row = apply_empirical_intervals(row, interval_table)
        else:
            warnings.append(
                f"no interval row for {city}/{source}; omitting calibrated interval"
            )

    return row.iloc[0], warnings


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


def _render_calibration(row: pd.Series | None) -> str:
    if row is None:
        return ""

    parts = [f"Model source: {row['source']}"]
    corrected_point = row.get("corrected_point_f")
    if corrected_point is not None and pd.notna(corrected_point):
        parts.append(f"Corrected point: {float(corrected_point):.1f}°F")

    lower = row.get("interval_lower_f")
    upper = row.get("interval_upper_f")
    if lower is not None and upper is not None and pd.notna(lower) and pd.notna(upper):
        parts.append(f"Empirical interval: [{float(lower):.1f}°F, {float(upper):.1f}°F]")

    return "Calibration: " + "  |  ".join(parts) + "\n"


def _render(
    station: Station,
    target: date,
    fc: NaiveForecast,
    *,
    calibration: pd.Series | None = None,
) -> str:
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
        f"{_render_calibration(calibration)}"
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
    parser.add_argument(
        "--selected-sources",
        type=Path,
        help=(
            "Optional source_selection/selected_sources.csv. When the city "
            "has an individual selected Open-Meteo source, predict from that "
            "source's members; openmeteo_naive keeps the pooled baseline."
        ),
    )
    parser.add_argument(
        "--bias-table",
        type=Path,
        help="Optional train_eval/bias_table.csv to apply a point correction.",
    )
    parser.add_argument(
        "--interval-table",
        type=Path,
        help="Optional train_eval/interval_table.csv to apply empirical intervals.",
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

    selected_source = None
    selected_applied = False
    if args.selected_sources:
        try:
            selected_source = _load_selected_source(args.selected_sources, args.city)
        except (OSError, ValueError) as error:
            print(f"ERROR: could not read selected sources: {error}", file=sys.stderr)
            return 1

        members, selected_applied = _members_for_selected_source(
            members, selected_source
        )
        if selected_source is None:
            print(
                f"  ! no selected source for {args.city}; using pooled members",
                file=sys.stderr,
            )
        elif selected_applied:
            print(f"  = using selected source: {selected_source}", file=sys.stderr)
        elif selected_source == "openmeteo_naive":
            print(
                "  = selected source is openmeteo_naive; using pooled members",
                file=sys.stderr,
            )
        else:
            print(
                f"  ! selected source {selected_source!r} returned no members; "
                "using pooled members",
                file=sys.stderr,
            )

    fc = naive_forecast_from_members(members)
    calibration = None
    if args.bias_table or args.interval_table:
        source = _prediction_source(selected_source, selected_applied=selected_applied)
        try:
            calibration, warnings = _apply_prediction_artifacts(
                city=args.city,
                source=source,
                target=target,
                point_f=fc.point_f,
                bias_table_path=args.bias_table,
                interval_table_path=args.interval_table,
            )
        except (OSError, ValueError) as error:
            print(f"ERROR: could not apply model artifacts: {error}", file=sys.stderr)
            return 1
        for warning in warnings:
            print(f"  ! {warning}", file=sys.stderr)

    print(_render(station, target, fc, calibration=calibration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

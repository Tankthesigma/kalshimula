"""Tiny shared helpers used by the source-level weather fetchers.

Keep this module dependency-free and pure. Conversions, defensive numeric
parsing, ISO date matching, station normalization, and compact error
formatting — nothing that touches the network or the filesystem belongs here.
"""

from __future__ import annotations

from datetime import date

# Canonical missing-value markers seen across NWS, NCEI, POWER, ASOS payloads.
_MISSING_MARKERS = frozenset(
    {
        "M",
        "NA",
        "N/A",
        "NAN",
        "NONE",
        "NULL",
        "MISSING",
        "",
    }
)


def c_to_f(value: int | float) -> float:
    """Celsius to Fahrenheit."""
    return float(value) * 9.0 / 5.0 + 32.0


def c_tenths_to_f(value: int | float) -> float:
    """NCEI-style tenths-of-a-degree-Celsius to Fahrenheit.

    NCEI Access Data Service returns TMAX as tenths of degrees Celsius.
    """
    return c_to_f(float(value) / 10.0)


def is_missing_value(value: object) -> bool:
    """True when ``value`` matches a known missing-data sentinel.

    Recognizes ``None``, ``bool`` (always missing — see safe_float for why),
    empty strings, whitespace-only strings, and the canonical text markers
    used by NCEI/POWER/ASOS feeds (``M``, ``NA``, ``N/A``, ``NaN``, ``None``,
    ``NULL``, ``missing``).
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return value.strip().upper() in _MISSING_MARKERS
    return False


def safe_float(value: object) -> float | None:
    """Coerce arbitrary JSON/CSV values to float, or None when not numeric.

    Treats common sentinels for missing data ("", "M", "NA", None, bool) as
    missing. ``True``/``False`` would otherwise convert to 1.0/0.0 silently,
    which is almost never what a weather payload means.
    """
    if is_missing_value(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def safe_int(value: object) -> int | None:
    """Coerce arbitrary JSON/CSV values to int, or None when not a whole number.

    Accepts ints, floats that are exact integers (``12.0`` → 12, ``12.3`` →
    None), and strings of either form. Rejects missing markers and anything
    non-numeric.
    """
    if is_missing_value(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        s = value.strip()
        try:
            return int(s)
        except ValueError:
            pass
        try:
            f = float(s)
        except ValueError:
            return None
        if f != f:
            return None
        if f.is_integer():
            return int(f)
        return None
    return None


def normalize_station(value: str) -> str:
    """Strip whitespace and uppercase a station code.

    Tolerant of non-strings: returns ``""`` for anything else.
    """
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def iso_date_prefix_matches(value: object, target: date) -> bool:
    """True when ``value`` is a string starting with ``target.isoformat()``.

    Used to match ISO-8601 datetimes like ``"2025-01-02T13:00:00-05:00"`` or
    NCEI's ``"2025-01-02T00:00:00"`` against a calendar date.
    """
    if not isinstance(value, str):
        return False
    return value.startswith(target.isoformat())


def compact_error(exc: BaseException, max_len: int = 160) -> str:
    """Render an exception as ``ExceptionType: message`` truncated to ``max_len``.

    Diagnostics surfaces (smoke harness, source-quality reports, CLI output)
    need a one-line representation of a caught exception that can sit safely
    in a CSV cell without leaking newlines or huge tracebacks.
    """
    name = type(exc).__name__
    message = str(exc).strip().replace("\r", " ").replace("\n", " ")
    rendered = f"{name}: {message}" if message else name
    if len(rendered) <= max_len:
        return rendered
    if max_len <= 1:
        return rendered[:max_len]
    return rendered[: max_len - 1] + "…"

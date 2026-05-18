"""Tiny shared helpers used by the source-level weather fetchers.

Keep this module dependency-free and pure. Conversions, defensive numeric
parsing, and ISO date matching — nothing that touches the network or the
filesystem belongs here.
"""

from __future__ import annotations

from datetime import date


def c_to_f(value: int | float) -> float:
    """Celsius to Fahrenheit."""
    return float(value) * 9.0 / 5.0 + 32.0


def c_tenths_to_f(value: int | float) -> float:
    """NCEI-style tenths-of-a-degree-Celsius to Fahrenheit.

    NCEI Access Data Service returns TMAX as tenths of degrees Celsius.
    """
    return c_to_f(float(value) / 10.0)


def safe_float(value: object) -> float | None:
    """Coerce arbitrary JSON/CSV values to float, or None when not numeric.

    Treats common sentinels for missing data ("", "M", "NA", None, bool) as
    missing. ``True``/``False`` would otherwise convert to 1.0/0.0 silently,
    which is almost never what a weather payload means.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.upper() in {"M", "NA", "N/A", "NAN", "NONE", "NULL"}:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def iso_date_prefix_matches(value: object, target: date) -> bool:
    """True when ``value`` is a string starting with ``target.isoformat()``.

    Used to match ISO-8601 datetimes like ``"2025-01-02T13:00:00-05:00"`` or
    NCEI's ``"2025-01-02T00:00:00"`` against a calendar date.
    """
    if not isinstance(value, str):
        return False
    return value.startswith(target.isoformat())

"""Pure-math tests for Phase 1-7 helpers.

Live Kalshi API tests are intentionally skipped.
"""

from __future__ import annotations

import pytest


def test_round_half_up_basic():
    from scripts.kalshi.calibration_check import round_half_up
    assert round_half_up(0.0) == 0
    assert round_half_up(0.499999) == 0
    assert round_half_up(0.5) == 1
    assert round_half_up(72.4) == 72
    assert round_half_up(72.5) == 73
    assert round_half_up(-0.5) == 0  # half-up rounds toward +inf
    assert round_half_up(-0.6) == -1


def test_raw_prob_basic():
    from scripts.kalshi.calibration_check import raw_prob
    residuals = [-2.0, -1.0, 0.0, 1.0, 2.0]
    # P(actual >= point + 0) = P(residual >= 0) = 3/5
    assert raw_prob(70.0, 70, residuals) == pytest.approx(0.6)
    # P(actual >= point + 1) = P(residual >= 1) = 2/5
    assert raw_prob(70.0, 71, residuals) == pytest.approx(0.4)
    # P(actual >= point + 3) = P(residual >= 3) = 0
    assert raw_prob(70.0, 73, residuals) == pytest.approx(0.0)
    # P(actual >= point - 3) = P(residual >= -3) = 1
    assert raw_prob(70.0, 67, residuals) == pytest.approx(1.0)
    # Empty residuals returns 0
    assert raw_prob(70.0, 70, []) == 0.0


def test_prob_helpers_match_definitions():
    from scripts.kalshi.build_edge_table import prob_geq, prob_gt, prob_lt, prob_between
    residuals = [-2.0, -1.0, 0.0, 1.0, 2.0]
    point = 70.0
    # >= 70 -> 3/5
    assert prob_geq(point, 70.0, residuals) == pytest.approx(0.6)
    # > 70  -> 2/5 (strict)
    assert prob_gt(point, 70.0, residuals) == pytest.approx(0.4)
    # < 70  -> 2/5 (strict)
    assert prob_lt(point, 70.0, residuals) == pytest.approx(0.4)
    # between 69 and 71 inclusive -> residuals in [-1, 1] = 3/5
    assert prob_between(point, 69.0, 71.0, residuals) == pytest.approx(0.6)
    # P_geq + P_lt should equal 1 - P(actual == X) — for discrete residuals that's
    # 1 - count(residual==0)/N = 1 - 1/5 = 0.8
    assert prob_geq(point, 70.0, residuals) + prob_lt(point, 70.0, residuals) == pytest.approx(1.0)


def test_recalibrate_returns_global_when_no_per_city():
    from scripts.kalshi.calibration_check import recalibrate
    per_city = {("nyc", 5): 0.7}
    glob = {3: 0.3, 5: 0.6}
    # raw 0.5 -> bucket 5, per-city wins
    out, scope = recalibrate(0.5, "nyc", per_city, glob)
    assert out == 0.7 and scope == "city_source"
    # raw 0.3 -> bucket 3, no per-city, falls back to global
    out, scope = recalibrate(0.3, "nyc", per_city, glob)
    assert out == 0.3 and scope == "global"
    # raw 0.1 -> bucket 1, no per-city or global, falls back to raw
    out, scope = recalibrate(0.1, "nyc", per_city, glob)
    assert out == 0.1 and scope == "none"


def test_brier_and_ece_match_manual():
    from scripts.kalshi.calibration_check import brier, ece
    events = [
        {"p": 0.0, "outcome": 0},
        {"p": 1.0, "outcome": 1},
        {"p": 0.5, "outcome": 0},
        {"p": 0.5, "outcome": 1},
    ]
    # Manual Brier: (0-0)^2 + (1-1)^2 + (0.5-0)^2 + (0.5-1)^2 = 0 + 0 + 0.25 + 0.25 = 0.5
    # Divided by 4 = 0.125
    assert brier(events, "p") == pytest.approx(0.125)
    # ECE: bucket 0 has [0.0, 0]; bucket 9 has [1.0, 1]; bucket 4 has [0.5/0, 0.5/1] (n=2, mean=0.5, obs=0.5, gap=0)
    # Wait — bucket index for p=0.0 = 0; p=1.0 -> bin 9 (clipped); p=0.5 -> bin 5; both 0.5s in bin 5
    # bin 0: n=1, mean=0, obs=0 -> gap 0
    # bin 9: n=1, mean=1, obs=1 -> gap 0
    # bin 5: n=2, mean=0.5, obs=0.5 -> gap 0
    assert ece(events, "p") == pytest.approx(0.0)


def test_kalshi_fee_round_trip_at_0_5():
    from scripts.kalshi.simulate_pnl import kalshi_fee
    # 0.07 * 0.5 * 0.5 = 0.0175
    assert kalshi_fee(0.5, contracts=1.0) == pytest.approx(0.0175)
    # 0 at edges
    assert kalshi_fee(0.0, contracts=1.0) == pytest.approx(0.0)
    assert kalshi_fee(1.0, contracts=1.0) == pytest.approx(0.0)
    # Scales with contracts
    assert kalshi_fee(0.5, contracts=10) == pytest.approx(0.175)


def test_kalshi_fee_clipped_at_bounds():
    from scripts.kalshi.simulate_pnl import kalshi_fee
    # Out-of-range prices get clipped
    assert kalshi_fee(-0.1, contracts=1.0) == pytest.approx(0.0)
    assert kalshi_fee(1.5, contracts=1.0) == pytest.approx(0.0)


def test_event_ticker_format():
    from scripts.kalshi.collect_markets import event_ticker
    from datetime import date
    # Format: KXHIGHNY-26MAY01
    assert event_ticker("KXHIGHNY", date(2026, 5, 1)) == "KXHIGHNY-26MAY01"
    assert event_ticker("KXHIGHCHI", date(2026, 12, 25)) == "KXHIGHCHI-26DEC25"


def test_safe_float_handles_missing():
    from scripts.kalshi.simulate_pnl import safe_float
    assert safe_float("1.5") == 1.5
    assert safe_float("") is None
    assert safe_float(None) is None
    assert safe_float("nope") is None
    assert safe_float("0") == 0.0

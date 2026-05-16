"""Naive multi-source ensemble.

Pool every member (and every deterministic forecast) from every Open-Meteo
source into one big bag of numbers, then convert that bag into:

* a point estimate (mean),
* a percentile-based interval,
* 1°F-bin probabilities by histogram counting.

This is the Milestone A baseline. Later milestones replace it with a calibrated
ML stack but keep the same output interface.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NaiveForecast:
    n_members: int
    point_f: float
    p10_f: float
    p50_f: float
    p90_f: float
    bin_probs: dict[int, float]            # bin (rounded-int °F) → probability
    per_source_counts: dict[str, int]      # source slug → member count


def _round_to_nws_int(x: float) -> int:
    """NWS reports daily high as integer °F. Use banker's-free half-up rounding."""
    return int(math.floor(x + 0.5))


def naive_forecast_from_members(
    members: pd.DataFrame, *, bin_min_prob: float = 0.005
) -> NaiveForecast:
    """Build a NaiveForecast from the long-format dataframe.

    Bins are integer-°F (the NWS reporting convention). Bins with probability
    below `bin_min_prob` are dropped from the returned dict to keep CLI output
    readable, but the dropped tail mass is uniformly redistributed across the
    surviving bins so probabilities still sum to 1.
    """
    if members.empty:
        raise ValueError("No forecast members returned — every source failed?")

    temps = members["temp_f"].to_numpy(dtype=float)
    n = len(temps)
    point = float(temps.mean())
    p10, p50, p90 = (float(np.percentile(temps, q)) for q in (10, 50, 90))

    counts: Counter[int] = Counter(_round_to_nws_int(t) for t in temps)
    raw_probs = {b: c / n for b, c in counts.items()}

    kept = {b: p for b, p in raw_probs.items() if p >= bin_min_prob}
    if kept:
        kept_total = sum(kept.values())
        if kept_total > 0:
            kept = {b: p / kept_total for b, p in kept.items()}
    else:
        kept = raw_probs

    kept = dict(sorted(kept.items()))
    per_source = (
        members.groupby("source").size().to_dict() if not members.empty else {}
    )

    return NaiveForecast(
        n_members=n,
        point_f=point,
        p10_f=p10,
        p50_f=p50,
        p90_f=p90,
        bin_probs=kept,
        per_source_counts=per_source,
    )

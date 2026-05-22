"""Phase 4: Compare model vs market.

Reads 03_edge_table.csv, computes Brier / log-loss / ECE for:
  - model raw, model recal, market alone
  - blends: 75/25, 50/50, 25/75 (model/market)

Breakdowns: by city, strike distance from corrected_point_f, price bucket,
contract type. Time of day skipped (price_time_utc is end-of-candlestick
period epoch; we use that for late-vs-early bucketing).

Outputs:
  reports/kalshi_edge/04_model_vs_market.csv
  reports/kalshi_edge/04_model_vs_market.md
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "kalshi_edge"

PRICE_BUCKETS = [
    (0.05, 0.15), (0.15, 0.30), (0.30, 0.45),
    (0.45, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 0.95),
]


def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_edge_rows() -> list[dict]:
    rows = []
    with (OUT_DIR / "03_edge_table.csv").open() as f:
        for r in csv.DictReader(f):
            if r.get("comparable_flag") != "yes":
                continue
            for fld in ("model_prob_raw", "model_prob_recal", "market_prob",
                         "strike_low", "strike_high", "actual_high_f", "abs_edge"):
                r[fld] = safe_float(r[fld])
            try:
                r["outcome_yes"] = int(float(r["outcome_yes"]))
            except (TypeError, ValueError):
                continue
            rows.append(r)
    return rows


def brier(rows, key):
    return sum((r[key] - r["outcome_yes"]) ** 2 for r in rows) / len(rows) if rows else float("nan")


def log_loss(rows, key, eps=1e-12):
    if not rows:
        return float("nan")
    total = 0.0
    for r in rows:
        p = min(max(r[key], eps), 1 - eps)
        y = r["outcome_yes"]
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(rows)


def ece(rows, key, n_buckets=10):
    if not rows:
        return float("nan")
    buckets = defaultdict(list)
    for r in rows:
        bi = min(int(r[key] * n_buckets), n_buckets - 1)
        buckets[bi].append(r)
    total = 0.0
    for evs in buckets.values():
        n = len(evs)
        mean_p = sum(e[key] for e in evs) / n
        obs = sum(e["outcome_yes"] for e in evs) / n
        total += (n / len(rows)) * abs(mean_p - obs)
    return total


def attach_blends(rows: list[dict]) -> None:
    for r in rows:
        m = r["model_prob_recal"]
        mk = r["market_prob"]
        r["blend_75m_25mk"] = 0.75 * m + 0.25 * mk
        r["blend_50_50"]    = 0.50 * m + 0.50 * mk
        r["blend_25m_75mk"] = 0.25 * m + 0.75 * mk


def attach_strike_distance(rows):
    """Approximate strike distance from corrected_point_f.

    We don't have corrected_point_f in edge_table; use (strike_low+strike_high)/2
    vs actual_high_f as a proxy. For threshold types, use the single strike.
    """
    for r in rows:
        actual = r["actual_high_f"]
        if r["contract_type"] == "threshold_greater":
            ref = r["strike_low"]
        elif r["contract_type"] == "threshold_less":
            ref = r["strike_high"]
        elif r["contract_type"] == "bin_between":
            lo, hi = r["strike_low"], r["strike_high"]
            ref = (lo + hi) / 2 if lo is not None and hi is not None else None
        else:
            ref = None
        r["strike_distance"] = (ref - actual) if (ref is not None and actual is not None) else None


def block(rows, name):
    if not rows:
        return None
    keys = ["model_prob_raw", "model_prob_recal", "market_prob",
            "blend_75m_25mk", "blend_50_50", "blend_25m_75mk"]
    out = {"group": name, "n": len(rows)}
    for k in keys:
        out[f"brier_{k}"] = brier(rows, k)
        out[f"logloss_{k}"] = log_loss(rows, k)
        out[f"ece_{k}"] = ece(rows, k)
    return out


def filter_by_price_bucket(rows, lo, hi):
    return [r for r in rows if r["market_prob"] is not None and lo <= r["market_prob"] < hi]


def filter_by_strike_distance(rows, lo, hi):
    return [r for r in rows if r["strike_distance"] is not None and lo <= r["strike_distance"] < hi]


def main():
    rows = load_edge_rows()
    attach_blends(rows)
    attach_strike_distance(rows)
    print(f"loaded {len(rows)} comparable edge rows")

    blocks: list[dict] = []

    blocks.append(block(rows, "ALL"))

    for ctype in sorted({r["contract_type"] for r in rows}):
        blocks.append(block([r for r in rows if r["contract_type"] == ctype], f"contract_type={ctype}"))

    for city in sorted({r["city"] for r in rows}):
        blocks.append(block([r for r in rows if r["city"] == city], f"city={city}"))

    for lo, hi in PRICE_BUCKETS:
        sub = filter_by_price_bucket(rows, lo, hi)
        blocks.append(block(sub, f"price_bucket={lo:.2f}-{hi:.2f}"))

    # strike-distance buckets in F (signed = strike - actual)
    sd_buckets = [(-12, -6), (-6, -3), (-3, -1), (-1, 1), (1, 3), (3, 6), (6, 12)]
    for lo, hi in sd_buckets:
        sub = filter_by_strike_distance(rows, lo, hi)
        blocks.append(block(sub, f"strike_dist_{lo:+d}_to_{hi:+d}"))

    # Drop Nones
    blocks = [b for b in blocks if b]

    # Write CSV
    fieldnames = list(blocks[0].keys())
    with (OUT_DIR / "04_model_vs_market.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for b in blocks:
            w.writerow(b)

    # Render markdown summary
    md = ["# Phase 4 — Model vs Market", ""]
    md.append("Brier / log-loss / ECE on comparable edge events. Lower is better.")
    md.append("")
    md.append("## Aggregate")
    md.append("")
    agg = blocks[0]
    md.append("| metric | model raw | model recal | market | blend 75m/25mk | blend 50/50 | blend 25m/75mk |")
    md.append("|---|---|---|---|---|---|---|")
    md.append(f"| Brier  | {agg['brier_model_prob_raw']:.4f} | {agg['brier_model_prob_recal']:.4f} | {agg['brier_market_prob']:.4f} | {agg['brier_blend_75m_25mk']:.4f} | {agg['brier_blend_50_50']:.4f} | {agg['brier_blend_25m_75mk']:.4f} |")
    md.append(f"| LogLoss| {agg['logloss_model_prob_raw']:.4f} | {agg['logloss_model_prob_recal']:.4f} | {agg['logloss_market_prob']:.4f} | {agg['logloss_blend_75m_25mk']:.4f} | {agg['logloss_blend_50_50']:.4f} | {agg['logloss_blend_25m_75mk']:.4f} |")
    md.append(f"| ECE    | {agg['ece_model_prob_raw']:.4f} | {agg['ece_model_prob_recal']:.4f} | {agg['ece_market_prob']:.4f} | {agg['ece_blend_75m_25mk']:.4f} | {agg['ece_blend_50_50']:.4f} | {agg['ece_blend_25m_75mk']:.4f} |")
    md.append("")
    md.append("(n = {})".format(agg["n"]))
    md.append("")
    md.append("Full breakdown in `04_model_vs_market.csv`.")

    md.append("")
    md.append("## Per-city Brier (recal vs market)")
    md.append("")
    md.append("| city | n | model_recal | market | blend 50/50 |")
    md.append("|---|---|---|---|---|")
    for b in blocks:
        if not b["group"].startswith("city="):
            continue
        city = b["group"].split("=", 1)[1]
        md.append(f"| {city} | {b['n']} | {b['brier_model_prob_recal']:.4f} | {b['brier_market_prob']:.4f} | {b['brier_blend_50_50']:.4f} |")

    md.append("")
    md.append("## Per price bucket")
    md.append("")
    md.append("| price bucket | n | model_recal | market | blend 50/50 |")
    md.append("|---|---|---|---|---|")
    for b in blocks:
        if not b["group"].startswith("price_bucket="):
            continue
        bucket = b["group"].split("=", 1)[1]
        md.append(f"| {bucket} | {b['n']} | {b['brier_model_prob_recal']:.4f} | {b['brier_market_prob']:.4f} | {b['brier_blend_50_50']:.4f} |")

    md.append("")
    md.append("## Verdict")
    md.append("")
    # Compare model_recal vs market vs best blend on Brier
    metrics = {
        "model_recal": agg["brier_model_prob_recal"],
        "market": agg["brier_market_prob"],
        "blend_50_50": agg["brier_blend_50_50"],
        "blend_75m_25mk": agg["brier_blend_75m_25mk"],
        "blend_25m_75mk": agg["brier_blend_25m_75mk"],
    }
    winner = min(metrics, key=metrics.get)
    md.append(f"- **Best aggregate Brier**: `{winner}` at {metrics[winner]:.4f}")
    md.append(f"- model_recal Brier vs market Brier: {agg['brier_model_prob_recal']:.4f} vs {agg['brier_market_prob']:.4f}")
    if agg['brier_model_prob_recal'] < agg['brier_market_prob']:
        md.append("- Model beats market on aggregate Brier.")
    else:
        md.append("- Market beats model on aggregate Brier — blending or filters needed.")

    (OUT_DIR / "04_model_vs_market.md").write_text("\n".join(md))
    print(f"wrote {OUT_DIR / '04_model_vs_market.csv'} and 04_model_vs_market.md")


if __name__ == "__main__":
    main()

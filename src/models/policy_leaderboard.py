"""Policy leaderboard helpers for model-intelligence reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LEADERBOARD_COLUMNS = [
    "source",
    "n_cities",
    "n_predictions",
    "n_events",
    "mae",
    "bias",
    "brier",
    "ece",
    "logloss",
    "city_stability",
    "worst_city",
    "worst_city_mae",
    "promoted_city_sources",
    "leakage_safe",
]


def build_policy_leaderboard(
    walkforward_city_source_summary: pd.DataFrame,
    *,
    source_contrarian_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact policy leaderboard from walk-forward summaries."""
    required = {"city", "source", "n_predictions", "n_events", "mae", "bias"}
    missing = required - set(walkforward_city_source_summary.columns)
    if missing:
        raise ValueError(f"walkforward summary missing columns: {sorted(missing)}")
    promoted_counts = _promoted_counts(source_contrarian_summary)
    rows = []
    for source, group in walkforward_city_source_summary.groupby("source", sort=True):
        worst = group.sort_values("mae", ascending=False).iloc[0]
        rows.append(
            {
                "source": source,
                "n_cities": int(group["city"].nunique()),
                "n_predictions": int(group["n_predictions"].sum()),
                "n_events": int(group["n_events"].sum()),
                "mae": _mean_column(group, "mae"),
                "bias": _mean_column(group, "bias"),
                "brier": _mean_column(group, "brier_raw"),
                "ece": _mean_column(group, "ece_raw"),
                "logloss": _mean_column(group, "logloss_raw"),
                "city_stability": _mean_column(group, "stability_score"),
                "worst_city": worst["city"],
                "worst_city_mae": float(worst["mae"]),
                "promoted_city_sources": int(promoted_counts.get(source, 0)),
                "leakage_safe": True,
            }
        )
    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS).sort_values(
        ["mae", "brier", "city_stability"],
        na_position="last",
    )


def render_policy_leaderboard(leaderboard: pd.DataFrame) -> str:
    """Render a markdown policy leaderboard."""
    lines = [
        "# Policy Leaderboard",
        "",
        "This leaderboard is model-only and contains no Kalshi prices or trading instructions.",
        "",
    ]
    if leaderboard.empty:
        return "\n".join([*lines, "No rows.", ""])
    lines.append("| source | n cities | n predictions | MAE | Brier | ECE | worst city | promoted combos |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for row in leaderboard.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.n_cities} | {row.n_predictions} | {_fmt(row.mae)} | "
            f"{_fmt(row.brier)} | {_fmt(row.ece)} | {row.worst_city} ({_fmt(row.worst_city_mae)}) | "
            f"{row.promoted_city_sources} |"
        )
    return "\n".join(lines) + "\n"


def write_policy_leaderboard(
    *,
    walkforward_summary_path: Path,
    output_dir: Path,
    source_contrarian_summary_path: Path | None = None,
) -> pd.DataFrame:
    """Read summaries and write leaderboard CSV/markdown."""
    walkforward = pd.read_csv(walkforward_summary_path)
    contrarian = (
        pd.read_csv(source_contrarian_summary_path)
        if source_contrarian_summary_path is not None and source_contrarian_summary_path.exists()
        else None
    )
    leaderboard = build_policy_leaderboard(
        walkforward,
        source_contrarian_summary=contrarian,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output_dir / "policy_leaderboard.csv", index=False)
    (output_dir / "policy_leaderboard.md").write_text(
        render_policy_leaderboard(leaderboard),
        encoding="utf-8",
    )
    return leaderboard


def _promoted_counts(source_contrarian_summary: pd.DataFrame | None) -> dict[str, int]:
    if source_contrarian_summary is None or source_contrarian_summary.empty:
        return {}
    if "promoted" not in source_contrarian_summary.columns:
        return {}
    promoted = source_contrarian_summary[
        source_contrarian_summary["promoted"].astype(str).str.lower() == "true"
    ]
    return promoted.groupby("source").size().astype(int).to_dict()


def _mean_column(group: pd.DataFrame, column: str) -> object:
    if column not in group.columns:
        return pd.NA
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if values.empty:
        return pd.NA
    return float(values.mean())


def _fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"

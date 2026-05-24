import pandas as pd

from src.models.source_provenance import (
    build_source_provenance_diagnostics,
    summarize_source_provenance,
)


def test_source_provenance_flags_identical_sources() -> None:
    rows = pd.DataFrame(
        [
            _row("chicago", "2026-01-01", "gfs_ens", 70),
            _row("chicago", "2026-01-01", "hrrr", 70),
            _row("chicago", "2026-01-02", "gfs_ens", 72),
            _row("chicago", "2026-01-02", "hrrr", 72),
        ]
    )

    summary = summarize_source_provenance(rows)

    pair = summary.iloc[0]
    assert pair["duplicate_flag"]
    assert pair["identical_rate"] == 1.0
    assert pair["max_abs_diff_f"] == 0.0


def test_source_provenance_does_not_flag_distinct_sources() -> None:
    rows = pd.DataFrame(
        [
            _row("chicago", "2026-01-01", "gfs_ens", 70),
            _row("chicago", "2026-01-01", "ecmwf_ens", 73),
            _row("chicago", "2026-01-02", "gfs_ens", 72),
            _row("chicago", "2026-01-02", "ecmwf_ens", 75),
        ]
    )

    summary = summarize_source_provenance(rows)

    assert not summary.iloc[0]["duplicate_flag"]


def test_source_provenance_report_mentions_duplicate_flags() -> None:
    diagnostics = build_source_provenance_diagnostics(
        pd.DataFrame(
            [
                _row("chicago", "2026-01-01", "gfs_ens", 70),
                _row("chicago", "2026-01-01", "hrrr", 70),
            ]
        )
    )

    assert "Duplicate Source Flags" in diagnostics.report
    assert diagnostics.manifest["row_counts"]["duplicate_flags"] == 1


def _row(city: str, target_date: str, source: str, point: float) -> dict:
    return {
        "city": city,
        "target_date": target_date,
        "source": source,
        "point_f": point,
    }

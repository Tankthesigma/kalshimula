import json

import pandas as pd

from src.source_contrarian_diagnostics_cli import main


def test_source_contrarian_diagnostics_cli_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "source_contrarian"
    pd.DataFrame(
        [
            _row("denver", "2025-01-01", "openmeteo_naive", 70, 73),
            _row("denver", "2025-01-01", "gfs_ens", 72, 73),
            _row("denver", "2025-01-02", "openmeteo_naive", 70, 68),
            _row("denver", "2025-01-02", "gfs_ens", 72, 68),
        ]
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--out-dir",
            str(output_dir),
            "--offsets",
            "-2,0,2",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "daily_source_deltas.csv").exists()
    assert (output_dir / "monthly_source_metrics.csv").exists()
    assert (output_dir / "source_contrarian_summary.csv").exists()
    assert (output_dir / "source_threshold_grid.csv").exists()
    assert (output_dir / "contrarian_value_index.md").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["offsets"] == [-2.0, 0.0, 2.0]
    assert manifest["row_counts"]["input_rows"] == 4
    assert "Wrote source contrarian diagnostics" in capsys.readouterr().out


def _row(city: str, target_date: str, source: str, point: float, actual: float) -> dict:
    return {
        "city": city,
        "target_date": target_date,
        "source": source,
        "point_f": point,
        "actual_high_f": actual,
    }

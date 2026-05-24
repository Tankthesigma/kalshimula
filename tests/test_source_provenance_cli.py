import pandas as pd

from src.source_provenance_cli import main


def test_source_provenance_cli_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    out_dir = tmp_path / "provenance"
    pd.DataFrame(
        [
            {"city": "chicago", "target_date": "2026-01-01", "source": "gfs_ens", "point_f": 70},
            {"city": "chicago", "target_date": "2026-01-01", "source": "hrrr", "point_f": 70},
        ]
    ).to_csv(input_path, index=False)

    exit_code = main(["--input", str(input_path), "--out-dir", str(out_dir)])

    assert exit_code == 0
    assert (out_dir / "source_provenance.csv").exists()
    assert (out_dir / "source_provenance_report.md").exists()
    assert (out_dir / "source_provenance_manifest.json").exists()
    assert "Wrote source provenance diagnostics" in capsys.readouterr().out

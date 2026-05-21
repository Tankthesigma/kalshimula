import pandas as pd

from src.interval_policy_cli import main


def test_interval_policy_cli_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    source_path = tmp_path / "recommended_sources.csv"
    output_dir = tmp_path / "model_policy"
    rows = pd.DataFrame(
        [
            _row("2025-01-01", 70, 70),
            _row("2025-01-02", 70, 72),
            _row("2025-01-03", 70, 74),
            _row("2025-01-04", 70, 78),
            _row("2025-01-05", 70, 74),
            _row("2025-01-06", 70, 76),
            _row("2025-01-07", 70, 74),
            _row("2025-01-08", 70, 76),
        ]
    )
    rows.to_csv(input_path, index=False)
    pd.DataFrame([{"city": "denver", "selected_source": "gfs_ens"}]).to_csv(
        source_path, index=False
    )

    code = main(
        [
            "--input",
            str(input_path),
            "--recommended-sources",
            str(source_path),
            "--out-dir",
            str(output_dir),
            "--validation-start",
            "2025-01-05",
            "--test-start",
            "2025-01-07",
            "--alphas",
            "0.5,0.1",
        ]
    )

    assert code == 0
    assert "city/source policies" in capsys.readouterr().out
    assert (output_dir / "interval_policy.csv").exists()
    assert (output_dir / "interval_policy_comparison.csv").exists()
    assert (output_dir / "interval_table.csv").exists()


def _row(target_date: str, point_f: float, actual_high_f: float) -> dict:
    return {
        "city": "denver",
        "target_date": target_date,
        "source": "gfs_ens",
        "point_f": point_f,
        "actual_high_f": actual_high_f,
        "absolute_error_f": abs(point_f - actual_high_f),
    }

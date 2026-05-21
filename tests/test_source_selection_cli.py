import pandas as pd

from src.source_selection_cli import main


def test_source_selection_cli_writes_outputs(tmp_path, capsys) -> None:
    validation_scores_path = tmp_path / "validation_scores.csv"
    evaluation_path = tmp_path / "evaluation.csv"
    output_dir = tmp_path / "source_selection"
    pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "method": "recent_180d",
                "validation_mae": 0.8,
            }
        ]
    ).to_csv(validation_scores_path, index=False)
    pd.DataFrame(
        [
            {
                "city": "denver",
                "source": "gfs_ens",
                "mae_raw": 1.0,
                "mae_corrected": 0.7,
                "interval_coverage_raw": 0.8,
                "interval_width_raw": 3.0,
            }
        ]
    ).to_csv(evaluation_path, index=False)

    code = main(
        [
            "--validation-scores",
            str(validation_scores_path),
            "--evaluation",
            str(evaluation_path),
            "--out-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    assert (output_dir / "selected_sources.csv").exists()
    assert "Wrote 1 selected sources" in capsys.readouterr().out

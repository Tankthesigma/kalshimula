import pandas as pd

from src.bias_policy_cli import main
from src.models.train_eval import train_eval_split


def test_bias_policy_cli_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "rows.csv"
    train_eval_dir = tmp_path / "train_eval"
    source_dir = tmp_path / "source_selection"
    out_dir = tmp_path / "model_policy"
    train_eval_dir.mkdir()
    source_dir.mkdir()
    rows = pd.DataFrame(
        [
            _row("2025-01-01", 100, 50),
            _row("2025-01-02", 100, 50),
            _row("2025-01-03", 70, 72),
            _row("2025-01-04", 70, 72),
            _row("2025-01-05", 70, 72),
            _row("2025-01-06", 70, 72),
            _row("2025-01-07", 70, 72),
            _row("2025-01-08", 70, 72),
        ]
    )
    rows.to_csv(input_path, index=False)
    train_eval = train_eval_split(
        rows,
        validation_start="2025-01-05",
        test_start="2025-01-07",
    )
    train_eval.evaluation.to_csv(train_eval_dir / "evaluation.csv", index=False)
    train_eval.selected_methods.to_csv(train_eval_dir / "selected_methods.csv", index=False)
    pd.DataFrame([{"city": "denver", "selected_source": "gfs_ens"}]).to_csv(
        source_dir / "recommended_sources.csv", index=False
    )

    code = main(
        [
            "--input",
            str(input_path),
            "--train-eval-dir",
            str(train_eval_dir),
            "--recommended-sources",
            str(source_dir / "recommended_sources.csv"),
            "--out-dir",
            str(out_dir),
            "--validation-start",
            "2025-01-05",
            "--test-start",
            "2025-01-07",
            "--recent-days",
            "2,4",
            "--alphas",
            "0.2",
        ]
    )

    assert code == 0
    assert "selected global_recent_" in capsys.readouterr().out
    assert (out_dir / "bias_policy_comparison.csv").exists()
    assert (out_dir / "model_policy.csv").exists()
    assert (out_dir / "bias_table.csv").exists()
    assert (out_dir / "interval_table.csv").exists()


def _row(target_date: str, point_f: float, actual_high_f: float) -> dict:
    return {
        "city": "denver",
        "target_date": target_date,
        "source": "gfs_ens",
        "point_f": point_f,
        "actual_high_f": actual_high_f,
        "absolute_error_f": abs(point_f - actual_high_f),
    }

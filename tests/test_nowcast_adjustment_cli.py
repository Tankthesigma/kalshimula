import json
from pathlib import Path

import pandas as pd

from src.nowcast_adjustment_cli import main
from tests.test_nowcast_adjustment import _features, _prediction_rows


def test_nowcast_adjustment_cli_writes_adjusted_predictions(tmp_path: Path, capsys) -> None:
    predictions = tmp_path / "predictions_nowcast.csv"
    features = tmp_path / "nowcast_features.csv"
    out_dir = tmp_path / "adjusted"
    _prediction_rows().to_csv(predictions, index=False)
    _features(70.6).to_csv(features, index=False)

    exit_code = main(
        [
            "--predictions-nowcast",
            str(predictions),
            "--nowcast-features",
            str(features),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "predictions_nowcast.csv").exists()
    assert (out_dir / "predictions_nowcast_manifest.json").exists()
    adjusted = pd.read_csv(out_dir / "predictions_nowcast.csv")
    assert adjusted["bin_lower_f"].tolist() == [71]
    assert json.loads(adjusted.iloc[0]["pmf_degree_json"]) == {"71": 1.0}
    assert "Wrote 1 adjusted nowcast rows" in capsys.readouterr().out

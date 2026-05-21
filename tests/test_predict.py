from datetime import date

import pandas as pd

from src import predict
from src.fetchers.openmeteo import ModelDailyHigh


def test_load_selected_source_matches_city_case_insensitively(tmp_path) -> None:
    selected_sources = tmp_path / "selected_sources.csv"
    selected_sources.write_text(
        "city,selected_source\nDenver,gfs_ens\nnyc,openmeteo_naive\n",
        encoding="utf-8",
    )

    assert predict._load_selected_source(selected_sources, "denver") == "gfs_ens"
    assert predict._load_selected_source(selected_sources, "NYC") == "openmeteo_naive"
    assert predict._load_selected_source(selected_sources, "chicago") is None


def test_members_for_selected_source_filters_individual_source() -> None:
    members = pd.DataFrame(
        {
            "source": ["gfs_ens", "gfs_ens", "ecmwf_ens"],
            "temp_f": [70.0, 72.0, 75.0],
        }
    )

    selected, applied = predict._members_for_selected_source(members, "gfs_ens")

    assert applied
    assert selected["source"].tolist() == ["gfs_ens", "gfs_ens"]


def test_members_for_selected_source_keeps_pool_for_openmeteo_naive() -> None:
    members = pd.DataFrame(
        {
            "source": ["gfs_ens", "ecmwf_ens"],
            "temp_f": [70.0, 75.0],
        }
    )

    selected, applied = predict._members_for_selected_source(
        members, "openmeteo_naive"
    )

    assert not applied
    assert selected.equals(members)


def test_predict_cli_uses_selected_source(monkeypatch, tmp_path, capsys) -> None:
    selected_sources = tmp_path / "selected_sources.csv"
    selected_sources.write_text(
        "city,selected_source\n"
        "denver,gfs_ens\n"
        "chicago,openmeteo_naive\n",
        encoding="utf-8",
    )

    def fake_fetch_all_parallel(station, target, *, use_historical):
        assert station.name == "Denver"
        assert target == date(2025, 1, 1)
        assert use_historical
        return [
            ModelDailyHigh(
                source="gfs_ens",
                target_date=target,
                members_f=[70.0, 72.0],
            ),
            ModelDailyHigh(
                source="ecmwf_ens",
                target_date=target,
                members_f=[80.0, 82.0],
            ),
        ]

    monkeypatch.setattr(predict, "_fetch_all_parallel", fake_fetch_all_parallel)

    exit_code = predict.main(
        [
            "--city",
            "denver",
            "--date",
            "2025-01-01",
            "--selected-sources",
            str(selected_sources),
        ]
    )
    output = capsys.readouterr()

    assert exit_code == 0
    assert "using selected source: gfs_ens" in output.err
    assert "Point estimate: 71.0" in output.out
    assert "Sources: gfs_ens(2)" in output.out
    assert "ecmwf_ens" not in output.out

from src import run_status_cli


def test_build_status_reports_progress_and_counts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "rows.csv").write_text(
        "city,target_date,source,point_f,actual_high_f,absolute_error_f\n"
        "denver,2025-01-01,gfs_ens,70,71,1\n"
        "denver,2025-01-01,openmeteo_naive,70,71,1\n"
        "chicago,2025-01-01,gfs_ens,30,31,1\n",
        encoding="utf-8",
    )
    (run_dir / "errors.csv").write_text(
        "city,target_date,error\n"
        "chicago,2025-01-02,boom\n",
        encoding="utf-8",
    )

    status = run_status_cli.build_status(
        run_dir=run_dir,
        cities=["denver", "chicago"],
        start=run_status_cli._parse_date("2025-01-01"),
        end=run_status_cli._parse_date("2025-01-02"),
        sources_per_day=2,
        openmeteo_mode="both",
    )

    assert "Rows: 3 / 8 theoretical (37.5%)" in status
    assert "City/date chunks: 1 / 4 (25.0%)" in status
    assert "Errors: 1" in status
    assert "Latest row: chicago, 2025-01-01, gfs_ens" in status
    assert "denver" in status
    assert "openmeteo_naive" in status


def test_run_status_cli_prints_status(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "rows.csv").write_text(
        "city,target_date,source,point_f,actual_high_f,absolute_error_f\n",
        encoding="utf-8",
    )

    code = run_status_cli.main(
        [
            "--run-dir",
            str(run_dir),
            "--cities",
            "denver,chicago",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-02",
            "--sources-per-day",
            "2",
            "--openmeteo-mode",
            "both",
        ]
    )

    assert code == 0
    assert "Rows: 0 / 8 theoretical (0.0%)" in capsys.readouterr().out


def test_chunk_progress_counts_naive_mode_completion(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "rows.csv").write_text(
        "city,target_date,source,point_f,actual_high_f,absolute_error_f\n"
        "denver,2025-01-01,openmeteo_naive,70,71,1\n"
        "denver,2025-01-02,gfs_ens,70,71,1\n",
        encoding="utf-8",
    )

    status = run_status_cli.build_status(
        run_dir=run_dir,
        cities=["denver"],
        start=run_status_cli._parse_date("2025-01-01"),
        end=run_status_cli._parse_date("2025-01-02"),
        sources_per_day=1,
        openmeteo_mode="naive",
    )

    assert "City/date chunks: 1 / 2 (50.0%)" in status

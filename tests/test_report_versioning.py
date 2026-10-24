"""Regression tests for preserving historical derived reports."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_tape_compare  # noqa: E402
import run_timing_luck  # noqa: E402


def test_tape_reproduction_writes_only_the_corrected_tracked_name(
    tmp_path, monkeypatch
):
    reports = tmp_path / "reports"
    reports.mkdir()
    historical = reports / "tape_comparison.csv"
    historical.write_text("historical\n", encoding="utf-8")
    monkeypatch.setattr(run_tape_compare, "REPORTS_DIR", reports)

    run_tape_compare._write_table(
        pd.DataFrame({"value": [1]}),
        tmp_path,
        "tape_comparison.csv",
    )

    assert historical.read_text(encoding="utf-8") == "historical\n"
    assert pd.read_csv(reports / "tape_comparison_corrected.csv").at[0, "value"] == 1


def test_positive_control_writes_only_the_corrected_tracked_name(
    tmp_path, monkeypatch
):
    reports = tmp_path / "reports"
    reports.mkdir()
    historical = reports / "vol_managed_control.csv"
    historical.write_text("historical\n", encoding="utf-8")
    monkeypatch.setattr(run_timing_luck, "REPORTS_DIR", reports)

    run_timing_luck._write_table(
        pd.DataFrame({"value": [1]}),
        tmp_path,
        "vol_managed_control.csv",
    )

    assert historical.read_text(encoding="utf-8") == "historical\n"
    assert pd.read_csv(reports / "vol_managed_control_corrected.csv").at[0, "value"] == 1


def test_timing_luck_module_does_not_import_matplotlib():
    """Test collection must not require matplotlib; plotting is a run-time import."""
    assert "plot_timing_luck_box" not in run_timing_luck.__dict__


def test_finalize_refuses_before_writing_when_inputs_are_missing(tmp_path, monkeypatch):
    import pytest

    import finalize_corrected_reports as finalize

    output = tmp_path / "output"
    output.mkdir()
    # every corrected input present except the CIZ metadata flag
    for name in finalize.MAPPINGS:
        (output / name).write_text("new\n", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    tracked = reports / "tape_comparison_corrected.csv"
    tracked.write_text("published\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["finalize", "--output-dir", str(output), "--report-dir", str(reports)],
    )

    with pytest.raises(SystemExit, match="--ciz-metadata"):
        finalize.main()

    assert tracked.read_text(encoding="utf-8") == "published\n"
    assert sorted(p.name for p in reports.iterdir()) == [tracked.name]


def test_finalize_never_copies_daily_return_files_into_reports():
    import finalize_corrected_reports as finalize

    # the full daily files carry SPY's CRSP returns and must stay in output/
    assert not [name for name in finalize.MAPPINGS if name.endswith((".csv.gz", ".json"))]

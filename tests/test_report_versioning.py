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

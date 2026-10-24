"""Synthetic tests for daily-return evidence verification.

The full evidence (with the benchmark, whose daily returns are vendor values)
lives in a local directory; only constituent-portfolio series are committed.
The verifier must check everything present and skip, not fail, what is local.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from report_evidence import (  # noqa: E402
    CONTROL_DAILY_NAME,
    CONTROL_MANIFEST_NAME,
    CONTROL_PUBLIC_MANIFEST_NAME,
    CONTROL_PUBLIC_NAME,
    TAPE_DAILY_NAME,
    TAPE_MANIFEST_NAME,
    TAPE_PUBLIC_MANIFEST_NAME,
    TAPE_PUBLIC_NAME,
    write_evidence,
)
from verify_corrected_reports import (  # noqa: E402
    compute_arithmetic_sharpe,
    compute_return_metrics,
    verify_corrected_reports,
)


def test_synthetic_return_arithmetic_matches_direct_formula():
    returns = pd.Series([0.0, 0.10, -0.05, 0.02])
    metrics = compute_return_metrics(returns, cash_rate_annual=0.01)

    expected_cagr = (1.10 * 0.95 * 1.02) ** (252.0 / 3.0) - 1.0
    expected_vol = float(returns.std(ddof=1) * np.sqrt(252.0))
    assert metrics["cagr"] == pytest.approx(expected_cagr)
    assert metrics["annualized_vol"] == pytest.approx(expected_vol)
    assert metrics["sharpe_geometric"] == pytest.approx((expected_cagr - 0.01) / expected_vol)
    assert metrics["max_drawdown"] == pytest.approx(0.05)

    cash = pd.Series([0.0001, 0.0001, 0.0001, 0.0001])
    excess = returns - cash
    assert compute_arithmetic_sharpe(returns, cash) == pytest.approx(
        excess.mean() / excess.std(ddof=1) * np.sqrt(252.0)
    )


def test_every_row_compounds_when_the_first_row_is_already_a_return():
    returns = pd.Series([0.10, -0.05, 0.02])
    metrics = compute_return_metrics(returns, cash_rate_annual=0.0, first_row_is_start=False)
    assert metrics["cagr"] == pytest.approx((1.10 * 0.95 * 1.02) ** (252.0 / 3.0) - 1.0)


def _write_synthetic_reports(tmp_path: Path) -> tuple[Path, Path]:
    report_dir = tmp_path / "reports"
    local_dir = tmp_path / "output" / "evidence"
    dates = pd.bdate_range("2020-01-02", periods=8)
    cash_rate = 0.02

    tape = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "legacy_strategy_return": [0.0, 0.01, -0.02, 0.015, 0.004, -0.003, 0.012, 0.001],
            "legacy_benchmark_return": [0.0, 0.009, -0.018, 0.013, 0.003, -0.002, 0.011, 0.002],
            "v2_strategy_return": [0.0, 0.0101, -0.0198, np.nan, 0.0041, -0.003, 0.0118, 0.0012],
            "v2_benchmark_return": [0.0, 0.0091, -0.0179, np.nan, 0.0031, -0.002, 0.0109, 0.0021],
        }
    )
    labels = {c: {"source_label": c} for c in tape.columns if c != "date"}
    fields = {
        "schema_version": 2,
        "date_column": "date",
        "metric_policy": {
            "periods_per_year": 252.0,
            "first_row_is_start": True,
            "geometric_sharpe_cash_rate_annual": {"legacy_dsf": cash_rate, "v2_ciz": cash_rate},
        },
    }
    write_evidence(tape, directory=local_dir, name=TAPE_DAILY_NAME, manifest_name=TAPE_MANIFEST_NAME,
                   return_columns=labels, manifest_fields=fields)
    write_evidence(tape, directory=report_dir, name=TAPE_PUBLIC_NAME,
                   manifest_name=TAPE_PUBLIC_MANIFEST_NAME, return_columns=labels,
                   manifest_fields=fields, columns=["legacy_strategy_return", "v2_strategy_return"])

    indexed = tape.set_index(pd.DatetimeIndex(dates))
    rows = []
    for name, strategy_col, bench_col in (
        ("legacy_dsf", "legacy_strategy_return", "legacy_benchmark_return"),
        ("v2_ciz", "v2_strategy_return", "v2_benchmark_return"),
    ):
        strategy = indexed[strategy_col].dropna()
        sm = compute_return_metrics(strategy, cash_rate_annual=cash_rate)
        bm = compute_return_metrics(indexed[bench_col].dropna(), cash_rate_annual=cash_rate)
        rows.append({
            "tape": name, "n_days": len(strategy),
            "first_date": strategy.index.min().date().isoformat(),
            "last_date": strategy.index.max().date().isoformat(),
            "strategy_cagr": sm["cagr"], "strategy_vol": sm["annualized_vol"],
            "strategy_sharpe": sm["sharpe_geometric"], "strategy_max_drawdown": sm["max_drawdown"],
            "index_cagr": bm["cagr"], "index_sharpe": bm["sharpe_geometric"],
            "strategy_return_column": strategy_col, "benchmark_return_column": bench_col,
        })
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_dir / "tape_comparison_corrected.csv", index=False)

    paired = tape[["legacy_strategy_return", "v2_strategy_return"]].dropna()
    gaps = (paired["v2_strategy_return"] - paired["legacy_strategy_return"]).abs() * 10_000.0
    pd.DataFrame([{
        "threshold_bps": 1.0, "strategy_days_compared": len(gaps),
        "strategy_days_over_threshold": int((gaps > 1.0).sum()),
        "strategy_mean_abs_gap_bps": gaps.mean(), "strategy_max_abs_gap_bps": gaps.max(),
        "legacy_strategy_return_column": "legacy_strategy_return",
        "v2_strategy_return_column": "v2_strategy_return",
    }]).to_csv(report_dir / "tape_return_differences_corrected.csv", index=False)

    control = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "cash_return": np.repeat((1.02 ** (1 / 365.25)) - 1, len(dates)),
            "index__buy_and_hold_return": tape["legacy_benchmark_return"],
            "basket__buy_and_hold_return": tape["legacy_strategy_return"] * 0.9 + 0.0002,
        }
    )
    clabels = {c: {"source_label": c} for c in control.columns if c != "date"}
    cfields = {
        "schema_version": 2,
        "date_column": "date",
        "cash_return_column": "cash_return",
        "metric_policy": {
            "periods_per_year": 252.0,
            "first_row_is_start": False,
            "geometric_sharpe_cash_rate_annual": cash_rate,
        },
    }
    write_evidence(control, directory=local_dir, name=CONTROL_DAILY_NAME,
                   manifest_name=CONTROL_MANIFEST_NAME, return_columns=clabels, manifest_fields=cfields)
    write_evidence(control, directory=report_dir, name=CONTROL_PUBLIC_NAME,
                   manifest_name=CONTROL_PUBLIC_MANIFEST_NAME, return_columns=clabels,
                   manifest_fields=cfields, columns=["cash_return", "basket__buy_and_hold_return"])
    cindexed = control.set_index(pd.DatetimeIndex(dates))
    crows = []
    for leg, column in (("index", "index__buy_and_hold_return"), ("basket", "basket__buy_and_hold_return")):
        values = cindexed[column]
        m = compute_return_metrics(values, cash_rate_annual=cash_rate, first_row_is_start=False)
        crows.append({
            "leg": leg, "variant": "buy_and_hold",
            "evaluation_start": dates.min().date().isoformat(),
            "evaluation_end": dates.max().date().isoformat(),
            "n_evaluation_days": len(values), "cagr": m["cagr"],
            "annualized_vol": m["annualized_vol"], "sharpe_geometric": m["sharpe_geometric"],
            "sharpe_arithmetic": compute_arithmetic_sharpe(values, cindexed["cash_return"]),
            "max_drawdown": m["max_drawdown"],
            "daily_return_column": column, "cash_return_column": "cash_return",
        })
    pd.DataFrame(crows).to_csv(report_dir / "vol_managed_control_corrected.csv", index=False)
    return report_dir, local_dir


def test_full_local_evidence_verifies_every_row(tmp_path):
    report_dir, local_dir = _write_synthetic_reports(tmp_path)
    checked = verify_corrected_reports(report_dir, evidence_dirs=[report_dir, local_dir])
    assert checked == {
        "tape_rows": 2, "tape_fields_skipped": 0, "strategy_gap_fields": 4,
        "control_rows": 2, "control_rows_skipped": 0,
    }


def test_committed_evidence_alone_skips_only_benchmark_series(tmp_path):
    report_dir, _ = _write_synthetic_reports(tmp_path)
    checked = verify_corrected_reports(report_dir, evidence_dirs=[report_dir])
    assert checked["tape_rows"] == 2
    assert checked["tape_fields_skipped"] == 4
    assert checked["control_rows"] == 1
    assert checked["control_rows_skipped"] == 1


def test_committed_copy_holds_no_benchmark_series(tmp_path):
    report_dir, _ = _write_synthetic_reports(tmp_path)
    for path in report_dir.glob("*.csv.gz"):
        columns = pd.read_csv(path, nrows=1).columns
        assert not any("benchmark" in c or c.startswith("index__") for c in columns)


def test_metric_tampering_fails(tmp_path):
    report_dir, local_dir = _write_synthetic_reports(tmp_path)
    path = report_dir / "vol_managed_control_corrected.csv"
    control = pd.read_csv(path)
    control.loc[1, "cagr"] += 0.01
    control.to_csv(path, index=False)
    with pytest.raises(ValueError, match="control.*cagr"):
        verify_corrected_reports(report_dir, evidence_dirs=[report_dir, local_dir])


def test_manifest_hash_detects_daily_return_tampering(tmp_path):
    report_dir, _ = _write_synthetic_reports(tmp_path)
    path = report_dir / TAPE_PUBLIC_NAME
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_corrected_reports(report_dir, evidence_dirs=[report_dir])


def test_committed_and_local_copies_must_agree(tmp_path):
    report_dir, local_dir = _write_synthetic_reports(tmp_path)
    tape = pd.read_csv(local_dir / TAPE_DAILY_NAME)
    tape.loc[3, "legacy_strategy_return"] += 0.001
    labels = {c: {"source_label": c} for c in tape.columns if c != "date"}
    write_evidence(
        tape, directory=local_dir, name=TAPE_DAILY_NAME, manifest_name=TAPE_MANIFEST_NAME,
        return_columns=labels,
        manifest_fields={"metric_policy": {"geometric_sharpe_cash_rate_annual": {"legacy_dsf": 0.02, "v2_ciz": 0.02}}},
    )
    with pytest.raises(ValueError, match="differs between files"):
        verify_corrected_reports(report_dir, evidence_dirs=[report_dir, local_dir])

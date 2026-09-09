"""Synthetic tests for committed aggregate-return verification."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from report_evidence import (  # noqa: E402
    CONTROL_DAILY_NAME,
    CONTROL_MANIFEST_NAME,
    TAPE_DAILY_NAME,
    TAPE_MANIFEST_NAME,
    committed_report_path,
    write_daily_evidence,
    write_manifest,
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
    assert metrics["sharpe_geometric"] == pytest.approx(
        (expected_cagr - 0.01) / expected_vol
    )
    assert metrics["max_drawdown"] == pytest.approx(0.05)

    cash = pd.Series([0.0001, 0.0001, 0.0001, 0.0001])
    excess = returns - cash
    assert compute_arithmetic_sharpe(returns, cash) == pytest.approx(
        excess.mean() / excess.std(ddof=1) * np.sqrt(252.0)
    )


def _write_synthetic_reports(tmp_path: Path) -> Path:
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "output"
    dates = pd.bdate_range("2020-01-02", periods=8)
    date_strings = dates.strftime("%Y-%m-%d")
    cash_rate = 0.02

    tape = pd.DataFrame(
        {
            "date": date_strings,
            "legacy_strategy_return": [
                0.0,
                0.01,
                -0.02,
                0.015,
                0.004,
                -0.003,
                0.012,
                0.001,
            ],
            "legacy_benchmark_return": [
                0.0,
                0.009,
                -0.018,
                0.013,
                0.003,
                -0.002,
                0.011,
                0.002,
            ],
            "v2_strategy_return": [
                0.0,
                0.0101,
                -0.0198,
                np.nan,
                0.0041,
                -0.003,
                0.0118,
                0.0012,
            ],
            "v2_benchmark_return": [
                0.0,
                0.0091,
                -0.0179,
                np.nan,
                0.0031,
                -0.002,
                0.0109,
                0.0021,
            ],
        }
    )
    artifact = write_daily_evidence(
        tape,
        output_dir=output_dir,
        report_dir=report_dir,
        name=TAPE_DAILY_NAME,
    )
    tape_columns = {
        column: {"source_label": column}
        for column in tape.columns
        if column != "date"
    }
    write_manifest(
        {
            **artifact,
            "schema_version": 1,
            "content": "synthetic portfolio returns",
            "date_column": "date",
            "return_columns": tape_columns,
            "metric_policy": {
                "periods_per_year": 252.0,
                "geometric_sharpe_cash_rate_annual": {
                    "legacy_dsf": cash_rate,
                    "v2_ciz": cash_rate,
                },
                "strategy_gap_threshold_bps": 1.0,
            },
        },
        output_dir=output_dir,
        report_dir=report_dir,
        name=TAPE_MANIFEST_NAME,
    )

    tape_rows = []
    for tape_name, strategy_column, benchmark_column in (
        (
            "legacy_dsf",
            "legacy_strategy_return",
            "legacy_benchmark_return",
        ),
        ("v2_ciz", "v2_strategy_return", "v2_benchmark_return"),
    ):
        strategy = tape.set_index(pd.DatetimeIndex(dates))[strategy_column].dropna()
        benchmark = tape.set_index(pd.DatetimeIndex(dates))[
            benchmark_column
        ].dropna()
        strategy_metrics = compute_return_metrics(
            strategy, cash_rate_annual=cash_rate
        )
        benchmark_metrics = compute_return_metrics(
            benchmark, cash_rate_annual=cash_rate
        )
        tape_rows.append(
            {
                "tape": tape_name,
                "n_days": len(strategy),
                "first_date": strategy.index.min().date().isoformat(),
                "last_date": strategy.index.max().date().isoformat(),
                "strategy_cagr": strategy_metrics["cagr"],
                "strategy_vol": strategy_metrics["annualized_vol"],
                "strategy_sharpe": strategy_metrics["sharpe_geometric"],
                "strategy_max_drawdown": strategy_metrics["max_drawdown"],
                "index_cagr": benchmark_metrics["cagr"],
                "index_sharpe": benchmark_metrics["sharpe_geometric"],
                "strategy_daily_returns_path": committed_report_path(
                    TAPE_DAILY_NAME
                ),
                "strategy_return_column": strategy_column,
                "benchmark_daily_returns_path": committed_report_path(
                    TAPE_DAILY_NAME
                ),
                "benchmark_return_column": benchmark_column,
                "daily_returns_manifest_path": committed_report_path(
                    TAPE_MANIFEST_NAME
                ),
            }
        )
    pd.DataFrame(tape_rows).to_csv(
        report_dir / "tape_comparison_corrected.csv", index=False
    )

    paired = tape[
        ["legacy_strategy_return", "v2_strategy_return"]
    ].dropna()
    gaps = (
        paired["v2_strategy_return"] - paired["legacy_strategy_return"]
    ).abs() * 10_000.0
    pd.DataFrame(
        [
            {
                "threshold_bps": 1.0,
                "strategy_days_compared": len(gaps),
                "strategy_days_over_threshold": int((gaps > 1.0).sum()),
                "strategy_mean_abs_gap_bps": gaps.mean(),
                "strategy_max_abs_gap_bps": gaps.max(),
                "strategy_daily_returns_path": committed_report_path(
                    TAPE_DAILY_NAME
                ),
                "legacy_strategy_return_column": "legacy_strategy_return",
                "v2_strategy_return_column": "v2_strategy_return",
                "daily_returns_manifest_path": committed_report_path(
                    TAPE_MANIFEST_NAME
                ),
            }
        ]
    ).to_csv(report_dir / "tape_return_differences_corrected.csv", index=False)

    control = pd.DataFrame(
        {
            "date": date_strings,
            "cash_return": np.repeat((1.02 ** (1 / 365.25)) - 1, len(dates)),
            "index__buy_and_hold_return": tape["legacy_benchmark_return"],
            "index__vol_managed_daily_net_return": (
                tape["legacy_benchmark_return"] * 0.8 - 0.0001
            ),
        }
    )
    control_artifact = write_daily_evidence(
        control,
        output_dir=output_dir,
        report_dir=report_dir,
        name=CONTROL_DAILY_NAME,
    )
    control_columns = {
        column: {"source_label": column}
        for column in control.columns
        if column != "date"
    }
    write_manifest(
        {
            **control_artifact,
            "schema_version": 1,
            "content": "synthetic portfolio returns",
            "date_column": "date",
            "cash_return_column": "cash_return",
            "return_columns": control_columns,
            "metric_policy": {
                "periods_per_year": 252.0,
                "geometric_sharpe_cash_rate_annual": cash_rate,
                "arithmetic_sharpe_cash_return_column": "cash_return",
            },
        },
        output_dir=output_dir,
        report_dir=report_dir,
        name=CONTROL_MANIFEST_NAME,
    )

    control_rows = []
    for variant, column in (
        ("buy_and_hold", "index__buy_and_hold_return"),
        (
            "vol_managed_daily_net",
            "index__vol_managed_daily_net_return",
        ),
    ):
        values = control.set_index(pd.DatetimeIndex(dates))[column]
        metrics = compute_return_metrics(values, cash_rate_annual=cash_rate)
        control_rows.append(
            {
                "leg": "index",
                "variant": variant,
                "evaluation_start": dates.min().date().isoformat(),
                "evaluation_end": dates.max().date().isoformat(),
                "n_evaluation_days": len(values),
                "cagr": metrics["cagr"],
                "annualized_vol": metrics["annualized_vol"],
                "sharpe_geometric": metrics["sharpe_geometric"],
                "sharpe_arithmetic": compute_arithmetic_sharpe(
                    values,
                    control.set_index(pd.DatetimeIndex(dates))["cash_return"],
                ),
                "max_drawdown": metrics["max_drawdown"],
                "daily_returns_path": committed_report_path(CONTROL_DAILY_NAME),
                "daily_return_column": column,
                "cash_return_column": "cash_return",
                "daily_returns_manifest_path": committed_report_path(
                    CONTROL_MANIFEST_NAME
                ),
            }
        )
    pd.DataFrame(control_rows).to_csv(
        report_dir / "vol_managed_control_corrected.csv", index=False
    )
    return report_dir


def test_synthetic_corrected_reports_verify_and_metric_tampering_fails(tmp_path):
    report_dir = _write_synthetic_reports(tmp_path)
    assert verify_corrected_reports(report_dir) == {
        "tape_rows": 2,
        "control_rows": 2,
        "strategy_gap_fields": 4,
    }

    path = report_dir / "vol_managed_control_corrected.csv"
    control = pd.read_csv(path)
    control.loc[0, "cagr"] += 0.01
    control.to_csv(path, index=False)
    with pytest.raises(ValueError, match="control.*cagr"):
        verify_corrected_reports(report_dir)


def test_synthetic_manifest_hash_detects_daily_return_tampering(tmp_path):
    report_dir = _write_synthetic_reports(tmp_path)
    manifest = json.loads(
        (report_dir / TAPE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["artifact_sha256"]

    path = report_dir / TAPE_DAILY_NAME
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_corrected_reports(report_dir)

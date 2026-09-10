#!/usr/bin/env python3
"""Recompute corrected headline arithmetic from committed daily aggregates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PERIODS_PER_YEAR = 252.0


def compute_return_metrics(
    returns: pd.Series,
    *,
    cash_rate_annual: float,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> dict[str, float]:
    """Independently recompute the headline metrics from simple returns."""
    rets = pd.to_numeric(returns, errors="raise").dropna().astype(float)
    if len(rets) < 2:
        raise ValueError("Need at least two daily returns.")
    if not np.isfinite(rets.to_numpy()).all():
        raise ValueError("Daily returns must be finite.")
    if (rets <= -1.0).any():
        raise ValueError("Daily returns must stay above -100%.")

    equity = (1.0 + rets).cumprod()
    total_periods = len(equity) - 1
    cagr = (
        (float(equity.iloc[-1]) / float(equity.iloc[0]))
        ** (float(periods_per_year) / total_periods)
        - 1.0
    )
    annualized_vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe_geometric = (
        (cagr - float(cash_rate_annual)) / annualized_vol
        if annualized_vol > 0.0
        else np.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    return {
        "cagr": float(cagr),
        "annualized_vol": annualized_vol,
        "sharpe_geometric": float(sharpe_geometric),
        "max_drawdown": abs(float(drawdown.min())),
    }


def compute_arithmetic_sharpe(
    returns: pd.Series,
    cash_returns: pd.Series,
    *,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> float:
    """Recompute root-N arithmetic Sharpe from aligned daily cash returns."""
    aligned = pd.concat(
        [
            pd.to_numeric(returns, errors="raise").rename("portfolio"),
            pd.to_numeric(cash_returns, errors="raise").rename("cash"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    excess = aligned["portfolio"] - aligned["cash"]
    std = float(excess.std(ddof=1))
    if std <= 0.0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def _resolve_report_path(report_dir: Path, reference: str) -> Path:
    relative = Path(str(reference))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid committed report path: {reference}")
    if relative.parts and relative.parts[0] == "reports":
        relative = Path(*relative.parts[1:])
    candidate = (report_dir / relative).resolve()
    root = report_dir.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Report path escapes report directory: {reference}")
    return candidate


def _load_evidence(
    report_dir: Path,
    artifact_reference: str,
    manifest_reference: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    artifact_path = _resolve_report_path(report_dir, artifact_reference)
    manifest_path = _resolve_report_path(report_dir, manifest_reference)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_path") != artifact_reference:
        raise ValueError(
            f"{manifest_path.name} points to {manifest.get('artifact_path')!r}, "
            f"not {artifact_reference!r}."
        )
    payload = artifact_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest.get("artifact_sha256"):
        raise ValueError(f"SHA-256 mismatch for {artifact_reference}.")

    daily = pd.read_csv(artifact_path)
    expected_columns = {"date", *manifest.get("return_columns", {}).keys()}
    if set(daily.columns) != expected_columns:
        raise ValueError(
            f"{artifact_reference} columns are {list(daily.columns)!r}; "
            f"manifest permits only {sorted(expected_columns)!r}."
        )
    dates = pd.to_datetime(daily["date"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{artifact_reference} dates must be unique and sorted.")
    daily.index = pd.DatetimeIndex(dates)
    daily = daily.drop(columns="date")

    if len(daily) != int(manifest["rows"]):
        raise ValueError(f"Row-count mismatch for {artifact_reference}.")
    if daily.index.min().date().isoformat() != manifest["first_date"]:
        raise ValueError(f"First-date mismatch for {artifact_reference}.")
    if daily.index.max().date().isoformat() != manifest["last_date"]:
        raise ValueError(f"Last-date mismatch for {artifact_reference}.")
    return daily, manifest


def _check_close(
    failures: list[str],
    label: str,
    computed: float,
    reported: Any,
    *,
    rtol: float,
    atol: float,
) -> None:
    expected = float(reported)
    if np.isnan(computed) and np.isnan(expected):
        return
    if not np.isclose(computed, expected, rtol=rtol, atol=atol):
        failures.append(
            f"{label}: computed {computed:.17g}, reported {expected:.17g}"
        )


def _verify_tape_reports(
    report_dir: Path,
    *,
    rtol: float,
    atol: float,
) -> tuple[int, int]:
    summary = pd.read_csv(report_dir / "tape_comparison_corrected.csv")
    required = {
        "tape",
        "n_days",
        "first_date",
        "last_date",
        "strategy_cagr",
        "strategy_vol",
        "strategy_sharpe",
        "strategy_max_drawdown",
        "index_cagr",
        "index_sharpe",
        "strategy_daily_returns_path",
        "strategy_return_column",
        "benchmark_daily_returns_path",
        "benchmark_return_column",
        "daily_returns_manifest_path",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Tape summary is missing evidence columns: {sorted(missing)}")

    failures: list[str] = []
    daily: pd.DataFrame | None = None
    manifest: dict[str, Any] | None = None
    for row in summary.to_dict(orient="records"):
        if row["strategy_daily_returns_path"] != row["benchmark_daily_returns_path"]:
            raise ValueError(f"{row['tape']} strategy and benchmark paths differ.")
        row_daily, row_manifest = _load_evidence(
            report_dir,
            row["strategy_daily_returns_path"],
            row["daily_returns_manifest_path"],
        )
        if daily is None:
            daily, manifest = row_daily, row_manifest
        strategy = row_daily[row["strategy_return_column"]].dropna()
        benchmark = row_daily[row["benchmark_return_column"]].dropna()
        cash_rates = row_manifest["metric_policy"][
            "geometric_sharpe_cash_rate_annual"
        ]
        metrics = compute_return_metrics(
            strategy,
            cash_rate_annual=float(cash_rates[row["tape"]]),
        )
        benchmark_metrics = compute_return_metrics(
            benchmark,
            cash_rate_annual=float(cash_rates[row["tape"]]),
        )
        prefix = f"tape {row['tape']}"
        for metric, column in (
            ("cagr", "strategy_cagr"),
            ("annualized_vol", "strategy_vol"),
            ("sharpe_geometric", "strategy_sharpe"),
            ("max_drawdown", "strategy_max_drawdown"),
        ):
            _check_close(
                failures,
                f"{prefix} {column}",
                metrics[metric],
                row[column],
                rtol=rtol,
                atol=atol,
            )
        for metric, column in (
            ("cagr", "index_cagr"),
            ("sharpe_geometric", "index_sharpe"),
        ):
            _check_close(
                failures,
                f"{prefix} {column}",
                benchmark_metrics[metric],
                row[column],
                rtol=rtol,
                atol=atol,
            )
        if len(strategy) != int(row["n_days"]):
            failures.append(
                f"{prefix} n_days: computed {len(strategy)}, "
                f"reported {int(row['n_days'])}"
            )
        if strategy.index.min().date().isoformat() != str(row["first_date"]):
            failures.append(f"{prefix} first_date does not match daily evidence")
        if strategy.index.max().date().isoformat() != str(row["last_date"]):
            failures.append(f"{prefix} last_date does not match daily evidence")

    assert daily is not None and manifest is not None
    differences = pd.read_csv(
        report_dir / "tape_return_differences_corrected.csv"
    )
    if len(differences) != 1:
        raise ValueError("Tape return-difference report must contain one row.")
    gap_row = differences.iloc[0]
    gap_required = {
        "threshold_bps",
        "strategy_days_compared",
        "strategy_days_over_threshold",
        "strategy_mean_abs_gap_bps",
        "strategy_max_abs_gap_bps",
        "strategy_daily_returns_path",
        "legacy_strategy_return_column",
        "v2_strategy_return_column",
        "daily_returns_manifest_path",
    }
    missing = gap_required - set(differences.columns)
    if missing:
        raise ValueError(f"Tape gap report is missing evidence columns: {sorted(missing)}")
    gap_daily, _ = _load_evidence(
        report_dir,
        str(gap_row["strategy_daily_returns_path"]),
        str(gap_row["daily_returns_manifest_path"]),
    )
    paired = gap_daily[
        [
            str(gap_row["legacy_strategy_return_column"]),
            str(gap_row["v2_strategy_return_column"]),
        ]
    ].dropna()
    gaps = (paired.iloc[:, 1] - paired.iloc[:, 0]).abs() * 10_000.0
    threshold = float(gap_row["threshold_bps"])
    exact_checks = {
        "strategy_days_compared": len(gaps),
        "strategy_days_over_threshold": int((gaps > threshold).sum()),
    }
    for column, computed in exact_checks.items():
        if int(gap_row[column]) != computed:
            failures.append(
                f"tape gap {column}: computed {computed}, "
                f"reported {int(gap_row[column])}"
            )
    for column, computed in (
        ("strategy_mean_abs_gap_bps", float(gaps.mean())),
        ("strategy_max_abs_gap_bps", float(gaps.max())),
    ):
        _check_close(
            failures,
            f"tape gap {column}",
            computed,
            gap_row[column],
            rtol=rtol,
            atol=atol,
        )
    if failures:
        raise ValueError("Corrected tape verification failed:\n- " + "\n- ".join(failures))
    return len(summary), len(exact_checks) + 2


def _verify_control_report(
    report_dir: Path,
    *,
    rtol: float,
    atol: float,
) -> int:
    control = pd.read_csv(report_dir / "vol_managed_control_corrected.csv")
    required = {
        "leg",
        "variant",
        "evaluation_start",
        "evaluation_end",
        "n_evaluation_days",
        "cagr",
        "annualized_vol",
        "sharpe_geometric",
        "sharpe_arithmetic",
        "max_drawdown",
        "daily_returns_path",
        "daily_return_column",
        "cash_return_column",
        "daily_returns_manifest_path",
    }
    missing = required - set(control.columns)
    if missing:
        raise ValueError(
            f"Volatility-control report is missing evidence columns: {sorted(missing)}"
        )

    failures: list[str] = []
    referenced_columns: set[str] = set()
    for row in control.to_dict(orient="records"):
        daily, manifest = _load_evidence(
            report_dir,
            row["daily_returns_path"],
            row["daily_returns_manifest_path"],
        )
        return_column = row["daily_return_column"]
        cash_column = row["cash_return_column"]
        referenced_columns.add(return_column)
        returns = daily[return_column].dropna()
        cash = daily[cash_column].reindex(returns.index)
        policy = manifest["metric_policy"]
        metrics = compute_return_metrics(
            returns,
            cash_rate_annual=float(
                policy["geometric_sharpe_cash_rate_annual"]
            ),
            periods_per_year=float(policy["periods_per_year"]),
        )
        metrics["sharpe_arithmetic"] = compute_arithmetic_sharpe(
            returns,
            cash,
            periods_per_year=float(policy["periods_per_year"]),
        )
        prefix = f"control {row['leg']} {row['variant']}"
        for metric, column in (
            ("cagr", "cagr"),
            ("annualized_vol", "annualized_vol"),
            ("sharpe_geometric", "sharpe_geometric"),
            ("sharpe_arithmetic", "sharpe_arithmetic"),
            ("max_drawdown", "max_drawdown"),
        ):
            _check_close(
                failures,
                f"{prefix} {column}",
                metrics[metric],
                row[column],
                rtol=rtol,
                atol=atol,
            )
        if len(returns) != int(row["n_evaluation_days"]):
            failures.append(
                f"{prefix} n_evaluation_days: computed {len(returns)}, "
                f"reported {int(row['n_evaluation_days'])}"
            )
        if returns.index.min().date().isoformat() != str(row["evaluation_start"]):
            failures.append(f"{prefix} evaluation_start does not match evidence")
        if returns.index.max().date().isoformat() != str(row["evaluation_end"]):
            failures.append(f"{prefix} evaluation_end does not match evidence")

    expected_return_columns = {
        column
        for column in manifest["return_columns"]
        if column != manifest["cash_return_column"]
    }
    if referenced_columns != expected_return_columns:
        raise ValueError(
            "Control report does not reference each committed portfolio-return "
            "column exactly once."
        )
    if failures:
        raise ValueError(
            "Corrected volatility-control verification failed:\n- "
            + "\n- ".join(failures)
        )
    return len(control)


def verify_corrected_reports(
    report_dir: Path = Path("reports"),
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict[str, int]:
    """Verify all corrected performance headlines against committed evidence."""
    tape_rows, gap_fields = _verify_tape_reports(
        report_dir, rtol=rtol, atol=atol
    )
    control_rows = _verify_control_report(report_dir, rtol=rtol, atol=atol)
    return {
        "tape_rows": tape_rows,
        "control_rows": control_rows,
        "strategy_gap_fields": gap_fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    checked = verify_corrected_reports(
        args.report_dir,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(
        "verified corrected report arithmetic: "
        f"{checked['tape_rows']} tape rows, "
        f"{checked['control_rows']} control rows, "
        f"{checked['strategy_gap_fields']} strategy-gap fields"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

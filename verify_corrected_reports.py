#!/usr/bin/env python3
"""Recompute corrected headline arithmetic from portfolio-level daily returns.

Evidence comes from two places. ``reports/`` holds committed series built from
the constituent portfolio; ``output/evidence/`` holds the full local copy that
also carries SPY's CRSP returns and is never committed. Every metric whose
series is present is recomputed; benchmark and index-leg metrics are skipped,
and counted as skipped, when only the committed copy exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from report_evidence import LOCAL_EVIDENCE_DIR


PERIODS_PER_YEAR = 252.0


def compute_return_metrics(
    returns: pd.Series,
    *,
    cash_rate_annual: float,
    periods_per_year: float = PERIODS_PER_YEAR,
    first_row_is_start: bool = True,
) -> dict[str, float]:
    """Independently recompute the headline metrics from simple returns.

    ``first_row_is_start`` says whether the first row is the starting point of
    the equity curve (a backtest's day 0, return zero) or already a return.
    In the second case every return compounds and the curve starts at 1.0.
    """
    rets = pd.to_numeric(returns, errors="raise").dropna().astype(float)
    if len(rets) < 2:
        raise ValueError("Need at least two daily returns.")
    if not np.isfinite(rets.to_numpy()).all():
        raise ValueError("Daily returns must be finite.")
    if (rets <= -1.0).any():
        raise ValueError("Daily returns must stay above -100%.")

    growth = (1.0 + rets).cumprod()
    if first_row_is_start:
        equity = growth
        total_periods = len(equity) - 1
    else:
        equity = pd.concat([pd.Series([1.0]), growth.reset_index(drop=True)], ignore_index=True)
        total_periods = len(rets)
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


@dataclass
class EvidenceColumn:
    """One daily-return series and the policy of the file it came from."""

    values: pd.Series
    policy: dict[str, Any]
    cash_column: str | None
    source: Path


def _load_manifest(manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = str(manifest.get("artifact_file", ""))
    if not name or Path(name).name != name:
        raise ValueError(f"{manifest_path.name} must name an artifact file in its own directory.")
    artifact_path = manifest_path.parent / name
    payload = artifact_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("artifact_sha256"):
        raise ValueError(f"SHA-256 mismatch for {artifact_path}.")

    daily = pd.read_csv(artifact_path)
    expected_columns = {"date", *manifest.get("return_columns", {}).keys()}
    if set(daily.columns) != expected_columns:
        raise ValueError(
            f"{artifact_path.name} columns are {list(daily.columns)!r}; "
            f"manifest permits only {sorted(expected_columns)!r}."
        )
    dates = pd.to_datetime(daily["date"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{artifact_path.name} dates must be unique and sorted.")
    daily.index = pd.DatetimeIndex(dates)
    daily = daily.drop(columns="date")
    if len(daily) != int(manifest["rows"]):
        raise ValueError(f"Row-count mismatch for {artifact_path.name}.")
    if daily.index.min().date().isoformat() != manifest["first_date"]:
        raise ValueError(f"First-date mismatch for {artifact_path.name}.")
    if daily.index.max().date().isoformat() != manifest["last_date"]:
        raise ValueError(f"Last-date mismatch for {artifact_path.name}.")
    return daily, manifest


def load_evidence(evidence_dirs: list[Path]) -> dict[str, EvidenceColumn]:
    """Every daily-return column available across the evidence directories.

    A column present in more than one file must carry identical values, so the
    committed subset cannot drift from the local full copy.
    """
    columns: dict[str, EvidenceColumn] = {}
    for directory in evidence_dirs:
        if not directory.is_dir():
            continue
        for manifest_path in sorted(directory.glob("*.manifest.json")):
            daily, manifest = _load_manifest(manifest_path)
            for column in daily.columns:
                entry = EvidenceColumn(
                    values=daily[column],
                    policy=manifest.get("metric_policy", {}),
                    cash_column=manifest.get("cash_return_column"),
                    source=manifest_path,
                )
                if column in columns:
                    previous = columns[column].values
                    same = previous.index.equals(entry.values.index) and np.allclose(
                        previous.to_numpy(dtype=float),
                        entry.values.to_numpy(dtype=float),
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    )
                    if not same:
                        raise ValueError(f"Evidence column {column} differs between files.")
                    continue
                columns[column] = entry
    return columns


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


def _metrics_for(column: EvidenceColumn, cash_rate: float) -> dict[str, float]:
    return compute_return_metrics(
        column.values.dropna(),
        cash_rate_annual=cash_rate,
        periods_per_year=float(column.policy.get("periods_per_year", PERIODS_PER_YEAR)),
        first_row_is_start=bool(column.policy.get("first_row_is_start", True)),
    )


def _verify_tape_reports(
    report_dir: Path,
    columns: dict[str, EvidenceColumn],
    *,
    rtol: float,
    atol: float,
) -> dict[str, int]:
    summary = pd.read_csv(report_dir / "tape_comparison_corrected.csv")
    required = {
        "tape", "n_days", "first_date", "last_date", "strategy_cagr", "strategy_vol",
        "strategy_sharpe", "strategy_max_drawdown", "index_cagr", "index_sharpe",
        "strategy_return_column", "benchmark_return_column",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Tape summary is missing evidence columns: {sorted(missing)}")

    failures: list[str] = []
    skipped = 0
    for row in summary.to_dict(orient="records"):
        prefix = f"tape {row['tape']}"
        strategy_col = columns.get(row["strategy_return_column"])
        if strategy_col is None:
            raise ValueError(f"{prefix}: strategy series {row['strategy_return_column']} is not in any evidence file.")
        cash_rate = float(strategy_col.policy["geometric_sharpe_cash_rate_annual"][row["tape"]])
        strategy = strategy_col.values.dropna()
        metrics = _metrics_for(strategy_col, cash_rate)
        for metric, column in (
            ("cagr", "strategy_cagr"),
            ("annualized_vol", "strategy_vol"),
            ("sharpe_geometric", "strategy_sharpe"),
            ("max_drawdown", "strategy_max_drawdown"),
        ):
            _check_close(failures, f"{prefix} {column}", metrics[metric], row[column], rtol=rtol, atol=atol)
        if len(strategy) != int(row["n_days"]):
            failures.append(f"{prefix} n_days: computed {len(strategy)}, reported {int(row['n_days'])}")
        if strategy.index.min().date().isoformat() != str(row["first_date"]):
            failures.append(f"{prefix} first_date does not match daily evidence")
        if strategy.index.max().date().isoformat() != str(row["last_date"]):
            failures.append(f"{prefix} last_date does not match daily evidence")

        benchmark_col = columns.get(row["benchmark_return_column"])
        if benchmark_col is None:
            # SPY's daily returns are CRSP values and only exist locally
            skipped += 2
            continue
        bench = _metrics_for(benchmark_col, cash_rate)
        for metric, column in (("cagr", "index_cagr"), ("sharpe_geometric", "index_sharpe")):
            _check_close(failures, f"{prefix} {column}", bench[metric], row[column], rtol=rtol, atol=atol)

    differences = pd.read_csv(report_dir / "tape_return_differences_corrected.csv")
    if len(differences) != 1:
        raise ValueError("Tape return-difference report must contain one row.")
    gap_row = differences.iloc[0]
    for name in ("legacy_strategy_return_column", "v2_strategy_return_column"):
        if str(gap_row[name]) not in columns:
            raise ValueError(f"Tape gap series {gap_row[name]} is not in any evidence file.")
    paired = pd.concat(
        [
            columns[str(gap_row["legacy_strategy_return_column"])].values,
            columns[str(gap_row["v2_strategy_return_column"])].values,
        ],
        axis=1,
    ).dropna()
    gaps = (paired.iloc[:, 1] - paired.iloc[:, 0]).abs() * 10_000.0
    threshold = float(gap_row["threshold_bps"])
    for column, computed in (
        ("strategy_days_compared", len(gaps)),
        ("strategy_days_over_threshold", int((gaps > threshold).sum())),
    ):
        if int(gap_row[column]) != computed:
            failures.append(f"tape gap {column}: computed {computed}, reported {int(gap_row[column])}")
    for column, computed in (
        ("strategy_mean_abs_gap_bps", float(gaps.mean())),
        ("strategy_max_abs_gap_bps", float(gaps.max())),
    ):
        _check_close(failures, f"tape gap {column}", computed, gap_row[column], rtol=rtol, atol=atol)
    if failures:
        raise ValueError("Corrected tape verification failed:\n- " + "\n- ".join(failures))
    return {"tape_rows": len(summary), "tape_fields_skipped": skipped, "strategy_gap_fields": 4}


def _verify_control_report(
    report_dir: Path,
    columns: dict[str, EvidenceColumn],
    *,
    rtol: float,
    atol: float,
) -> dict[str, int]:
    control = pd.read_csv(report_dir / "vol_managed_control_corrected.csv")
    required = {
        "leg", "variant", "evaluation_start", "evaluation_end", "n_evaluation_days", "cagr",
        "annualized_vol", "sharpe_geometric", "sharpe_arithmetic", "max_drawdown",
        "daily_return_column", "cash_return_column",
    }
    missing = required - set(control.columns)
    if missing:
        raise ValueError(f"Volatility-control report is missing evidence columns: {sorted(missing)}")

    failures: list[str] = []
    verified = skipped = 0
    referenced: set[str] = set()
    for row in control.to_dict(orient="records"):
        prefix = f"control {row['leg']} {row['variant']}"
        series = columns.get(row["daily_return_column"])
        referenced.add(row["daily_return_column"])
        if series is None:
            if row["leg"] == "index":
                # overlays on SPY's CRSP returns only exist in the local copy
                skipped += 1
                continue
            raise ValueError(f"{prefix}: series {row['daily_return_column']} is not in any evidence file.")
        cash = columns.get(row["cash_return_column"])
        if cash is None:
            raise ValueError(f"{prefix}: cash series {row['cash_return_column']} is missing.")
        returns = series.values.dropna()
        metrics = _metrics_for(series, float(series.policy["geometric_sharpe_cash_rate_annual"]))
        metrics["sharpe_arithmetic"] = compute_arithmetic_sharpe(
            returns,
            cash.values.reindex(returns.index),
            periods_per_year=float(series.policy.get("periods_per_year", PERIODS_PER_YEAR)),
        )
        for metric in ("cagr", "annualized_vol", "sharpe_geometric", "sharpe_arithmetic", "max_drawdown"):
            _check_close(failures, f"{prefix} {metric}", metrics[metric], row[metric], rtol=rtol, atol=atol)
        if len(returns) != int(row["n_evaluation_days"]):
            failures.append(
                f"{prefix} n_evaluation_days: computed {len(returns)}, reported {int(row['n_evaluation_days'])}"
            )
        if returns.index.min().date().isoformat() != str(row["evaluation_start"]):
            failures.append(f"{prefix} evaluation_start does not match evidence")
        if returns.index.max().date().isoformat() != str(row["evaluation_end"]):
            failures.append(f"{prefix} evaluation_end does not match evidence")
        verified += 1

    # every portfolio series in a control evidence file must back some row
    for name, entry in columns.items():
        if entry.cash_column is None or name == entry.cash_column:
            continue
        if name not in referenced:
            failures.append(f"evidence column {name} is not referenced by the control report")
    if failures:
        raise ValueError("Corrected volatility-control verification failed:\n- " + "\n- ".join(failures))
    return {"control_rows": verified, "control_rows_skipped": skipped}


def verify_corrected_reports(
    report_dir: Path = Path("reports"),
    *,
    evidence_dirs: list[Path] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict[str, int]:
    """Verify all corrected performance headlines against available evidence."""
    dirs = evidence_dirs if evidence_dirs is not None else [report_dir, LOCAL_EVIDENCE_DIR]
    columns = load_evidence(dirs)
    return {
        **_verify_tape_reports(report_dir, columns, rtol=rtol, atol=atol),
        **_verify_control_report(report_dir, columns, rtol=rtol, atol=atol),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        action="append",
        help="evidence directory to read (repeatable); default reports/ and output/evidence/",
    )
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    checked = verify_corrected_reports(
        args.report_dir,
        evidence_dirs=args.evidence_dir,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(
        "verified corrected report arithmetic: "
        f"{checked['tape_rows']} tape rows, "
        f"{checked['strategy_gap_fields']} strategy-gap fields, "
        f"{checked['control_rows']} control rows; skipped "
        f"{checked['tape_fields_skipped']} benchmark fields and "
        f"{checked['control_rows_skipped']} index-leg rows that need the local "
        f"evidence in {LOCAL_EVIDENCE_DIR}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

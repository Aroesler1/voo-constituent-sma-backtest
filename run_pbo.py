#!/usr/bin/env python3
"""Probability of Backtest Overfitting for the SMA-length sweep.

The report already carries the Deflated Sharpe Ratio, which asks whether the
SELECTED configuration's Sharpe survives the fact that several were tried. PBO
asks a sharper question about the selection itself: pick the best configuration
in sample, and how often does it land in the bottom half out of sample? Bailey,
Borwein, Lopez de Prado and Zhu, "The Probability of Backtest Overfitting"
(Journal of Computational Finance 20(4), 2017).

Input is the per-configuration daily return panel written by `main.py` as
`output/sma_sweep_returns.csv`: one column per SMA length, one row per trading
day, all five evaluated on the same dates.

Usage:
    python run_pbo.py
    python run_pbo.py --returns output/sma_sweep_returns.csv --n-splits 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from statistics_mt import (
    deflated_sharpe,
    per_period_sharpe,
    probability_of_backtest_overfitting,
)

TRADING_DAYS = 252


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--returns", type=Path,
                    default=Path("output/sma_sweep_returns.csv"),
                    help="per-configuration daily return panel from main.py")
    ap.add_argument("--n-splits", type=int, default=16,
                    help="CSCV blocks; must be even (16 -> C(16,8) = 12,870 splits)")
    ap.add_argument("--out", type=Path, default=Path("output/pbo_sweep.csv"))
    args = ap.parse_args(argv)

    if not args.returns.exists():
        print(f"{args.returns} not found. Run main.py first; it writes the "
              f"per-configuration sweep returns alongside the other outputs.")
        return 1

    panel = pd.read_csv(args.returns, index_col=0, parse_dates=True)
    panel = panel.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if panel.shape[1] < 2:
        print(f"need at least 2 configurations, found {panel.shape[1]}")
        return 1

    result = probability_of_backtest_overfitting(panel, n_splits=args.n_splits)

    sharpes = {c: per_period_sharpe(panel[c]) for c in panel.columns}
    trial_sharpes = list(sharpes.values())
    best_cfg = max(sharpes, key=sharpes.get)
    dsr = {
        c: deflated_sharpe(panel[c], n_trials=len(panel.columns),
                           trial_sharpes=trial_sharpes)
        for c in panel.columns
    }

    print(f"sweep: {panel.shape[1]} configurations, {len(panel):,} daily observations "
          f"({panel.index.min().date()} -> {panel.index.max().date()})")
    print(f"CSCV:  {result['n_splits']} blocks, {result['n_combinations']:,} "
          f"symmetric splits, {result['obs_used']:,} observations used\n")

    table = pd.DataFrame({
        "sharpe_annualised": {c: sharpes[c] * np.sqrt(TRADING_DAYS) for c in panel.columns},
        "deflated_sharpe": dsr,
    })
    print(table.to_string(float_format=lambda v: f"{v:0.4f}"))

    print(f"\nbest in-sample configuration (full sample): {best_cfg}")
    print(f"median out-of-sample rank of the in-sample winner: "
          f"{result['median_oos_rank']:.1f} of {result['n_configs']}")
    print(f"PBO = {result['pbo']:.3f}")

    # With five configurations the out-of-sample rank takes five values, so the
    # logit is quantised and PBO is coarse. Say so rather than reporting three
    # decimals as if they were meaningful.
    print(f"\nPBO is the fraction of the {result['n_combinations']:,} splits in which the "
          f"in-sample best\nconfiguration ranked in the bottom half out of sample. With "
          f"{result['n_configs']} configurations\nthe rank takes only {result['n_configs']} "
          f"values, so this is an indicator, not a precise probability.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "pbo": [result["pbo"]],
        "n_combinations": [result["n_combinations"]],
        "n_configs": [result["n_configs"]],
        "n_splits": [result["n_splits"]],
        "obs_used": [result["obs_used"]],
        "median_oos_rank": [result["median_oos_rank"]],
        "best_config_full_sample": [best_cfg],
    }).to_csv(args.out, index=False)
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

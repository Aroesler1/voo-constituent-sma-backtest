"""Legacy CRSP tape against the post-2025 CIZ tape, on the headline run.

CRSP's January 2025 release of Flat File Format 1.0 (SIZ) was the last of the
legacy tape; only Format 2.0 (CIZ) is updated now. Schwarz, Walter & Weiss
(*JFQA*, 24 February 2026; SSRN 5074864) measure the effect on monthly returns
and note that the daily returns "did not change materially", which is what makes
a daily-frequency strategy a clean test rather than a redundant one.

This script rebuilds the panel from ``crsp.dsf_v2``, reruns the headline
configuration on both tapes, counts the constituent-days whose return differs by
more than a basis point, and writes the comparison to ``output/``.

The v2 pull is large. It caches per ticker under ``data_cache/crsp_v2/`` and is
resumable: rerunning after an interruption fetches only what is missing.

    python run_tape_compare.py

Requires a WRDS entitlement covering ``crsp.dsf_v2``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_engine import run_buy_and_hold, run_constituent_backtest
from config import BacktestConfig, load_config
from crsp_v2 import (
    compare_return_panels,
    fetch_benchmark_series_v2,
    fetch_constituent_prices_v2,
    largest_return_differences,
)
from metrics import compute_metrics
from panel import BacktestPanel, build_panel
from preprocessing import build_rebalance_calendar

LOGGER = logging.getLogger(__name__)

#: Tracked copy of the summary tables; see run_timing_luck.REPORTS_DIR. Only
#: portfolio-level aggregates and difference counts go here. The per-ticker-day
#: table of largest disagreements deliberately does not: it carries raw CRSP
#: return values and stays in the gitignored output/ directory.
REPORTS_DIR = Path("reports")


def _write_table(frame: pd.DataFrame, out_dir: Path, name: str) -> None:
    """Write one derived table to output/ and to the tracked reports/ copy."""
    frame.to_csv(out_dir / name, index=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORTS_DIR / name, index=False)


def _setup_logging(output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    fh = logging.FileHandler(out / "tape_compare.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def run_headline(panel: BacktestPanel, config: BacktestConfig, tape: str) -> dict[str, Any]:
    """Run the README's headline configuration on a panel.

    The headline is the pre-specified 200-day SMA on the semi-monthly reporting
    schedule, which is what ``main.py`` reports; keeping it identical is the
    point of the comparison.
    """
    calendar = build_rebalance_calendar(panel.trading_index, config.REBALANCE_DEFAULT)
    result = run_constituent_backtest(
        price_df=panel.close_df,
        return_df=panel.close_returns,
        membership_mask=panel.membership,
        active_mask=panel.active_mask,
        rebalance_calendar=calendar,
        config=config,
        cash_curve=panel.cash_curve,
        store_weights=False,
        store_cost_attribution=False,
    )
    met = compute_metrics(
        result["equity_curve"],
        result["weekly_returns"],
        result["trade_log"],
        result["positions"],
        panel.effective_cash_rate,
        active_count=result.get("active_count"),
        eligible_count=result.get("eligible_count"),
        exposure=result.get("exposure"),
    )

    bench = run_buy_and_hold(panel.spy_close.reindex(panel.trading_index).ffill(), config, panel.cash_curve)
    bench_met = compute_metrics(
        bench["equity_curve"],
        bench["weekly_returns"],
        bench["trade_log"],
        bench["positions"],
        panel.effective_cash_rate,
    )

    return {
        "tape": tape,
        "n_tickers": len(panel.valid_cols),
        "n_days": len(panel.trading_index),
        "first_date": str(pd.Timestamp(panel.trading_index.min()).date()),
        "last_date": str(pd.Timestamp(panel.trading_index.max()).date()),
        "strategy_cagr": met["cagr"],
        "strategy_vol": met["annualized_vol"],
        "strategy_sharpe": met["sharpe"],
        "strategy_max_drawdown": met["max_drawdown"],
        "strategy_annual_turnover": met["annual_turnover"],
        "strategy_total_trades": met["total_trades"],
        "strategy_cost_bps_annualized": met["total_cost_bps_annualized"],
        "index_cagr": bench_met["cagr"],
        "index_sharpe": bench_met["sharpe"],
        "returns": result["period_returns"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold-bps",
        type=float,
        default=1.0,
        help="Return difference above which a constituent-day counts as altered.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config()
    _setup_logging(config.OUTPUT_DIR)
    out_dir = Path(config.OUTPUT_DIR)

    LOGGER.info("Building legacy (crsp.dsf) panel.")
    legacy = build_panel(config)
    legacy_head = run_headline(legacy, config, "legacy_dsf")
    LOGGER.info("Legacy headline CAGR=%.4f Sharpe=%.4f", legacy_head["strategy_cagr"], legacy_head["strategy_sharpe"])

    LOGGER.info("Building v2 (crsp.dsf_v2) panel; the first run pulls the whole universe.")
    v2 = build_panel(
        config,
        constituent_fetcher=fetch_constituent_prices_v2,
        benchmark_fetcher=fetch_benchmark_series_v2,
    )
    v2_head = run_headline(v2, config, "v2_dsf_v2")
    LOGGER.info("v2 headline CAGR=%.4f Sharpe=%.4f", v2_head["strategy_cagr"], v2_head["strategy_sharpe"])

    # build_panel fills uncovered constituent-days with a zero return rather
    # than NaN, so validity has to come from the price matrix. Without this the
    # denominator counts millions of ticker-days on which neither tape holds a
    # security and the two trivially agree.
    legacy_valid = legacy.close_df.notna()
    v2_valid = v2.close_df.notna()

    diff = compare_return_panels(
        legacy.close_returns,
        v2.close_returns,
        legacy_valid=legacy_valid,
        v2_valid=v2_valid,
        threshold_bps=float(args.threshold_bps),
    )
    LOGGER.info("Constituent-day return differences: %s", diff)

    # Split off each security's final covered day. The legacy tape compounds its
    # delisting return into that row and CIZ's DlyRet does not, so those rows
    # measure a schema gap in this loader rather than a difference in the tape's
    # numbers. Reporting both makes the size of each visible.
    final_day = legacy_valid.apply(lambda col: col[::-1].idxmax() if col.any() else pd.NaT)
    is_final = pd.DataFrame(False, index=legacy_valid.index, columns=legacy_valid.columns)
    for ticker, day in final_day.items():
        if pd.notna(day):
            is_final.loc[day, ticker] = True

    diff_ex_final = compare_return_panels(
        legacy.close_returns,
        v2.close_returns,
        legacy_valid=legacy_valid & ~is_final,
        v2_valid=v2_valid & ~is_final,
        threshold_bps=float(args.threshold_bps),
    )
    LOGGER.info("Excluding each security's final day: %s", diff_ex_final)
    _write_table(pd.DataFrame([diff_ex_final]), out_dir, "tape_return_differences_ex_final_day.csv")

    worst = largest_return_differences(
        legacy.close_returns, v2.close_returns,
        legacy_valid=legacy_valid, v2_valid=v2_valid, top_n=25,
    )
    worst.to_csv(out_dir / "tape_largest_differences.csv", index=False)
    LOGGER.info("Largest tape disagreements:\n%s", worst.head(10).to_string(index=False))

    # Strategy-level agreement: the two daily return series on identical dates.
    common = legacy_head["returns"].index.intersection(v2_head["returns"].index)
    strat_gap_bps = (v2_head["returns"].loc[common] - legacy_head["returns"].loc[common]).abs() * 10_000.0
    diff["strategy_days_compared"] = int(len(common))
    diff["strategy_days_over_threshold"] = int((strat_gap_bps > float(args.threshold_bps)).sum())
    diff["strategy_mean_abs_gap_bps"] = float(strat_gap_bps.mean())
    diff["strategy_max_abs_gap_bps"] = float(strat_gap_bps.max()) if len(common) else np.nan

    summary = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "returns"} for row in (legacy_head, v2_head)]
    )
    _write_table(summary, out_dir, "tape_comparison.csv")
    _write_table(pd.DataFrame([diff]), out_dir, "tape_return_differences.csv")

    LOGGER.info("Headline on both tapes:\n%s", summary.to_string(index=False))
    LOGGER.info("Done in %.1f s.", time.perf_counter() - started)


if __name__ == "__main__":
    main()

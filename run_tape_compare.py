"""Legacy CRSP tape against the post-2025 CIZ tape, on the headline run.

CRSP's January 2025 release of Flat File Format 1.0 (SIZ) was the last of the
legacy tape; only Format 2.0 (CIZ) is updated now. Schwarz, Walter & Weiss
(*JFQA*, 24 February 2026; SSRN 5074864) measure the effect on monthly returns
and note that the daily returns "did not change materially", which is what makes
a daily-frequency strategy a clean test rather than a redundant one.

This script rebuilds both panels through the pinned 2024-12-31 endpoint, first
writes identifier-matched coverage and ordinary/terminal-session differences,
then reruns the same headline configuration on both tapes.

The v2 pull is large. New extracts use versioned caches. The frozen unversioned
cache remains a read-only fallback.

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
    load_verified_terminal_outcomes,
)
from metrics import compute_metrics
from panel import BacktestPanel, build_panel
from preprocessing import build_rebalance_calendar
from report_evidence import (
    TAPE_DAILY_NAME,
    TAPE_MANIFEST_NAME,
    committed_report_path,
    write_daily_evidence,
    write_manifest,
)

LOGGER = logging.getLogger(__name__)

#: Tracked copy of the summary tables; see run_timing_luck.REPORTS_DIR. Only
#: portfolio-level aggregates and difference counts go here. The per-ticker-day
#: table of largest disagreements deliberately does not: it carries raw CRSP
#: return values and stays in the gitignored output/ directory.
REPORTS_DIR = Path("reports")
CORRECTED_REPORT_NAMES = {
    "tape_comparison.csv": "tape_comparison_corrected.csv",
    "tape_coverage.csv": "tape_coverage_corrected.csv",
    "tape_return_differences.csv": "tape_return_differences_corrected.csv",
    "tape_return_differences_by_session.csv": (
        "tape_return_differences_by_session_corrected.csv"
    ),
}


def _write_table(frame: pd.DataFrame, out_dir: Path, name: str) -> None:
    """Write one derived table to output/ and to the tracked reports/ copy."""
    frame.to_csv(out_dir / name, index=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORTS_DIR / CORRECTED_REPORT_NAMES.get(name, name), index=False)


def _terminal_mask_from_outcomes(
    permnos: pd.DataFrame,
    outcomes: pd.DataFrame,
    date_column: str,
) -> pd.DataFrame:
    """Map vendor terminal dates to ticker cells by permanent identifier."""
    mask = pd.DataFrame(False, index=permnos.index, columns=permnos.columns)
    if date_column not in outcomes.columns:
        return mask
    events = outcomes.dropna(subset=[date_column, "permno"]).copy()
    events[date_column] = pd.to_datetime(events[date_column])
    events = events[events[date_column].isin(mask.index)]
    for date, group in events.groupby(date_column):
        mask.loc[date] = permnos.loc[date].isin(group["permno"]).to_numpy()
    return mask


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
        "benchmark_returns": bench["period_returns"],
        "cash_rate_annual": panel.effective_cash_rate,
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
    if config.END_DATE != "2024-12-31":
        raise ValueError("Tape comparison requires the pinned END_DATE=2024-12-31.")
    if not config.CIZ_TERMINAL_RETURNS_PATH or not config.CIZ_TERMINAL_METADATA_PATH:
        raise ValueError(
            "Tape comparison requires a verified CIZ terminal-return extract and sidecar."
        )
    terminal_outcomes, terminal_metadata = load_verified_terminal_outcomes(
        config.CIZ_TERMINAL_RETURNS_PATH,
        config.CIZ_TERMINAL_METADATA_PATH,
    )
    LOGGER.info(
        "Verified CIZ terminal source: product=%s table=%s semantics=%s",
        terminal_metadata["source_product"],
        terminal_metadata["source_table"],
        terminal_metadata["return_semantics"],
    )

    LOGGER.info("Building legacy (crsp.dsf) panel.")
    legacy = build_panel(config)

    LOGGER.info("Building v2 panel from versioned or frozen CIZ caches.")
    v2 = build_panel(
        config,
        constituent_fetcher=fetch_constituent_prices_v2,
        benchmark_fetcher=fetch_benchmark_series_v2,
    )

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
        legacy_permno=legacy.permno_df,
        v2_permno=v2.permno_df,
        threshold_bps=float(args.threshold_bps),
    )
    legacy_terminal = (
        legacy.terminal_mask.reindex_like(legacy_valid).fillna(False)
        | _terminal_mask_from_outcomes(
            legacy.permno_df,
            terminal_outcomes,
            "legacy_event_date",
        )
    )
    v2_terminal = (
        v2.terminal_mask.reindex_like(v2_valid).fillna(False)
        | _terminal_mask_from_outcomes(
            v2.permno_df,
            terminal_outcomes,
            "event_date",
        )
    )
    terminal = legacy_terminal | v2_terminal
    segmented_diffs = []
    for segment, mask in (
        ("ordinary_session", ~terminal),
        ("terminal_session", terminal),
    ):
        row = compare_return_panels(
            legacy.close_returns,
            v2.close_returns,
            legacy_valid=legacy_valid & mask,
            v2_valid=v2_valid & mask,
            legacy_permno=legacy.permno_df,
            v2_permno=v2.permno_df,
            threshold_bps=float(args.threshold_bps),
        )
        row["segment"] = segment
        segmented_diffs.append(row)

    coverage = pd.DataFrame(
        [
            {
                "tape": name,
                "n_tickers": len(panel.valid_cols),
                "n_covered_constituent_days": int(valid.to_numpy().sum()),
                "n_terminal_sessions": int(terminal_mask.to_numpy().sum()),
                "n_terminal_returns_applied": int(
                    panel.terminal_applied_mask.to_numpy().sum()
                ),
                "first_date": panel.trading_index.min().date().isoformat(),
                "last_date": panel.trading_index.max().date().isoformat(),
            }
            for name, panel, valid, terminal_mask in (
                ("legacy_dsf", legacy, legacy_valid, legacy_terminal),
                ("v2_ciz", v2, v2_valid, v2_terminal),
            )
        ]
    )
    _write_table(coverage, out_dir, "tape_coverage.csv")
    _write_table(
        pd.DataFrame(segmented_diffs),
        out_dir,
        "tape_return_differences_by_session.csv",
    )
    _write_table(pd.DataFrame([diff]), out_dir, "tape_return_differences.csv")
    LOGGER.info("Coverage:\n%s", coverage.to_string(index=False))
    LOGGER.info(
        "Ordinary and terminal-session differences:\n%s",
        pd.DataFrame(segmented_diffs).to_string(index=False),
    )

    worst = largest_return_differences(
        legacy.close_returns, v2.close_returns,
        legacy_valid=legacy_valid,
        v2_valid=v2_valid,
        legacy_permno=legacy.permno_df,
        v2_permno=v2.permno_df,
        top_n=25,
    )
    worst.to_csv(out_dir / "tape_largest_differences.csv", index=False)
    LOGGER.info("Largest tape disagreements:\n%s", worst.head(10).to_string(index=False))

    # Coverage and return-difference evidence is now durable. Only then run and
    # report strategy metrics with identical membership, costs, and endpoint.
    legacy_head = run_headline(legacy, config, "legacy_dsf")
    v2_head = run_headline(v2, config, "v2_ciz")
    LOGGER.info(
        "Legacy headline CAGR=%.4f Sharpe=%.4f",
        legacy_head["strategy_cagr"],
        legacy_head["strategy_sharpe"],
    )
    LOGGER.info(
        "v2 headline CAGR=%.4f Sharpe=%.4f",
        v2_head["strategy_cagr"],
        v2_head["strategy_sharpe"],
    )

    # Strategy-level agreement: the two daily return series on identical dates.
    common = legacy_head["returns"].index.intersection(v2_head["returns"].index)
    strat_gap_bps = (v2_head["returns"].loc[common] - legacy_head["returns"].loc[common]).abs() * 10_000.0
    diff["strategy_days_compared"] = int(len(common))
    diff["strategy_days_over_threshold"] = int(
        (strat_gap_bps > float(args.threshold_bps)).sum()
    )
    diff["strategy_mean_abs_gap_bps"] = float(strat_gap_bps.mean())
    diff["strategy_max_abs_gap_bps"] = (
        float(strat_gap_bps.max()) if len(common) else np.nan
    )

    daily = pd.concat(
        {
            "legacy_strategy_return": legacy_head["returns"],
            "legacy_benchmark_return": legacy_head["benchmark_returns"],
            "v2_strategy_return": v2_head["returns"],
            "v2_benchmark_return": v2_head["benchmark_returns"],
        },
        axis=1,
    )
    daily.index.name = "date"
    daily = daily.reset_index()
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    artifact = write_daily_evidence(
        daily,
        output_dir=out_dir,
        report_dir=REPORTS_DIR,
        name=TAPE_DAILY_NAME,
    )
    source_labels = {
        "legacy_strategy_return": {
            "source_label": "legacy_dsf:constituent_sma200:semi_monthly",
            "tape": "legacy_dsf",
            "portfolio": "constituent_sma200_semi_monthly",
        },
        "legacy_benchmark_return": {
            "source_label": "legacy_dsf:sp500_total_return:buy_and_hold",
            "tape": "legacy_dsf",
            "portfolio": "sp500_total_return_buy_and_hold",
        },
        "v2_strategy_return": {
            "source_label": "v2_ciz:constituent_sma200:semi_monthly",
            "tape": "v2_ciz",
            "portfolio": "constituent_sma200_semi_monthly",
        },
        "v2_benchmark_return": {
            "source_label": "v2_ciz:sp500_total_return:buy_and_hold",
            "tape": "v2_ciz",
            "portfolio": "sp500_total_return_buy_and_hold",
        },
    }
    write_manifest(
        {
            **artifact,
            "schema_version": 1,
            "content": "portfolio-level daily simple returns only",
            "date_column": "date",
            "return_columns": source_labels,
            "metric_policy": {
                "periods_per_year": 252.0,
                "geometric_sharpe_cash_rate_annual": {
                    "legacy_dsf": legacy_head["cash_rate_annual"],
                    "v2_ciz": v2_head["cash_rate_annual"],
                },
                "strategy_gap_threshold_bps": float(args.threshold_bps),
            },
        },
        output_dir=out_dir,
        report_dir=REPORTS_DIR,
        name=TAPE_MANIFEST_NAME,
    )

    daily_path = committed_report_path(TAPE_DAILY_NAME)
    manifest_path = committed_report_path(TAPE_MANIFEST_NAME)
    diff["strategy_daily_returns_path"] = daily_path
    diff["legacy_strategy_return_column"] = "legacy_strategy_return"
    diff["v2_strategy_return_column"] = "v2_strategy_return"
    diff["daily_returns_manifest_path"] = manifest_path
    # The constituent-only evidence was persisted before either strategy ran.
    # Refresh this aggregate report now that strategy-level evidence is known.
    _write_table(pd.DataFrame([diff]), out_dir, "tape_return_differences.csv")

    summary = pd.DataFrame(
        [
            {
                k: v
                for k, v in row.items()
                if k not in {"returns", "benchmark_returns", "cash_rate_annual"}
            }
            for row in (legacy_head, v2_head)
        ]
    )
    summary["strategy_daily_returns_path"] = daily_path
    summary["strategy_return_column"] = summary["tape"].map(
        {
            "legacy_dsf": "legacy_strategy_return",
            "v2_ciz": "v2_strategy_return",
        }
    )
    summary["benchmark_daily_returns_path"] = daily_path
    summary["benchmark_return_column"] = summary["tape"].map(
        {
            "legacy_dsf": "legacy_benchmark_return",
            "v2_ciz": "v2_benchmark_return",
        }
    )
    summary["daily_returns_manifest_path"] = manifest_path
    _write_table(summary, out_dir, "tape_comparison.csv")

    LOGGER.info("Headline on both tapes:\n%s", summary.to_string(index=False))
    LOGGER.info("Done in %.1f s.", time.perf_counter() - started)


if __name__ == "__main__":
    main()

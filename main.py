"""Main orchestrator for institutional constituent-level VOO SMA backtests."""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_engine import run_buy_and_hold, run_constituent_backtest
from config import config_hash, get_periods, load_config
from data_loader import build_snapshot_manifest_row
from metrics import compute_drawdown_series, compute_metrics, compute_rolling_sharpe
from panel import build_panel, snapshot_row
from preprocessing import build_rebalance_calendar, estimate_proxy_fidelity
from reporting import (
    plot_active_breadth,
    plot_cost_diagnostics,
    plot_drawdowns,
    plot_equity_curves,
    plot_regime_comparison,
    plot_rolling_sharpe,
    plot_schedule_comparison,
    plot_schedule_risk_return,
    plot_sma_sweep,
    print_summary_table,
    write_detailed_report,
)
from statistics_mt import (
    deflated_sharpe,
    per_period_sharpe,
    probability_of_backtest_overfitting,
    romano_wolf_stepdown,
)
from strategy import compute_sma_matrix, generate_active_mask

LOGGER = logging.getLogger(__name__)


def setup_logging(output_dir: str) -> None:
    """Configure console + file logging."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "backtest.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(file_handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _package_versions() -> dict[str, str]:
    packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "requests",
        "python-dotenv",
        "statsmodels",
        "pyarrow",
        "fredapi",
    ]
    out: dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not-installed"
    return out



def _slice_result(result: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    eq = result["equity_curve"].loc[(result["equity_curve"].index >= start) & (result["equity_curve"].index <= end)]
    rets = result["weekly_returns"].loc[(result["weekly_returns"].index >= start) & (result["weekly_returns"].index <= end)]
    pos = result["positions"].loc[(result["positions"].index >= start) & (result["positions"].index <= end)]

    tr = result.get("trade_log", pd.DataFrame()).copy()
    if not tr.empty and "date" in tr.columns:
        tr["date"] = pd.to_datetime(tr["date"])
        tr = tr[(tr["date"] >= start) & (tr["date"] <= end)]

    act = result.get("active_count")
    if isinstance(act, pd.Series):
        act = act.loc[(act.index >= start) & (act.index <= end)]

    elig = result.get("eligible_count")
    if isinstance(elig, pd.Series):
        elig = elig.loc[(elig.index >= start) & (elig.index <= end)]

    exp = result.get("exposure")
    if isinstance(exp, pd.Series):
        exp = exp.loc[(exp.index >= start) & (exp.index <= end)]

    return {
        "equity_curve": eq,
        "weekly_returns": rets,
        "positions": pos,
        "trade_log": tr,
        "active_count": act,
        "eligible_count": elig,
        "exposure": exp,
    }


def _period_metrics(
    results_by_label: dict[str, dict[str, Any]],
    periods: dict[str, tuple[str, str]],
    cash_rate: float,
) -> dict[str, dict[str, Any]]:
    """Compute period decomposition payload for reporting."""
    out: dict[str, dict[str, Any]] = {}
    ref_index = next(iter(results_by_label.values()))["equity_curve"].index

    for period_name, (start_str, end_str) in periods.items():
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        overlap_start = max(start, ref_index.min())
        overlap_end = min(end, ref_index.max())

        entry: dict[str, Any] = {
            "start": None,
            "end": None,
            "metrics": {},
            "valid": False,
        }

        if overlap_start > overlap_end:
            out[period_name] = entry
            continue
        entry["start"] = overlap_start.date().isoformat()
        entry["end"] = overlap_end.date().isoformat()
        entry["valid"] = True

        for label, result in results_by_label.items():
            sliced = _slice_result(result, overlap_start, overlap_end)
            if len(sliced["equity_curve"]) < 2:
                entry["metrics"][label] = {"cagr": np.nan, "max_drawdown": np.nan}
                continue
            m = compute_metrics(
                sliced["equity_curve"],
                sliced["weekly_returns"],
                sliced["trade_log"],
                sliced["positions"],
                cash_rate,
                active_count=sliced.get("active_count"),
                eligible_count=sliced.get("eligible_count"),
                exposure=sliced.get("exposure"),
            )
            entry["metrics"][label] = {"cagr": m["cagr"], "max_drawdown": m["max_drawdown"]}

        out[period_name] = entry

    return out


def _compute_passive_proxy_returns(
    close_returns: pd.DataFrame,
    membership: pd.DataFrame,
    base_weights: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Compute passive proxy basket returns and coverage diagnostics."""
    idx = close_returns.index
    cols = close_returns.columns

    ret = close_returns.reindex(index=idx, columns=cols)
    mem = membership.reindex(index=idx, columns=cols).fillna(False)
    base = base_weights.reindex(index=idx, columns=cols).fillna(0.0)

    proxy_ret = pd.Series(0.0, index=idx, dtype=float)
    coverage = pd.Series(0.0, index=idx, dtype=float)

    for i in range(1, len(idx)):
        prev = idx[i - 1]
        dt = idx[i]

        w = base.loc[prev].where(mem.loc[prev], 0.0).astype(float)
        w = w.clip(lower=0.0)

        if w.sum() <= 0 and mem.loc[prev].sum() > 0:
            count = int(mem.loc[prev].sum())
            w = mem.loc[prev].astype(float) / float(count)

        total_target = float(w.sum())
        if total_target <= 0:
            proxy_ret.loc[dt] = 0.0
            coverage.loc[dt] = 0.0
            continue

        available = ret.loc[dt].notna()
        covered_weight = float(w.where(available, 0.0).sum())
        coverage.loc[dt] = covered_weight / total_target

        w_exec = w.where(available, 0.0)
        if w_exec.sum() > 0:
            w_exec = w_exec / float(w_exec.sum())

        r = ret.loc[dt].fillna(0.0)
        proxy_ret.loc[dt] = float((w_exec * r).sum())

    return {"returns": proxy_ret, "coverage": coverage}


def _proxy_regime_deltas(
    proxy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods: dict[str, tuple[str, str]],
    cash_rate: float,
) -> pd.DataFrame:
    """Compute regime-level deltas between proxy basket and ETF benchmark."""
    rows: list[dict[str, Any]] = []

    proxy_eq = (1.0 + proxy_returns.fillna(0.0)).cumprod()
    bench_eq = (1.0 + benchmark_returns.fillna(0.0)).cumprod()

    for name, (start_str, end_str) in periods.items():
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)

        p_eq = proxy_eq.loc[(proxy_eq.index >= start) & (proxy_eq.index <= end)]
        b_eq = bench_eq.loc[(bench_eq.index >= start) & (bench_eq.index <= end)]

        if len(p_eq) < 2 or len(b_eq) < 2:
            rows.append(
                {
                    "period": name,
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "proxy_cagr": np.nan,
                    "benchmark_cagr": np.nan,
                    "cagr_diff_bps": np.nan,
                    "proxy_sharpe": np.nan,
                    "benchmark_sharpe": np.nan,
                    "sharpe_diff": np.nan,
                    "proxy_max_drawdown": np.nan,
                    "benchmark_max_drawdown": np.nan,
                    "max_drawdown_diff": np.nan,
                }
            )
            continue

        p_ret = proxy_returns.reindex(p_eq.index).fillna(0.0)
        b_ret = benchmark_returns.reindex(b_eq.index).fillna(0.0)

        p_metrics = compute_metrics(
            p_eq,
            p_ret,
            pd.DataFrame(),
            pd.Series(1, index=p_eq.index, dtype="Int64"),
            cash_rate,
        )
        b_metrics = compute_metrics(
            b_eq,
            b_ret,
            pd.DataFrame(),
            pd.Series(1, index=b_eq.index, dtype="Int64"),
            cash_rate,
        )

        rows.append(
            {
                "period": name,
                "start": max(start, p_eq.index.min()).date().isoformat(),
                "end": min(end, p_eq.index.max()).date().isoformat(),
                "proxy_cagr": p_metrics["cagr"],
                "benchmark_cagr": b_metrics["cagr"],
                "cagr_diff_bps": (p_metrics["cagr"] - b_metrics["cagr"]) * 10000.0,
                "proxy_sharpe": p_metrics["sharpe"],
                "benchmark_sharpe": b_metrics["sharpe"],
                "sharpe_diff": p_metrics["sharpe"] - b_metrics["sharpe"],
                "proxy_max_drawdown": p_metrics["max_drawdown"],
                "benchmark_max_drawdown": b_metrics["max_drawdown"],
                "max_drawdown_diff": p_metrics["max_drawdown"] - b_metrics["max_drawdown"],
            }
        )

    return pd.DataFrame(rows)



def main() -> None:
    """Run full constituent-level institutional backtest workflow."""
    started = time.perf_counter()

    config = load_config()
    setup_logging(config.OUTPUT_DIR)

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(config.CACHE_DIR).mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting institutional constituent-level VOO SMA backtest.")
    panel = build_panel(config)

    sp500_events = panel.sp500_events
    sec_holdings = panel.sec_holdings
    voo_daily, voo_source = panel.voo_daily, panel.voo_source
    spy_daily, spy_source = panel.spy_daily, panel.spy_source
    spy_close = panel.spy_close
    trading_index = panel.trading_index
    extended_voo_proxy = panel.extended_voo_proxy
    membership = panel.membership
    constituentsnapshot_rows = panel.constituentsnapshot_rows
    close_df = panel.close_df
    close_returns = panel.close_returns
    base_weights = panel.base_weights
    source_by_date = panel.source_by_date
    coverage_ratio = panel.coverage_ratio
    provider_mix = panel.provider_mix
    active_mask = panel.active_mask
    cash_curve = panel.cash_curve
    effective_cash_rate = panel.effective_cash_rate

    # 8) Run schedule sweep (default = retail baseline)
    schedule_metrics: dict[str, dict[str, float]] = {}
    schedule_rows: list[dict[str, Any]] = []
    strategy_res: dict[str, Any] | None = None

    for freq in config.REBALANCE_SWEEP_VALUES:
        LOGGER.info("Running rebalance schedule: %s", freq)
        cal = build_rebalance_calendar(trading_index, freq)
        res = run_constituent_backtest(
            price_df=close_df,
            return_df=close_returns,
            membership_mask=membership,
            active_mask=active_mask,
            rebalance_calendar=cal,
            config=config,
            cash_curve=cash_curve,
            store_weights=freq == config.REBALANCE_DEFAULT,
            store_cost_attribution=freq == config.REBALANCE_DEFAULT,
        )
        m = compute_metrics(
            res["equity_curve"],
            res["weekly_returns"],
            res["trade_log"],
            res["positions"],
            effective_cash_rate,
            active_count=res.get("active_count"),
            eligible_count=res.get("eligible_count"),
            exposure=res.get("exposure"),
        )

        schedule_metrics[freq] = m
        schedule_rows.append(
            {
                "frequency": freq,
                "cagr": m["cagr"],
                "annualized_vol": m["annualized_vol"],
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
                "annual_turnover": m["annual_turnover"],
                "avg_trade_cost_bps": m["avg_trade_cost_bps"],
                "avg_active_names": m["avg_active_names"],
                "active_breadth_pct": m["active_breadth_pct"],
                "is_default": bool(freq == config.REBALANCE_DEFAULT),
            }
        )
        if freq == config.REBALANCE_DEFAULT:
            strategy_res = res
        else:
            del res
            gc.collect()

    freq_order = ["daily", "weekly", "semi_monthly", "monthly"]
    schedule_df = pd.DataFrame(schedule_rows)
    schedule_df["frequency"] = pd.Categorical(schedule_df["frequency"], categories=freq_order, ordered=True)
    schedule_df = schedule_df.sort_values("frequency").reset_index(drop=True)

    if strategy_res is None:
        raise ValueError(f"REBAlANCE_DEFAULT={config.REBALANCE_DEFAULT} missing from schedule run set.")

    strategy_metrics = schedule_metrics[config.REBALANCE_DEFAULT]

    # 9) Benchmarks
    voo_bh_res = run_buy_and_hold(
        extended_voo_proxy.reindex(strategy_res["equity_curve"].index).ffill(),
        config,
        cash_curve,
    )
    spy_bh_res = run_buy_and_hold(spy_close.reindex(strategy_res["equity_curve"].index).ffill(), config, cash_curve)

    voo_bh_metrics = compute_metrics(
        voo_bh_res["equity_curve"],
        voo_bh_res["weekly_returns"],
        voo_bh_res["trade_log"],
        voo_bh_res["positions"],
        effective_cash_rate,
        active_count=voo_bh_res.get("active_count"),
        eligible_count=voo_bh_res.get("eligible_count"),
        exposure=voo_bh_res.get("exposure"),
    )
    spy_bh_metrics = compute_metrics(
        spy_bh_res["equity_curve"],
        spy_bh_res["weekly_returns"],
        spy_bh_res["trade_log"],
        spy_bh_res["positions"],
        effective_cash_rate,
        active_count=spy_bh_res.get("active_count"),
        eligible_count=spy_bh_res.get("eligible_count"),
        exposure=spy_bh_res.get("exposure"),
    )

    # 10) Proxy fidelity diagnostics
    passive_proxy = _compute_passive_proxy_returns(close_returns, membership, base_weights)
    proxy_returns = passive_proxy["returns"]
    proxy_coverage = passive_proxy["coverage"]

    voo_benchmark_returns = extended_voo_proxy.reindex(proxy_returns.index).pct_change(fill_method=None).fillna(0.0)

    fidelity_overall = estimate_proxy_fidelity(
        proxy_returns=proxy_returns,
        benchmark_returns=voo_benchmark_returns,
        coverage=proxy_coverage,
        te_high=float(config.PROXY_TRACKING_ERROR_HIGH),
        te_medium=float(config.PROXY_TRACKING_ERROR_MEDIUM),
    )

    fidelity_regimes = _proxy_regime_deltas(
        proxy_returns=proxy_returns,
        benchmark_returns=voo_benchmark_returns,
        periods=get_periods(),
        cash_rate=effective_cash_rate,
    )

    fidelity_report = pd.concat(
        [
            fidelity_overall.assign(section="overall"),
            fidelity_regimes.assign(section="regime"),
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    # 11) Optional SMA-length sweep (default schedule)
    default_calendar = build_rebalance_calendar(trading_index, config.REBALANCE_DEFAULT)
    sma_rows: list[dict[str, Any]] = []
    sweep_excess_returns: list[pd.Series] = []
    for sma_len in config.SMA_SWEEP_VALUES:
        LOGGER.info("Running SMA sweep length: %s", sma_len)
        sma_i = compute_sma_matrix(close_df, int(sma_len))
        active_i = generate_active_mask(
            prices=close_df,
            sma=sma_i,
            signal_type=config.SIGNAL_TYPE,
            entry_band_bps=config.ENTRY_BAND_BPS,
            exit_band_bps=config.EXIT_BAND_BPS,
        )
        res_i = run_constituent_backtest(
            price_df=close_df,
            return_df=close_returns,
            membership_mask=membership,
            active_mask=active_i,
            rebalance_calendar=default_calendar,
            config=config,
            cash_curve=cash_curve,
            store_weights=False,
            store_cost_attribution=False,
        )
        met_i = compute_metrics(
            res_i["equity_curve"],
            res_i["weekly_returns"],
            res_i["trade_log"],
            res_i["positions"],
            effective_cash_rate,
            active_count=res_i.get("active_count"),
            eligible_count=res_i.get("eligible_count"),
            exposure=res_i.get("exposure"),
        )
        sweep_excess_returns.append(
            res_i["period_returns"] - effective_cash_rate / 252.0
        )
        sma_rows.append(
            {
                "sma_length": int(sma_len),
                "cagr": met_i["cagr"],
                "sharpe": met_i["sharpe"],
                "max_drawdown": met_i["max_drawdown"],
                "annual_turnover": met_i["annual_turnover"],
            }
        )
        del sma_i
        del active_i
        del res_i
        gc.collect()

    sma_sweep_df = pd.DataFrame(sma_rows)

    # Family-wise error control across the sweep. The Deflated Sharpe asks
    # whether the SELECTED configuration beats the expected best of N noise
    # strategies; Romano-Wolf asks which configurations have a mean excess
    # return above zero while holding the probability of ANY false rejection
    # across the whole family at 5%. The two answer different questions and
    # both belong in a sweep report.
    if len(sweep_excess_returns) >= 2:
        try:
            # The null is deliberately the BENCHMARK, not cash. Excess return
            # over cash is a weak null that any long-equity strategy clears --
            # testing against it found all five configurations "significant",
            # which says only that stocks beat T-bills. The question worth
            # asking is whether a configuration beats simply holding the index.
            bench_ret = spy_close.pct_change(fill_method=None)
            sweep_panel = pd.concat(
                [
                    (s + effective_cash_rate / 252.0)      # undo the cash excess
                    .sub(bench_ret.reindex(s.index).fillna(0.0))
                    .rename(f"sma_{int(n)}")
                    for s, n in zip(sweep_excess_returns, config.SMA_SWEEP_VALUES)
                ],
                axis=1,
            )
            rw = romano_wolf_stepdown(sweep_panel, alpha=0.05, n_boot=1000)
            n_sig = int(rw["significant"].sum())
            LOGGER.info(
                "Romano-Wolf stepdown vs benchmark (FWER 5%%): %s of %s "
                "configurations significant",
                n_sig, len(rw),
            )
            rw.to_csv(Path(config.OUTPUT_DIR) / "romano_wolf_sweep.csv", index=False)
        except Exception as exc:
            LOGGER.warning("Romano-Wolf stepdown skipped: %s", exc)

        # Persist the per-configuration daily return panel. It is the input the
        # Probability of Backtest Overfitting needs, and without it PBO cannot
        # be recomputed without rerunning the whole backtest.
        try:
            sweep_returns = pd.concat(
                [
                    (s + effective_cash_rate / 252.0).rename(f"sma_{int(n)}")
                    for s, n in zip(sweep_excess_returns, config.SMA_SWEEP_VALUES)
                ],
                axis=1,
            )
            sweep_returns.to_csv(Path(config.OUTPUT_DIR) / "sma_sweep_returns.csv")

            # PBO asks whether picking the best configuration IN SAMPLE tells
            # you anything about its OUT-OF-SAMPLE rank. The Deflated Sharpe
            # asks whether the selected Sharpe survives the search; these are
            # different questions and the report carries both.
            pbo = probability_of_backtest_overfitting(sweep_returns, n_splits=16)
            LOGGER.info(
                "PBO (CSCV, %s blocks, %s splits): %.3f; median OOS rank of the "
                "in-sample winner %.1f of %s",
                pbo["n_splits"], pbo["n_combinations"], pbo["pbo"],
                pbo["median_oos_rank"], pbo["n_configs"],
            )
            pd.DataFrame({
                "pbo": [pbo["pbo"]],
                "n_combinations": [pbo["n_combinations"]],
                "n_configs": [pbo["n_configs"]],
                "n_splits": [pbo["n_splits"]],
                "obs_used": [pbo["obs_used"]],
                "median_oos_rank": [pbo["median_oos_rank"]],
            }).to_csv(Path(config.OUTPUT_DIR) / "pbo_sweep.csv", index=False)
        except Exception as exc:
            LOGGER.warning("PBO skipped: %s", exc)

    # Deflated Sharpe (Bailey / Lopez de Prado) for every sweep configuration.
    # The multiple-testing pool counts every configuration this run evaluates:
    # the SMA grid plus the rebalance-schedule sweep. Cross-trial Sharpe
    # dispersion comes from the SMA sweep's own daily excess returns.
    if sweep_excess_returns:
        trial_sharpes = [per_period_sharpe(s) for s in sweep_excess_returns]
        n_trials = len(config.SMA_SWEEP_VALUES) + len(config.REBALANCE_SWEEP_VALUES)
        sma_sweep_df["deflated_sharpe"] = [
            deflated_sharpe(s, n_trials=n_trials, trial_sharpes=trial_sharpes)
            for s in sweep_excess_returns
        ]
        strategy_excess = (
            strategy_res["period_returns"] - effective_cash_rate / 252.0
        )
        strategy_metrics["deflated_sharpe"] = deflated_sharpe(
            strategy_excess, n_trials=n_trials, trial_sharpes=trial_sharpes
        )
        LOGGER.info(
            "Deflated Sharpe (n_trials=%s): strategy=%.3f",
            n_trials,
            strategy_metrics["deflated_sharpe"],
        )

    # 12) Summary tables + decomposition
    summary_metrics = {
        "Strategy": strategy_metrics,
        "Buy-and-Hold": voo_bh_metrics,
        "S&P 500 TR": spy_bh_metrics,
    }

    periods_payload = _period_metrics(
        {
            "Strategy": strategy_res,
            "Buy-and-Hold": voo_bh_res,
            "S&P 500 TR": spy_bh_res,
        },
        get_periods(),
        effective_cash_rate,
    )
    periods_payload["__schedule_comparison__"] = schedule_df
    periods_payload["__proxy_fidelity__"] = fidelity_overall

    summary_text = print_summary_table(summary_metrics, periods_payload)

    # 13) Reporting artifacts
    summary_df = pd.DataFrame(summary_metrics).T
    summary_df.index.name = "strategy"
    summary_df.to_csv(output_dir / "results_summary.csv")

    (output_dir / "results_summary.txt").write_text(summary_text, encoding="utf-8")

    schedule_df.to_csv(output_dir / "schedule_comparison.csv", index=False)
    sma_sweep_df.to_csv(output_dir / "sma_sweep.csv", index=False)

    active_breadth_df = pd.DataFrame(
        {
            "date": strategy_res["equity_curve"].index,
            "active_count": strategy_res["active_count"].values,
            "eligible_count": strategy_res["eligible_count"].values,
            "active_breadth_pct": (
                strategy_res["active_count"].values
                / np.where(strategy_res["eligible_count"].values > 0, strategy_res["eligible_count"].values, np.nan)
            ),
            "exposure": strategy_res["exposure"].values,
        }
    )
    active_breadth_df.to_csv(output_dir / "active_breadth.csv", index=False)

    fidelity_report.to_csv(output_dir / "proxy_fidelity_report.csv", index=False)

    strategy_res["weights"].to_parquet(output_dir / "weights_default.parquet")
    strategy_res["trade_log"].to_csv(output_dir / "trade_log.csv", index=False)
    strategy_res["cost_attribution"].to_csv(output_dir / "cost_attribution.csv", index=False)

    equity_export = pd.DataFrame(
        {
            "date": strategy_res["equity_curve"].index,
            "strategy_equity": strategy_res["equity_curve"].values,
            "buy_and_hold_voo_equity": voo_bh_res["equity_curve"].reindex(strategy_res["equity_curve"].index).values,
            "sp500_tr_equity": spy_bh_res["equity_curve"].reindex(strategy_res["equity_curve"].index).values,
            "strategy_return": strategy_res["weekly_returns"].values,
            "buy_and_hold_voo_return": voo_bh_res["weekly_returns"].reindex(strategy_res["equity_curve"].index).values,
            "sp500_tr_return": spy_bh_res["weekly_returns"].reindex(strategy_res["equity_curve"].index).values,
            "strategy_exposure": strategy_res["exposure"].values,
        }
    )
    equity_export.to_csv(output_dir / "equity_curves.csv", index=False)

    strategy_dd = compute_drawdown_series(strategy_res["equity_curve"])
    bh_dd = compute_drawdown_series(voo_bh_res["equity_curve"])

    rolling_window = 756  # ~3y daily window
    strategy_rs = compute_rolling_sharpe(strategy_res["weekly_returns"], window_weeks=rolling_window, cash_rate=effective_cash_rate)
    bh_rs = compute_rolling_sharpe(voo_bh_res["weekly_returns"], window_weeks=rolling_window, cash_rate=effective_cash_rate)

    plot_equity_curves(
        strategy_eq=strategy_res["equity_curve"],
        bh_eq=voo_bh_res["equity_curve"],
        positions=strategy_res["exposure"],
        output_dir=config.OUTPUT_DIR,
    )
    plot_drawdowns(strategy_dd, bh_dd, config.OUTPUT_DIR)
    plot_rolling_sharpe(strategy_rs, bh_rs, config.OUTPUT_DIR)
    plot_sma_sweep(sma_sweep_df, config.OUTPUT_DIR)
    plot_schedule_comparison(schedule_df, config.OUTPUT_DIR)
    plot_schedule_risk_return(schedule_df, voo_bh_metrics, config.OUTPUT_DIR)
    plot_active_breadth(active_breadth_df, config.OUTPUT_DIR)
    plot_cost_diagnostics(strategy_res["trade_log"], strategy_res["equity_curve"], config.OUTPUT_DIR)
    plot_regime_comparison(periods_payload, config.OUTPUT_DIR)
    write_detailed_report(
        output_dir=config.OUTPUT_DIR,
        summary_metrics=summary_metrics,
        schedule_df=schedule_df,
        sma_sweep_df=sma_sweep_df,
        fidelity_report=fidelity_report,
        periods=periods_payload,
        config=config,
    )

    # 14) Run manifest
    snapshot_rows: list[dict[str, Any]] = [
        snapshot_row(voo_source, "VOO", voo_daily, config.START_DATE, config.END_DATE),
        snapshot_row(spy_source, "SPY", spy_daily, config.START_DATE, config.END_DATE),
        snapshot_row("secproxy", "VOO", sec_holdings, config.PRE2019_PROXY_CUTOFF, config.END_DATE),
        snapshot_row("sp500pub", "membership", sp500_events, config.START_DATE, config.END_DATE),
    ]
    snapshot_rows.extend(constituentsnapshot_rows)

    if cash_curve is not None:
        snapshot_rows.append(
            build_snapshot_manifest_row(
                vendor="fred",
                ticker=str(cash_curve.attrs.get("source_series", config.CASH_RATE_SOURCE)),
                snapshot_id=cash_curve.attrs.get("snapshot_id"),
                fetched_at_utc=cash_curve.attrs.get("fetched_at_utc"),
                sha256=cash_curve.attrs.get("sha256"),
                requested_start=str(cash_curve.attrs.get("requested_start", config.START_DATE)),
                requested_end=str(cash_curve.attrs.get("requested_end", config.END_DATE)),
            )
        )

    manifest = {
        "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "config_hash": config_hash(config),
        "config": {
            k: v
            for k, v in vars(config).items()
            if k not in {"FRED_API_KEY"}
        },
        "package_versions": _package_versions(),
        "snapshots": snapshot_rows,
        "strategy_mode": config.STRATEGY_MODE,
        "price_data_provider": config.PRIMARY_PRICE_SOURCE,
        "price_data_provider_mix": provider_mix,
        "default_rebalance": config.REBALANCE_DEFAULT,
        "schedule_sweep": config.REBALANCE_SWEEP_VALUES,
        "universe_metadata": {
            "post_2019_source": config.UNIVERSE_POST_2019_SOURCE,
            "pre_2019_source": config.UNIVERSE_PRE_2019_SOURCE,
            "holdings_lag_business_days": config.HOLDINGS_LAG_BUSINESS_DAYS,
            "days_sec_proxy": int((pd.Series(source_by_date) == "sec_proxy").sum()),
            "days_preproxy": int((pd.Series(source_by_date) == "sp500_public_history").sum()),
            "avg_eligible_names": float(membership.sum(axis=1).mean()),
            "avg_price_coverage": float(coverage_ratio.mean()),
        },
        "proxy_fidelity": {
            "overall": fidelity_overall.iloc[0].to_dict() if not fidelity_overall.empty else {},
            "regime_rows": len(fidelity_regimes),
        },
        "schedule_metrics": schedule_df.to_dict(orient="records"),
    }

    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)

    elapsed = time.perf_counter() - started
    LOGGER.info("Pipeline complete in %.2f seconds.", elapsed)


if __name__ == "__main__":
    main()

"""Rebalance timing luck, index-versus-stock timing, and a positive control.

Three studies that share one panel, which is why they share one entrypoint:
building the point-in-time CRSP panel dominates the runtime, and each study on
its own would pay for it again.

1. Timing luck. Every SMA length is run against all 27 signal-evaluation
   schedules (daily, five weekly anchors, twenty-one monthly anchors) and the
   dispersion is measured against the gap between the daily rule and the index.
2. Index versus stock. The same five rules applied to the S&P 500 total-return
   series instead of to its constituents, with the shortfall split into cost and
   whipsaw by rerunning both with costs switched off.
3. Positive control. Moreira-Muir volatility-managed exposure, to establish that
   the pipeline can find a timing effect the literature says exists.

Outputs land in ``output/`` as CSVs plus ``timing_luck_box.png``.

    python run_timing_luck.py
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_engine import run_buy_and_hold, run_constituent_backtest
from config import BacktestConfig, load_config
from metrics import compute_metrics
from panel import BacktestPanel, build_panel, extract_adjusted_series
from report_evidence import (
    CONTROL_DAILY_NAME,
    CONTROL_MANIFEST_NAME,
    committed_report_path,
    write_daily_evidence,
    write_manifest,
)
from spread_edge import edge_spread_series
from statistics_mt import deflated_sharpe, per_period_sharpe, romano_wolf_stepdown
from strategy import compute_sma_matrix, generate_active_mask
from timing_luck import (
    HEADLINE_SCHEDULE,
    EvaluationSchedule,
    build_evaluation_calendar,
    enumerate_schedules,
    schedule_diagnostics,
    timing_luck_summary,
)
from vol_managed import (
    DEFAULT_CAP,
    DEFAULT_WINDOW,
    apply_vol_management,
    calibrate_c,
    cash_returns_from_annual_yield,
)

LOGGER = logging.getLogger(__name__)

# The evaluation schedule every cross-study comparison is anchored on. Part 1
# scales its dispersion against the daily rule, so Parts 2 and 3 use the daily
# rule too rather than the README's semi-monthly reporting default.
REFERENCE_SCHEDULE = EvaluationSchedule("daily")

#: Tracked copy of the summary tables. output/ is gitignored and needs a WRDS
#: entitlement to regenerate, so the tables the README quotes are also written
#: here and committed. These are portfolio-level aggregates only: no CRSP row
#: and nothing from data_cache/ is ever written to this directory.
REPORTS_DIR = Path("reports")
CORRECTED_REPORT_NAMES = {
    "vol_managed_control.csv": "vol_managed_control_corrected.csv",
    "vol_managed_romano_wolf.csv": "vol_managed_romano_wolf_corrected.csv",
}


def _write_table(frame: pd.DataFrame, out_dir: Path, name: str, index: bool = False) -> None:
    """Write one derived table to output/ and to the tracked reports/ copy."""
    frame.to_csv(out_dir / name, index=index)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        REPORTS_DIR / CORRECTED_REPORT_NAMES.get(name, name),
        index=index,
    )


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
    fh = logging.FileHandler(out / "timing_luck.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def zero_cost_config(config: BacktestConfig) -> BacktestConfig:
    """Return a copy with every modelled implementation cost switched off.

    Used for the cost-versus-whipsaw decomposition: the difference between a
    run under this config and the same run under the real one is, by
    construction, exactly the cost drag.
    """
    return dataclasses.replace(
        config,
        ENABLE_ENHANCED_COST_MODEL=False,
        SLIPPAGE_BPS=0.0,
        OPEN_AUCTION_SLIPPAGE_BPS=0.0,
        COMMISSION_PER_TRADE=0.0,
        COMMISSION_PER_SHARE=0.0,
        MIN_COMMISSION_PER_ORDER=0.0,
        EXPLICIT_FEE_BPS=0.0,
        INCLUDE_REGULATORY_FEES=False,
    )


def _metrics_row(result: dict[str, Any], cash_rate: float) -> dict[str, float]:
    return compute_metrics(
        result["equity_curve"],
        result["weekly_returns"],
        result["trade_log"],
        result["positions"],
        cash_rate,
        active_count=result.get("active_count"),
        eligible_count=result.get("eligible_count"),
        exposure=result.get("exposure"),
    )


# --------------------------------------------------------------------------
# Part 1: timing luck
# --------------------------------------------------------------------------


def run_schedule_sweep(
    panel: BacktestPanel,
    config: BacktestConfig,
    sma_lengths: list[int],
    schedules: list[EvaluationSchedule],
    *,
    checkpoint: Path | None = None,
) -> pd.DataFrame:
    """Run every (SMA length, evaluation schedule) pair.

    The signal mask depends only on the length, so it is computed once per
    length and reused across all schedules; only the execution calendar varies.
    """
    rows: list[dict[str, Any]] = []
    total = len(sma_lengths) * len(schedules)
    done = 0

    for length in sma_lengths:
        LOGGER.info("Timing-luck sweep: building signal for SMA %s.", length)
        sma = compute_sma_matrix(panel.close_df, int(length))
        active = generate_active_mask(
            prices=panel.close_df,
            sma=sma,
            signal_type=config.SIGNAL_TYPE,
            entry_band_bps=config.ENTRY_BAND_BPS,
            exit_band_bps=config.EXIT_BAND_BPS,
        )
        del sma
        gc.collect()

        for schedule in schedules:
            started = time.perf_counter()
            calendar = build_evaluation_calendar(panel.trading_index, schedule)
            result = run_constituent_backtest(
                price_df=panel.close_df,
                return_df=panel.close_returns,
                membership_mask=panel.membership,
                active_mask=active,
                rebalance_calendar=calendar,
                config=config,
                cash_curve=panel.cash_curve,
                store_weights=False,
                store_cost_attribution=False,
            )
            met = _metrics_row(result, panel.effective_cash_rate)
            diag = schedule_diagnostics(panel.trading_index, schedule)
            rows.append(
                {
                    "sma_length": int(length),
                    "frequency": schedule.frequency,
                    "anchor": schedule.anchor,
                    "label": schedule.label,
                    "is_anchor_variant": bool(schedule.is_anchor_variant),
                    "cagr": met["cagr"],
                    "sharpe": met["sharpe"],
                    "annualized_vol": met["annualized_vol"],
                    "max_drawdown": met["max_drawdown"],
                    "annual_turnover": met["annual_turnover"],
                    "avg_trade_cost_bps": met["avg_trade_cost_bps"],
                    "total_cost_bps_annualized": met["total_cost_bps_annualized"],
                    "n_rebalances": diag["n_rebalances"],
                    "n_short_period_fallbacks": diag["n_short_period_fallbacks"],
                }
            )
            done += 1
            LOGGER.info(
                "[%s/%s] sma=%s %s: CAGR=%.4f Sharpe=%.4f turnover=%.2f (%.1fs)",
                done, total, length, schedule.label, met["cagr"], met["sharpe"],
                met["annual_turnover"], time.perf_counter() - started,
            )
            del result
            gc.collect()

            if checkpoint is not None:
                pd.DataFrame(rows).to_csv(checkpoint, index=False)

        del active
        gc.collect()

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Part 2: index versus stock
# --------------------------------------------------------------------------


def _benchmark_ohlc(panel: BacktestPanel, ticker: str) -> pd.DataFrame:
    """Adjusted OHLC for a benchmark ETF, on the strategy trading calendar."""
    raw = panel.spy_daily if ticker == "SPY" else panel.voo_daily
    frame = pd.DataFrame(
        {
            field: extract_adjusted_series(raw, field).sort_index()
            for field in ("open", "high", "low", "close")
        }
    )
    return frame.reindex(panel.trading_index)


def index_spread_bps(panel: BacktestPanel, config: BacktestConfig) -> tuple[pd.Series, dict[str, Any]]:
    """EDGE effective spread for the index instrument, in basis points.

    The series being timed is the S&P 500 total-return path, whose tradable
    instrument is SPY before VOO's 2010 inception and VOO after. SPY has CRSP
    coverage across the whole sample and VOO does not, so SPY's own EDGE
    estimate is used throughout and VOO's is reported alongside it as a
    cross-check over the window where both exist. Using the more liquid of the
    two ETFs is the assumption that favours the index-level rule, which is the
    right direction for a test whose conclusion is that the index-level rule
    still loses.
    """
    spy = _benchmark_ohlc(panel, "SPY")
    spy_spread = edge_spread_series(spy["open"], spy["high"], spy["low"], spy["close"]) * 10_000.0
    spy_spread = spy_spread.clip(lower=float(config.SPREAD_FLOOR_BPS), upper=float(config.SPREAD_CAP_BPS))

    notes: dict[str, Any] = {
        "instrument": "SPY",
        "spy_first_estimate": str(spy_spread.dropna().index.min().date()) if spy_spread.notna().any() else None,
        "spy_mean_bps": float(spy_spread.mean(skipna=True)),
    }

    try:
        # CRSP reuses the VOO ticker: it belonged to Vornado Operating Co from
        # 1998 to 2003 before the Vanguard ETF listed in 2010, and the name
        # history admits both. Truncating at the ETF's inception keeps the
        # cross-check on the ETF rather than on a spliced series.
        voo = _benchmark_ohlc(panel, "VOO")
        voo = voo.loc[voo.index >= pd.Timestamp(config.VOO_INCEPTION)].dropna(how="all")
        voo_spread = edge_spread_series(voo["open"], voo["high"], voo["low"], voo["close"]) * 10_000.0
        voo_spread = voo_spread.clip(lower=float(config.SPREAD_FLOOR_BPS), upper=float(config.SPREAD_CAP_BPS))
        overlap = spy_spread.dropna().index.intersection(voo_spread.dropna().index)
        notes["voo_first_estimate"] = str(voo_spread.dropna().index.min().date()) if voo_spread.notna().any() else None
        notes["voo_mean_bps"] = float(voo_spread.mean(skipna=True))
        notes["overlap_days"] = int(len(overlap))
        notes["mean_abs_gap_bps"] = (
            float((spy_spread.loc[overlap] - voo_spread.loc[overlap]).abs().mean()) if len(overlap) else np.nan
        )
    except Exception as exc:  # a missing VOO series must not sink the study
        LOGGER.warning("VOO spread cross-check unavailable: %s", exc)

    # Before the first EDGE estimate (the 126-day warm-up) hold the first value
    # constant rather than dropping to the floor.
    filled = spy_spread.ffill().bfill().fillna(float(config.SPREAD_FLOOR_BPS))
    notes["applied_mean_bps"] = float(filled.mean())
    return filled, notes


def run_index_level_rule(
    panel: BacktestPanel,
    config: BacktestConfig,
    length: int,
    schedule: EvaluationSchedule,
    spread_bps: pd.Series,
) -> dict[str, Any]:
    """Run one SMA rule on the S&P 500 total-return series as a single asset.

    Built explicitly rather than through ``backtest_engine.run_backtest`` so the
    ETF's EDGE spread and realised volatility reach the cost model; the wrapper
    would silently fall back to the spread floor.
    """
    index_ohlc = _benchmark_ohlc(panel, "SPY")
    close = index_ohlc["close"].astype(float)

    price = pd.DataFrame({"SP500TR": close})
    returns = pd.DataFrame({"SP500TR": close.pct_change(fill_method=None).fillna(0.0)})
    membership = pd.DataFrame(True, index=price.index, columns=price.columns)

    sma = compute_sma_matrix(price, int(length))
    active = generate_active_mask(
        prices=price,
        sma=sma,
        signal_type=config.SIGNAL_TYPE,
        entry_band_bps=config.ENTRY_BAND_BPS,
        exit_band_bps=config.EXIT_BAND_BPS,
    )

    price.attrs["open_df"] = pd.DataFrame({"SP500TR": index_ohlc["open"].astype(float)})
    price.attrs["spread_bps_est"] = pd.DataFrame({"SP500TR": spread_bps.reindex(price.index)})
    price.attrs["sigma_20d"] = pd.DataFrame(
        {"SP500TR": returns["SP500TR"].rolling(config.VOL_LOOKBACK_DAYS, min_periods=5).std()}
    )
    # An S&P 500 ETF's dollar volume dwarfs a retail order, so participation and
    # therefore modelled impact are nil; the ADV is set high enough to say so.
    price.attrs["adv_usd"] = pd.DataFrame({"SP500TR": 1e11}, index=price.index)

    return run_constituent_backtest(
        price_df=price,
        return_df=returns,
        membership_mask=membership,
        active_mask=active,
        rebalance_calendar=build_evaluation_calendar(panel.trading_index, schedule),
        config=config,
        cash_curve=panel.cash_curve,
        store_weights=False,
        store_cost_attribution=False,
    )


def run_constituent_rule(
    panel: BacktestPanel,
    config: BacktestConfig,
    length: int,
    schedule: EvaluationSchedule,
) -> dict[str, Any]:
    """Run one SMA rule across the constituent universe."""
    sma = compute_sma_matrix(panel.close_df, int(length))
    active = generate_active_mask(
        prices=panel.close_df,
        sma=sma,
        signal_type=config.SIGNAL_TYPE,
        entry_band_bps=config.ENTRY_BAND_BPS,
        exit_band_bps=config.EXIT_BAND_BPS,
    )
    del sma
    gc.collect()
    result = run_constituent_backtest(
        price_df=panel.close_df,
        return_df=panel.close_returns,
        membership_mask=panel.membership,
        active_mask=active,
        rebalance_calendar=build_evaluation_calendar(panel.trading_index, schedule),
        config=config,
        cash_curve=panel.cash_curve,
        store_weights=False,
        store_cost_attribution=False,
    )
    del active
    gc.collect()
    return result


def run_index_vs_stock(
    panel: BacktestPanel,
    config: BacktestConfig,
    sma_lengths: list[int],
    spread_bps: pd.Series,
) -> pd.DataFrame:
    """Index-level and constituent-level rules, with and without costs.

    Returns one row per (length, level, cost setting) so the decomposition can
    be assembled downstream without rerunning anything.
    """
    free = zero_cost_config(config)
    rows: list[dict[str, Any]] = []

    for length in sma_lengths:
        for costs_on, cfg in (("on", config), ("off", free)):
            idx_res = run_index_level_rule(panel, cfg, length, REFERENCE_SCHEDULE, spread_bps)
            idx_met = _metrics_row(idx_res, panel.effective_cash_rate)
            rows.append(
                {
                    "sma_length": int(length),
                    "level": "index",
                    "costs": costs_on,
                    "cagr": idx_met["cagr"],
                    "sharpe": idx_met["sharpe"],
                    "annualized_vol": idx_met["annualized_vol"],
                    "max_drawdown": idx_met["max_drawdown"],
                    "annual_turnover": idx_met["annual_turnover"],
                    "total_cost_bps_annualized": idx_met["total_cost_bps_annualized"],
                }
            )
            LOGGER.info("index sma=%s costs=%s CAGR=%.4f", length, costs_on, idx_met["cagr"])
            del idx_res
            gc.collect()

            con_res = run_constituent_rule(panel, cfg, length, REFERENCE_SCHEDULE)
            con_met = _metrics_row(con_res, panel.effective_cash_rate)
            rows.append(
                {
                    "sma_length": int(length),
                    "level": "constituent",
                    "costs": costs_on,
                    "cagr": con_met["cagr"],
                    "sharpe": con_met["sharpe"],
                    "annualized_vol": con_met["annualized_vol"],
                    "max_drawdown": con_met["max_drawdown"],
                    "annual_turnover": con_met["annual_turnover"],
                    "total_cost_bps_annualized": con_met["total_cost_bps_annualized"],
                }
            )
            LOGGER.info("constituent sma=%s costs=%s CAGR=%.4f", length, costs_on, con_met["cagr"])
            del con_res
            gc.collect()

    return pd.DataFrame(rows)


def decompose_shortfall(levels: pd.DataFrame) -> pd.DataFrame:
    """Split the constituent shortfall into cost and whipsaw.

    With ``on``/``off`` denoting the cost setting::

        shortfall = index_on - constituent_on
                  = (index_off - constituent_off)          <- whipsaw
                  + (constituent_drag - index_drag)        <- cost

    where ``drag = off - on`` for each level. The identity is exact, so the two
    components sum to the shortfall by construction rather than by fitting.
    """
    wide = levels.pivot_table(index="sma_length", columns=["level", "costs"], values="cagr")
    out = pd.DataFrame(index=wide.index)
    out["index_on"] = wide[("index", "on")]
    out["index_off"] = wide[("index", "off")]
    out["constituent_on"] = wide[("constituent", "on")]
    out["constituent_off"] = wide[("constituent", "off")]
    out["index_cost_drag"] = out["index_off"] - out["index_on"]
    out["constituent_cost_drag"] = out["constituent_off"] - out["constituent_on"]
    out["shortfall"] = out["index_on"] - out["constituent_on"]
    out["whipsaw_component"] = out["index_off"] - out["constituent_off"]
    out["cost_component"] = out["constituent_cost_drag"] - out["index_cost_drag"]
    out["residual"] = out["shortfall"] - out["whipsaw_component"] - out["cost_component"]
    return out.reset_index()


# --------------------------------------------------------------------------
# Part 3: positive control
# --------------------------------------------------------------------------


def equal_weight_constituent_returns(panel: BacktestPanel) -> pd.Series:
    """Daily return of an equal-weight portfolio of point-in-time members.

    This is the passive constituent portfolio the SMA rule trades against: the
    same names, the same weighting, no timing. Applying the volatility overlay
    to it asks whether timing works on this universe at all, independently of
    whether the moving average is the right timing signal.
    """
    mask = panel.membership.reindex(index=panel.trading_index, columns=panel.close_returns.columns).fillna(False)
    usable = mask & panel.close_returns.notna() & panel.close_df.notna()
    counts = usable.sum(axis=1)
    total = panel.close_returns.where(usable).sum(axis=1)
    out = (total / counts.replace(0, np.nan)).fillna(0.0)
    out.name = "equal_weight_constituents"
    return out


def _annualized_arithmetic_sharpe(returns: pd.Series, cash: pd.Series) -> float:
    """Sharpe on arithmetic daily excess returns, annualised by root-252.

    ``metrics.compute_metrics`` reports (CAGR - cash) / volatility, which is a
    geometric measure. Moreira and Muir report the arithmetic one, so both are
    carried here: the overlay cuts volatility and therefore closes part of the
    gap between the two, and quoting only one of them would decide the question
    by choice of statistic.
    """
    excess = (returns - cash.reindex(returns.index).fillna(0.0)).dropna()
    sd = float(excess.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(252.0))


def run_vol_managed_control(
    panel: BacktestPanel,
    config: BacktestConfig,
    legs: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the volatility-managed overlay on each supplied return series.

    Each leg is run at both weight-update frequencies, gross and net of the
    overlay's own turnover cost. The monthly variant is the paper's; the daily
    variant is what a literal reading of "trailing 21-day realised variance"
    implies. Reporting both is the point: the gap between them is the cost of
    retrading a slow signal quickly.

    Args:
        panel: Loaded panel, for the cash curve and trading calendar.
        config: Runtime config, for the initial capital.
        legs: Mapping of leg name to ``{"returns", "cost_bps", "description"}``.

    Returns:
        A metrics table and the panel of net-overlay-minus-benchmark daily
        returns used for the Romano-Wolf test, followed by the aggregate daily
        return series needed to reproduce every performance metric.
    """
    dates = pd.DatetimeIndex(panel.trading_index)
    train_end = dates[len(dates) // 2]
    evaluation_dates = dates[dates > train_end]
    if evaluation_dates.empty:
        raise ValueError("Volatility-control evaluation window is empty.")
    LOGGER.info("Vol-managed training half ends %s.", train_end.date())

    if panel.cash_curve is not None:
        annual_cash = (
            panel.cash_curve.reindex(dates)
            .ffill()
            .shift(1)
            .fillna(config.CASH_RATE_ANNUAL)
        )
    else:
        annual_cash = pd.Series(config.CASH_RATE_ANNUAL, index=dates)
    cash_daily = cash_returns_from_annual_yield(annual_cash)

    rows: list[dict[str, Any]] = []
    excess: dict[str, pd.Series] = {}
    daily: dict[str, pd.Series] = {
        "cash_return": cash_daily.loc[evaluation_dates].rename("cash_return")
    }

    def _row(**kwargs: Any) -> dict[str, Any]:
        rets = kwargs["rets"].reindex(evaluation_dates)
        equity = (1.0 + rets).cumprod() * float(config.INITIAL_CAPITAL)
        met = compute_metrics(
            equity,
            rets,
            pd.DataFrame(),
            pd.Series(1, index=equity.index, dtype="Int64"),
            panel.effective_cash_rate,
        )
        return {
            "leg": kwargs["leg"],
            "variant": kwargs["variant"],
            "description": kwargs["description"],
            "train_start": dates.min().date().isoformat(),
            "train_end": train_end.date().isoformat(),
            "evaluation_start": evaluation_dates.min().date().isoformat(),
            "evaluation_end": evaluation_dates.max().date().isoformat(),
            "n_evaluation_days": len(evaluation_dates),
            "cagr": met["cagr"],
            "annualized_vol": met["annualized_vol"],
            "sharpe_geometric": met["sharpe"],
            "sharpe_arithmetic": _annualized_arithmetic_sharpe(rets, cash_daily),
            "max_drawdown": met["max_drawdown"],
            "c": kwargs.get("c", np.nan),
            "avg_weight": kwargs.get("avg_weight", 1.0),
            "pct_time_levered": kwargs.get("pct_time_levered", 0.0),
            "annual_turnover": kwargs.get("annual_turnover", 0.0),
            "overlay_cost_bps": kwargs.get("overlay_cost_bps", 0.0),
        }

    for name, spec in legs.items():
        base = spec["returns"].reindex(dates).astype(float).fillna(0.0)
        c = calibrate_c(base, train_end=train_end, window=DEFAULT_WINDOW)
        daily[f"{name}__buy_and_hold_return"] = base.loc[evaluation_dates]

        rows.append(
            _row(
                leg=name,
                variant="buy_and_hold",
                description=spec["description"],
                rets=base,
            )
        )

        for update in ("monthly", "daily"):
            managed = apply_vol_management(
                base,
                c=c,
                cash_returns=cash_daily,
                window=DEFAULT_WINDOW,
                cap=DEFAULT_CAP,
                update=update,
                cost_bps=float(spec["cost_bps"]),
                initial_capital=float(config.INITIAL_CAPITAL),
            )
            shared = {
                "leg": name,
                "description": spec["description"],
                "c": c,
                "avg_weight": float(
                    managed["weights"].loc[evaluation_dates].mean()
                ),
                "pct_time_levered": float(
                    (managed["weights"].loc[evaluation_dates] > 1.0).mean()
                ),
                "annual_turnover": float(
                    managed["turnover"].loc[evaluation_dates].sum()
                    / (
                        (
                            evaluation_dates[-1] - evaluation_dates[0]
                        ).days
                        + 1
                    )
                    * 365.25
                ),
            }
            rows.append(
                _row(
                    variant=f"vol_managed_{update}_gross",
                    rets=managed["gross_returns"],
                    overlay_cost_bps=0.0,
                    **shared,
                )
            )
            daily[f"{name}__vol_managed_{update}_gross_return"] = managed[
                "gross_returns"
            ].loc[evaluation_dates]
            rows.append(
                _row(
                    variant=f"vol_managed_{update}_net",
                    rets=managed["returns"],
                    overlay_cost_bps=float(spec["cost_bps"]),
                    **shared,
                )
            )
            daily[f"{name}__vol_managed_{update}_net_return"] = managed[
                "returns"
            ].loc[evaluation_dates]
            excess[f"{name}__{update}"] = (
                managed["returns"].loc[evaluation_dates]
                - base.loc[evaluation_dates]
            ).rename(f"{name}__{update}")

    return (
        pd.DataFrame(rows),
        pd.concat(excess.values(), axis=1),
        pd.DataFrame(daily, index=evaluation_dates),
    )


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", default="1,2,3", help="Comma-separated subset of {1,2,3}.")
    args = parser.parse_args()
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}

    started = time.perf_counter()
    config = load_config()
    _setup_logging(config.OUTPUT_DIR)
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building panel.")
    panel = build_panel(config)

    index_bh = run_buy_and_hold(panel.spy_close.reindex(panel.trading_index).ffill(), config, panel.cash_curve)
    index_met = _metrics_row(index_bh, panel.effective_cash_rate)
    index_cagr = float(index_met["cagr"])
    LOGGER.info("S&P 500 TR buy-and-hold: CAGR=%.4f Sharpe=%.4f", index_cagr, index_met["sharpe"])

    lengths = [int(n) for n in config.SMA_SWEEP_VALUES]

    if "1" in parts:
        LOGGER.info("=== Part 1: rebalance timing luck ===")
        # The 27 anchors, plus the semi-monthly schedule main.py reports. The
        # headline is not an anchor variant and does not enter the dispersion
        # statistics; it is run here so the README can say where it lands.
        variants = run_schedule_sweep(
            panel, config, lengths, [*enumerate_schedules(), HEADLINE_SCHEDULE],
            checkpoint=out_dir / "timing_luck_variants.csv",
        )
        _write_table(variants, out_dir, "timing_luck_variants.csv")
        summary = timing_luck_summary(variants, index_cagr=index_cagr)
        _write_table(summary, out_dir, "timing_luck_summary.csv")
        from reporting import plot_timing_luck_box

        plot_timing_luck_box(variants, index_cagr, config.OUTPUT_DIR)
        # output/ is gitignored, so the README's copy goes somewhere tracked.
        plot_timing_luck_box(variants, index_cagr, "figures")
        LOGGER.info("Part 1 summary:\n%s", summary.to_string(index=False))

    if "2" in parts:
        LOGGER.info("=== Part 2: index versus stock ===")
        spread_bps, spread_notes = index_spread_bps(panel, config)
        pd.DataFrame([spread_notes]).to_csv(out_dir / "index_spread_notes.csv", index=False)
        LOGGER.info("Index spread assumption: %s", spread_notes)

        levels = run_index_vs_stock(panel, config, lengths, spread_bps)
        _write_table(levels, out_dir, "index_vs_stock.csv")
        decomposition = decompose_shortfall(levels)
        _write_table(decomposition, out_dir, "index_vs_stock_decomposition.csv")
        LOGGER.info("Part 2 decomposition:\n%s", decomposition.to_string(index=False))

    if "3" in parts:
        LOGGER.info("=== Part 3: volatility-managed positive control ===")
        index_returns = panel.spy_close.reindex(panel.trading_index).ffill().pct_change(fill_method=None).fillna(0.0)
        ew_returns = equal_weight_constituent_returns(panel)

        spread_bps, _ = index_spread_bps(panel, config)
        # One-way overlay cost: half the effective spread plus the modelled
        # opening-auction slippage, the same components the engine charges.
        index_cost_bps = float(spread_bps.mean()) / 2.0 + float(config.OPEN_AUCTION_SLIPPAGE_BPS)
        # The constituent overlay trades the whole basket, so it pays the
        # per-name cost the constituent runs actually realise.
        sma_res = run_constituent_rule(panel, config, config.SMA_LENGTH_DAYS, REFERENCE_SCHEDULE)
        sma_met = _metrics_row(sma_res, panel.effective_cash_rate)
        basket_cost_bps = float(sma_met["avg_trade_cost_bps"])
        sma_returns = sma_res["period_returns"].copy()
        del sma_res
        gc.collect()

        legs = {
            "index": {
                "returns": index_returns,
                "cost_bps": index_cost_bps,
                "description": "S&P 500 total-return series",
            },
            "constituents_equal_weight": {
                "returns": ew_returns,
                "cost_bps": basket_cost_bps,
                "description": "Equal-weight point-in-time constituents",
            },
            "constituents_sma200": {
                "returns": sma_returns,
                "cost_bps": basket_cost_bps,
                "description": f"Constituent SMA-{config.SMA_LENGTH_DAYS}, daily evaluation",
            },
        }

        control, excess_panel, daily = run_vol_managed_control(panel, config, legs)

        # Same multiple-testing treatment as the SMA sweep: Romano-Wolf over the
        # family of overlays against their own buy-and-hold, then a Deflated
        # Sharpe for each with the whole family as the trial pool.
        rw = romano_wolf_stepdown(excess_panel, alpha=0.05, n_boot=1000)
        _write_table(rw, out_dir, "vol_managed_romano_wolf.csv")
        LOGGER.info("Vol-managed Romano-Wolf:\n%s", rw.to_string(index=False))

        trial_sharpes = [per_period_sharpe(excess_panel[col]) for col in excess_panel.columns]
        n_trials = int(len(excess_panel.columns))
        dsr = {
            col: deflated_sharpe(excess_panel[col], n_trials=n_trials, trial_sharpes=trial_sharpes)
            for col in excess_panel.columns
        }
        control["deflated_sharpe_vs_bh"] = control.apply(
            lambda r: dsr.get(f"{r['leg']}__{r['variant'].split('_')[2]}", np.nan)
            if r["variant"].endswith("_net")
            else np.nan,
            axis=1,
        )
        daily.index.name = "date"
        daily = daily.reset_index()
        daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
        artifact = write_daily_evidence(
            daily,
            output_dir=out_dir,
            report_dir=REPORTS_DIR,
            name=CONTROL_DAILY_NAME,
        )
        source_labels = {
            "cash_return": {
                "source_label": "cash_sleeve:elapsed_calendar_day_return",
                "leg": "cash",
                "variant": "cash_return",
            }
        }
        for row in control.to_dict(orient="records"):
            column = f"{row['leg']}__{row['variant']}_return"
            source_labels[column] = {
                "source_label": f"{row['leg']}:{row['variant']}",
                "leg": row["leg"],
                "variant": row["variant"],
                "description": row["description"],
            }
        write_manifest(
            {
                **artifact,
                "schema_version": 1,
                "content": "portfolio-level daily simple returns only",
                "date_column": "date",
                "cash_return_column": "cash_return",
                "return_columns": source_labels,
                "metric_policy": {
                    "periods_per_year": 252.0,
                    "geometric_sharpe_cash_rate_annual": (
                        panel.effective_cash_rate
                    ),
                    "arithmetic_sharpe_cash_return_column": "cash_return",
                },
            },
            output_dir=out_dir,
            report_dir=REPORTS_DIR,
            name=CONTROL_MANIFEST_NAME,
        )
        control["daily_returns_path"] = committed_report_path(CONTROL_DAILY_NAME)
        control["daily_return_column"] = control.apply(
            lambda r: f"{r['leg']}__{r['variant']}_return",
            axis=1,
        )
        control["cash_return_column"] = "cash_return"
        control["daily_returns_manifest_path"] = committed_report_path(
            CONTROL_MANIFEST_NAME
        )
        _write_table(control, out_dir, "vol_managed_control.csv")
        LOGGER.info("Part 3 control:\n%s", control.to_string(index=False))

    LOGGER.info("Done in %.1f s.", time.perf_counter() - started)


if __name__ == "__main__":
    main()

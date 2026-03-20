"""Reporting utilities for plots and tabular summaries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _shade_regimes(ax: plt.Axes, positions: pd.Series) -> None:
    """Shade long/cash regimes in the background."""
    pos = positions.fillna(0.0).astype(float)
    if pos.empty:
        return

    state = (pos > 0).astype(int)
    start = state.index[0]
    cur = int(state.iloc[0])

    for i in range(1, len(state)):
        v = int(state.iloc[i])
        if v != cur:
            color = "#2ca25f" if cur == 1 else "#de2d26"
            ax.axvspan(start, state.index[i], color=color, alpha=0.08, linewidth=0)
            start = state.index[i]
            cur = v

    color = "#2ca25f" if cur == 1 else "#de2d26"
    ax.axvspan(start, state.index[-1], color=color, alpha=0.08, linewidth=0)


def plot_equity_curves(
    strategy_eq: pd.Series,
    bh_eq: pd.Series,
    positions: pd.Series,
    output_dir: str,
) -> None:
    """Plot strategy versus benchmark equity curves on log scale.

    Args:
        strategy_eq: Strategy equity curve.
        bh_eq: Buy-and-hold benchmark equity curve.
        positions: Strategy long/cash indicator or exposure series.
        output_dir: Output directory.
    """
    out_dir = _ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strategy_eq.index, strategy_eq.values, lw=2.0, label="Strategy")
    ax.plot(bh_eq.index, bh_eq.values, lw=2.0, label="Buy-and-Hold VOO Proxy")
    _shade_regimes(ax, positions)

    ax.set_yscale("log")
    ax.set_title("Constituent 200-Day SMA Strategy vs Buy-and-Hold VOO Proxy")
    ax.set_ylabel("Equity (log scale)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    save_path = out_dir / "equity_curves.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved equity curve plot: %s", save_path)


def plot_drawdowns(strategy_dd: pd.Series, bh_dd: pd.Series, output_dir: str) -> None:
    """Plot underwater curves for strategy and benchmark.

    Args:
        strategy_dd: Strategy drawdown series.
        bh_dd: Buy-and-hold drawdown series.
        output_dir: Output directory.
    """
    out_dir = _ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(strategy_dd.index, strategy_dd.values, lw=1.8, label="Strategy")
    ax.plot(bh_dd.index, bh_dd.values, lw=1.8, label="Buy-and-Hold VOO Proxy")
    ax.fill_between(strategy_dd.index, strategy_dd.values, 0.0, alpha=0.15)
    ax.fill_between(bh_dd.index, bh_dd.values, 0.0, alpha=0.10)

    ax.set_title("Drawdowns (Underwater)")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend()

    save_path = out_dir / "drawdowns.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved drawdown plot: %s", save_path)


def plot_rolling_sharpe(strategy_rs: pd.Series, bh_rs: pd.Series, output_dir: str) -> None:
    """Plot rolling Sharpe ratios for strategy and benchmark.

    Args:
        strategy_rs: Strategy rolling Sharpe.
        bh_rs: Buy-and-hold rolling Sharpe.
        output_dir: Output directory.
    """
    out_dir = _ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(strategy_rs.index, strategy_rs.values, lw=1.8, label="Strategy")
    ax.plot(bh_rs.index, bh_rs.values, lw=1.8, label="Buy-and-Hold VOO Proxy")
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.5)

    ax.set_title("Rolling Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.25)
    ax.legend()

    save_path = out_dir / "rolling_sharpe.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved rolling Sharpe plot: %s", save_path)


def plot_sma_sweep(sweep_results: pd.DataFrame, output_dir: str) -> None:
    """Plot CAGR and Sharpe across SMA lengths.

    Args:
        sweep_results: DataFrame with columns ``sma_length``, ``cagr``, ``sharpe``.
        output_dir: Output directory.
    """
    out_dir = _ensure_output_dir(output_dir)

    if sweep_results.empty:
        LOGGER.warning("SMA sweep results empty; skipping sma_sweep plot.")
        return

    df = sweep_results.copy().sort_values("sma_length")
    x = np.arange(len(df))
    is_default = df["sma_length"].astype(int) == 200

    cagr_colors = np.where(is_default, "#1f78b4", "#a6cee3")
    sharpe_colors = np.where(is_default, "#e31a1c", "#fcbba1")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(x, df["cagr"], color=cagr_colors)
    axes[0].set_title("SMA Sweep: CAGR")
    axes[0].set_ylabel("CAGR")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, df["sharpe"], color=sharpe_colors)
    axes[1].set_title("SMA Sweep: Sharpe")
    axes[1].set_ylabel("Sharpe")
    axes[1].set_xlabel("SMA Length")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["sma_length"].astype(int).astype(str))
    axes[1].grid(True, axis="y", alpha=0.25)

    save_path = out_dir / "sma_sweep.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved SMA sweep plot: %s", save_path)


def plot_schedule_comparison(schedule_results: pd.DataFrame, output_dir: str) -> None:
    """Plot schedule comparison for CAGR and Sharpe.

    Args:
        schedule_results: DataFrame with columns ``frequency``, ``cagr``, ``sharpe``.
        output_dir: Output directory.
    """
    out_dir = _ensure_output_dir(output_dir)

    if schedule_results.empty:
        LOGGER.warning("schedule_results empty; skipping schedule comparison plot.")
        return

    order = ["daily", "weekly", "semi_monthly", "monthly"]
    df = schedule_results.copy()
    df["frequency"] = pd.Categorical(df["frequency"], categories=order, ordered=True)
    df = df.sort_values("frequency")

    x = np.arange(len(df))
    if "is_default" in df.columns:
        is_default = df["is_default"].fillna(False).astype(bool).to_numpy()
    else:
        is_default = (df["frequency"].astype(str) == "daily").to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(x, df["cagr"], color=np.where(is_default, "#2c7fb8", "#7fcdbb"))
    axes[0].set_title("Rebalance Schedule Comparison: CAGR")
    axes[0].set_ylabel("CAGR")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, df["sharpe"], color=np.where(is_default, "#d95f0e", "#fdbe85"))
    axes[1].set_title("Rebalance Schedule Comparison: Sharpe")
    axes[1].set_ylabel("Sharpe")
    axes[1].set_xlabel("Rebalance Frequency")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["frequency"].astype(str))
    axes[1].grid(True, axis="y", alpha=0.25)

    save_path = out_dir / "schedule_comparison.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved schedule comparison plot: %s", save_path)


def plot_active_breadth(active_breadth: pd.DataFrame, output_dir: str) -> None:
    """Plot active breadth and active count over time.

    Args:
        active_breadth: DataFrame with ``date``, ``active_breadth_pct``, ``active_count``.
        output_dir: Output directory.
    """
    out_dir = _ensure_output_dir(output_dir)

    if active_breadth.empty:
        LOGGER.warning("active_breadth empty; skipping active breadth plot.")
        return

    df = active_breadth.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["date"], df["active_breadth_pct"], color="#1f78b4", lw=1.8, label="Active Breadth")
    ax1.set_ylabel("Active Breadth")
    ax1.set_ylim(0, max(1.0, float(df["active_breadth_pct"].max(skipna=True) * 1.1)))

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["active_count"], color="#e31a1c", lw=1.2, alpha=0.7, label="Active Names")
    ax2.set_ylabel("Active Names")

    ax1.set_title("Active Breadth Through Time")
    ax1.grid(True, alpha=0.25)

    save_path = out_dir / "active_breadth.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved active breadth plot: %s", save_path)


def plot_cost_diagnostics(
    trade_log: pd.DataFrame,
    equity_curve: pd.Series,
    output_dir: str,
) -> None:
    """Plot monthly turnover and cost drag through time."""
    out_dir = _ensure_output_dir(output_dir)
    if trade_log.empty:
        LOGGER.warning("trade_log empty; skipping cost diagnostics plot.")
        return

    trades = trade_log.copy()
    trades["date"] = pd.to_datetime(trades["date"])
    daily = trades.groupby("date").agg(
        total_cost_usd=("total_cost_usd", "sum"),
        turnover_fraction=("turnover_fraction", "sum"),
    )
    nav = equity_curve.reindex(daily.index).ffill().replace(0.0, np.nan)
    daily["cost_bps"] = (daily["total_cost_usd"] / nav) * 10000.0
    monthly = daily.resample("ME").sum(min_count=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(monthly.index, monthly["turnover_fraction"], color="#2b8cbe", lw=1.8)
    axes[0].set_title("Monthly Turnover")
    axes[0].set_ylabel("Turnover")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(monthly.index, monthly["cost_bps"], color="#d95f0e", lw=1.8)
    axes[1].set_title("Monthly Trading Cost Drag")
    axes[1].set_ylabel("Cost (bps)")
    axes[1].grid(True, alpha=0.25)

    save_path = out_dir / "cost_diagnostics.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved cost diagnostics plot: %s", save_path)


def plot_schedule_risk_return(
    schedule_results: pd.DataFrame,
    benchmark_metrics: dict[str, float],
    output_dir: str,
) -> None:
    """Plot risk/return/turnover trade-off across rebalance schedules."""
    out_dir = _ensure_output_dir(output_dir)
    if schedule_results.empty:
        LOGGER.warning("schedule_results empty; skipping risk/return plot.")
        return

    df = schedule_results.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    size = np.clip(df["annual_turnover"].fillna(0.0).to_numpy(dtype=float) * 20.0, 80.0, 1000.0)
    ax.scatter(
        df["annualized_vol"],
        df["cagr"],
        s=size,
        c=np.where(df.get("is_default", False), "#d7301f", "#74a9cf"),
        alpha=0.8,
        edgecolors="black",
        linewidths=0.6,
    )
    for _, row in df.iterrows():
        ax.annotate(str(row["frequency"]), (row["annualized_vol"], row["cagr"]), xytext=(5, 5), textcoords="offset points")

    ax.scatter(
        [benchmark_metrics.get("annualized_vol", np.nan)],
        [benchmark_metrics.get("cagr", np.nan)],
        s=180,
        c="#238b45",
        marker="D",
        edgecolors="black",
        linewidths=0.6,
        label="Buy-and-Hold VOO Proxy",
    )

    ax.set_title("Schedule Trade-Off: CAGR vs Volatility (bubble size = turnover)")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("CAGR")
    ax.grid(True, alpha=0.25)
    ax.legend()

    save_path = out_dir / "schedule_risk_return.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved schedule risk/return plot: %s", save_path)


def plot_regime_comparison(periods: dict[str, Any], output_dir: str) -> None:
    """Plot regime CAGR and max drawdown comparison for strategy and benchmark."""
    out_dir = _ensure_output_dir(output_dir)
    rows: list[dict[str, float | str]] = []
    for period_name, payload in periods.items():
        if str(period_name).startswith("__") or not isinstance(payload, dict) or not payload.get("valid", True):
            continue
        metrics_block = payload.get("metrics", {})
        if not isinstance(metrics_block, dict):
            continue
        strat = metrics_block.get("Strategy", {})
        bench = metrics_block.get("Buy-and-Hold", {})
        rows.append(
            {
                "period": period_name,
                "strategy_cagr": strat.get("cagr", np.nan),
                "benchmark_cagr": bench.get("cagr", np.nan),
                "strategy_max_drawdown": strat.get("max_drawdown", np.nan),
                "benchmark_max_drawdown": bench.get("max_drawdown", np.nan),
            }
        )

    if not rows:
        LOGGER.warning("No valid period rows; skipping regime comparison plot.")
        return

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(x - width / 2, df["strategy_cagr"], width=width, color="#2171b5", label="Strategy")
    axes[0].bar(x + width / 2, df["benchmark_cagr"], width=width, color="#41ab5d", label="Buy-and-Hold")
    axes[0].set_title("Regime CAGR Comparison")
    axes[0].set_ylabel("CAGR")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - width / 2, df["strategy_max_drawdown"], width=width, color="#cb181d", label="Strategy")
    axes[1].bar(x + width / 2, df["benchmark_max_drawdown"], width=width, color="#fdae6b", label="Buy-and-Hold")
    axes[1].set_title("Regime Max Drawdown Comparison")
    axes[1].set_ylabel("Max Drawdown")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["period"], rotation=30, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    save_path = out_dir / "regime_comparison.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved regime comparison plot: %s", save_path)


def _format_metric(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    return f"{float(value):.4f}"


def print_summary_table(metrics_dict: dict[str, dict], periods: dict | None = None) -> str:
    """Build a formatted summary table.

    Args:
        metrics_dict: Mapping of strategy label to metric dict.
        periods: Optional period decomposition payload. Can include optional keys
            ``__schedule_comparison__`` and ``__proxy_fidelity__``.

    Returns:
        Formatted multi-section table string.
    """
    metric_order = [
        "cagr",
        "annualized_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "hit_ratio",
        "avg_win",
        "avg_loss",
        "avg_holding_period_weeks",
        "total_trades",
        "annual_turnover",
        "pct_time_in_market",
        "max_drawdown_duration_weeks",
        "avg_trade_cost_bps",
        "total_cost_bps_annualized",
        "avg_implementation_shortfall_bps",
        "avg_active_names",
        "active_breadth_pct",
        "gross_exposure_mean",
    ]

    rows: list[str] = []
    labels = list(metrics_dict.keys())

    header = "Metric".ljust(34) + " | " + " | ".join([x.ljust(18) for x in labels])
    rows.append(header)
    rows.append("-" * len(header))

    for metric in metric_order:
        line = metric.ljust(34) + " | "
        vals = [_format_metric(metrics_dict[label].get(metric, np.nan)).ljust(18) for label in labels]
        rows.append(line + " | ".join(vals))

    schedule_df: pd.DataFrame | None = None
    fidelity_df: pd.DataFrame | None = None

    if periods and isinstance(periods, dict):
        schedule_obj = periods.get("__schedule_comparison__")
        if isinstance(schedule_obj, pd.DataFrame):
            schedule_df = schedule_obj

        fidelity_obj = periods.get("__proxy_fidelity__")
        if isinstance(fidelity_obj, pd.DataFrame):
            fidelity_df = fidelity_obj

    if periods:
        period_rows: list[str] = []
        for period_name, payload in periods.items():
            if str(period_name).startswith("__"):
                continue
            if not isinstance(payload, dict):
                continue
            if not payload.get("valid", True):
                continue

            start = payload.get("start", "n/a")
            end = payload.get("end", "n/a")
            period_rows.append(f"{period_name} [{start}..{end}]")

            metrics_block = payload.get("metrics", {})
            if isinstance(metrics_block, dict):
                for lbl, m in metrics_block.items():
                    cagr = _format_metric(m.get("cagr", np.nan))
                    mdd = _format_metric(m.get("max_drawdown", np.nan))
                    period_rows.append(f"  {lbl}: CAGR={cagr}, MaxDD={mdd}")

        if period_rows:
            rows.append("")
            rows.append("Period Decomposition (CAGR / Max DD)")
            rows.append("-" * 80)
            rows.extend(period_rows)

    if schedule_df is not None and not schedule_df.empty:
        rows.append("")
        rows.append("Schedule Comparison")
        rows.append("-" * 80)
        for _, r in schedule_df.iterrows():
            rows.append(
                f"{str(r.get('frequency', 'n/a')).ljust(14)} "
                f"CAGR={_format_metric(r.get('cagr', np.nan))} "
                f"Sharpe={_format_metric(r.get('sharpe', np.nan))} "
                f"Turnover={_format_metric(r.get('annual_turnover', np.nan))}"
            )

    if fidelity_df is not None and not fidelity_df.empty:
        rows.append("")
        rows.append("Proxy Fidelity")
        rows.append("-" * 80)
        for _, r in fidelity_df.iterrows():
            rows.append(
                f"tracking_error_ann={_format_metric(r.get('tracking_error_ann', np.nan))} "
                f"corr={_format_metric(r.get('correlation', np.nan))} "
                f"mean_abs_diff_bps={_format_metric(r.get('mean_abs_diff_bps', np.nan))} "
                f"grade={r.get('fidelity_grade', 'n/a')}"
            )

    table = "\n".join(rows)
    LOGGER.info("\n%s", table)
    return table


def write_detailed_report(
    *,
    output_dir: str,
    summary_metrics: dict[str, dict[str, float]],
    schedule_df: pd.DataFrame,
    sma_sweep_df: pd.DataFrame,
    fidelity_report: pd.DataFrame,
    periods: dict[str, Any],
    config: Any,
) -> Path:
    """Write a markdown report summarizing the full backtest outcome."""
    out_dir = _ensure_output_dir(output_dir)
    default_label = str(getattr(config, "REBALANCE_DEFAULT", "default"))
    strategy = summary_metrics.get("Strategy", {})
    benchmark = summary_metrics.get("Buy-and-Hold", {})
    fidelity_overall = fidelity_report.loc[fidelity_report["section"].eq("overall")].head(1).copy()

    lines: list[str] = []
    lines.append("# Constituent-Level VOO SMA Backtest Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- Default implementation schedule: `{default_label}` with `200-day SMA`, `next_open` execution, and point-in-time constituent membership."
    )
    lines.append(
        f"- Strategy CAGR `{_format_metric(strategy.get('cagr', np.nan))}`, Sharpe `{_format_metric(strategy.get('sharpe', np.nan))}`, max drawdown `{_format_metric(strategy.get('max_drawdown', np.nan))}`."
    )
    lines.append(
        f"- Buy-and-hold VOO proxy CAGR `{_format_metric(benchmark.get('cagr', np.nan))}`, Sharpe `{_format_metric(benchmark.get('sharpe', np.nan))}`, max drawdown `{_format_metric(benchmark.get('max_drawdown', np.nan))}`."
    )
    lines.append(
        f"- Average active names `{_format_metric(strategy.get('avg_active_names', np.nan))}` and average active breadth `{_format_metric(strategy.get('active_breadth_pct', np.nan))}`."
    )
    lines.append("")
    lines.append("## Implementation Realism")
    lines.append("")
    lines.append("- Cash sleeve uses time-varying 3M Treasury data when available.")
    lines.append("- Execution assumes next-session open, opening-auction slippage, Corwin-Schultz spread estimates, retail-sized market impact, and sell-side FINRA TAF.")
    lines.append("- Point-in-time membership is lagged and snapshot-backed; price snapshots are frozen for reproducibility.")
    lines.append("- Residual limitation: pre-2019 constituent history remains a public proxy, not a licensed S&P point-in-time master.")
    lines.append("")
    lines.append("## Headline Metrics")
    lines.append("")
    summary_df = pd.DataFrame(summary_metrics).T
    lines.append("```text")
    lines.append(summary_df.to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Schedule Comparison")
    lines.append("")
    lines.append("```text")
    lines.append(schedule_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## SMA Sweep")
    lines.append("")
    lines.append("```text")
    lines.append(sma_sweep_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    if not fidelity_overall.empty:
        lines.append("## Proxy Fidelity")
        lines.append("")
        lines.append("```text")
        lines.append(fidelity_overall.to_string(index=False))
        lines.append("```")
        lines.append("")
    lines.append("## Regime Breakdown")
    lines.append("")
    for period_name, payload in periods.items():
        if str(period_name).startswith("__") or not isinstance(payload, dict) or not payload.get("valid", True):
            continue
        lines.append(f"### {period_name}")
        metrics_block = payload.get("metrics", {})
        if isinstance(metrics_block, dict):
            for label, metric_dict in metrics_block.items():
                lines.append(
                    f"- {label}: CAGR `{_format_metric(metric_dict.get('cagr', np.nan))}`, Max DD `{_format_metric(metric_dict.get('max_drawdown', np.nan))}`"
                )
        lines.append("")

    lines.append("## Generated Artifacts")
    lines.append("")
    artifacts = [
        "equity_curves.png",
        "drawdowns.png",
        "rolling_sharpe.png",
        "active_breadth.png",
        "cost_diagnostics.png",
        "schedule_comparison.png",
        "schedule_risk_return.png",
        "sma_sweep.png",
        "regime_comparison.png",
        "results_summary.csv",
        "schedule_comparison.csv",
        "sma_sweep.csv",
        "proxy_fidelity_report.csv",
        "trade_log.csv",
        "cost_attribution.csv",
        "run_manifest.json",
    ]
    for artifact in artifacts:
        lines.append(f"- `{artifact}`")

    report_path = out_dir / "detailed_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved detailed markdown report: %s", report_path)
    return report_path

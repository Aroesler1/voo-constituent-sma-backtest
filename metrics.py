"""Performance and risk metric calculations for constituent-level backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _infer_periods_per_year(index: pd.Index) -> float:
    """Infer annualization factor from date spacing."""
    if len(index) < 3:
        return 252.0

    dt_index = pd.DatetimeIndex(pd.to_datetime(index))
    median_days = float(dt_index.to_series().diff().dt.days.dropna().median())

    if median_days <= 2.0:
        return 252.0
    if median_days <= 8.0:
        return 52.0
    if median_days <= 18.0:
        return 24.0
    return 12.0


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return underwater equity series: (equity / running_max) - 1.

    Args:
        equity_curve: Equity curve indexed by date.

    Returns:
        Drawdown series with values <= 0.
    """
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0


def _max_drawdown_duration_periods(drawdown: pd.Series) -> int:
    """Compute longest consecutive drawdown duration in periods."""
    underwater = drawdown < 0
    if underwater.empty:
        return 0

    durations: list[int] = []
    cur = 0
    for flag in underwater.astype(bool):
        if flag:
            cur += 1
        else:
            if cur > 0:
                durations.append(cur)
            cur = 0

    if cur > 0:
        durations.append(cur)
    return int(max(durations) if durations else 0)


def _holding_period_lengths(positions: pd.Series) -> list[int]:
    """Extract consecutive in-market durations from a binary position series."""
    pos = positions.fillna(0).astype(int)
    lengths: list[int] = []
    cur = 0

    for value in pos:
        if value == 1:
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
            cur = 0

    if cur > 0:
        lengths.append(cur)
    return lengths


def _matched_round_trips(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Match BUY/SELL ticker lots FIFO and compute realized trade outcomes."""
    if trade_log.empty or "direction" not in trade_log.columns or "price" not in trade_log.columns:
        return pd.DataFrame(columns=["ticker", "entry_date", "exit_date", "return", "holding_days"])

    df = trade_log.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "ticker" not in df.columns:
        df["ticker"] = "ASSET"
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    lots: dict[str, list[dict[str, float | pd.Timestamp]]] = {}
    rows: list[dict[str, float | pd.Timestamp | str]] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        side = str(row["direction"]).upper()
        px = float(row["price"])
        dt = pd.Timestamp(row["date"])

        if not np.isfinite(px) or px <= 0:
            continue

        notional = float(pd.to_numeric(row.get("trade_notional", np.nan), errors="coerce"))
        if not np.isfinite(notional) or notional <= 0:
            notional = float(pd.to_numeric(row.get("equity_before", np.nan), errors="coerce"))
        qty = notional / px if np.isfinite(notional) and notional > 0 else 1.0
        if not np.isfinite(qty) or qty <= 0:
            qty = 1.0

        fifo = lots.setdefault(ticker, [])
        if side == "BUY":
            fifo.append({"entry_date": dt, "entry_price": px, "qty": qty})
            continue

        if side != "SELL":
            continue

        remaining = qty
        while remaining > 1e-12 and fifo:
            lot = fifo[0]
            matched = min(float(lot["qty"]), remaining)
            entry_price = float(lot["entry_price"])
            if entry_price > 0 and matched > 0:
                rows.append(
                    {
                        "ticker": ticker,
                        "entry_date": pd.Timestamp(lot["entry_date"]),
                        "exit_date": dt,
                        "return": px / entry_price - 1.0,
                        "holding_days": max((dt - pd.Timestamp(lot["entry_date"])).days, 0),
                        "qty": matched,
                    }
                )
            lot["qty"] = float(lot["qty"]) - matched
            remaining -= matched
            if float(lot["qty"]) <= 1e-12:
                fifo.pop(0)

    if not rows:
        return pd.DataFrame(columns=["ticker", "entry_date", "exit_date", "return", "holding_days", "qty"])
    return pd.DataFrame(rows)


def compute_metrics(
    equity_curve: pd.Series,
    weekly_returns: pd.Series,
    trade_log: pd.DataFrame,
    positions: pd.Series,
    cash_rate: float,
    active_count: pd.Series | None = None,
    eligible_count: pd.Series | None = None,
    exposure: pd.Series | None = None,
) -> dict[str, float]:
    """Compute portfolio performance, risk, trading, and breadth diagnostics.

    Args:
        equity_curve: Equity curve.
        weekly_returns: Period returns (name retained for compatibility).
        trade_log: Trade rows.
        positions: Binary in/out market indicator.
        cash_rate: Annual risk-free/cash rate.
        active_count: Optional count of active holdings by period.
        eligible_count: Optional count of eligible holdings by period.
        exposure: Optional gross invested fraction by period.

    Returns:
        Dictionary of metrics.
    """
    eq = equity_curve.dropna().astype(float)
    rets = weekly_returns.reindex(eq.index).dropna().astype(float)

    if len(eq) < 2:
        raise ValueError("Need at least two equity observations for metrics.")

    periods_per_year = _infer_periods_per_year(eq.index)
    total_periods = len(eq) - 1
    years = total_periods / periods_per_year if total_periods > 0 else np.nan

    initial = float(eq.iloc[0])
    final = float(eq.iloc[-1])

    cagr = (final / initial) ** (periods_per_year / total_periods) - 1.0 if total_periods > 0 and initial > 0 else np.nan
    annualized_vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year)) if len(rets) > 1 else np.nan
    sharpe = (cagr - cash_rate) / annualized_vol if pd.notna(annualized_vol) and annualized_vol > 0 else np.nan

    downside = rets[rets < 0]
    downside_dev = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else np.nan
    sortino = (cagr - cash_rate) / downside_dev if pd.notna(downside_dev) and downside_dev > 0 else np.nan

    dd = compute_drawdown_series(eq)
    max_drawdown = abs(float(dd.min())) if len(dd) else np.nan
    calmar = cagr / max_drawdown if pd.notna(cagr) and pd.notna(max_drawdown) and max_drawdown > 0 else np.nan

    matched_trades = _matched_round_trips(trade_log)
    rt = matched_trades["return"].tolist() if not matched_trades.empty else []
    wins = [x for x in rt if x > 0]
    losses = [x for x in rt if x < 0]

    hit_ratio = len(wins) / len(rt) if rt else np.nan
    avg_win = float(np.mean(wins)) if wins else np.nan
    avg_loss = float(np.mean(losses)) if losses else np.nan

    if not matched_trades.empty:
        avg_holding_period_weeks = float(matched_trades["holding_days"].mean() / 7.0)
    else:
        hold_lengths = _holding_period_lengths(positions)
        avg_holding_period_periods = float(np.mean(hold_lengths)) if hold_lengths else np.nan
        avg_holding_period_weeks = (
            avg_holding_period_periods * (52.0 / periods_per_year)
            if pd.notna(avg_holding_period_periods)
            else np.nan
        )

    total_trades = float(len(rt))

    if not trade_log.empty and "turnover_fraction" in trade_log.columns:
        tr = trade_log.copy()
        tr["date"] = pd.to_datetime(tr["date"])
        turnover_by_date = tr.groupby("date")["turnover_fraction"].sum()
        annual_turnover = float(turnover_by_date.sum() / years) if years and years > 0 else np.nan
    else:
        turn_events = positions.fillna(0).astype(float).diff().abs().fillna(0.0).sum()
        annual_turnover = float(turn_events / years) if years and years > 0 else np.nan

    pct_time_in_market = float((positions.fillna(0).astype(int) == 1).mean())

    mdd_periods = _max_drawdown_duration_periods(dd)
    max_drawdown_duration_weeks = float(mdd_periods * (52.0 / periods_per_year))

    avg_trade_cost_bps = np.nan
    total_cost_bps_annualized = np.nan
    avg_implementation_shortfall_bps = np.nan

    if not trade_log.empty:
        if "total_cost_bps" in trade_log.columns:
            avg_trade_cost_bps = float(pd.to_numeric(trade_log["total_cost_bps"], errors="coerce").mean())

        if "total_cost_usd" in trade_log.columns and years and years > 0:
            tr = trade_log.copy()
            tr["date"] = pd.to_datetime(tr["date"])
            daily_cost = tr.groupby("date")["total_cost_usd"].sum()
            daily_nav = None
            if "equity_before" in tr.columns:
                daily_nav = pd.to_numeric(
                    tr.groupby("date")["equity_before"].max(),
                    errors="coerce",
                ).replace(0.0, np.nan)
            if daily_nav is not None and len(daily_nav.dropna()) > 0:
                cost_rate = (daily_cost / daily_nav).replace([np.inf, -np.inf], np.nan).dropna()
                if len(cost_rate) > 0:
                    total_cost_bps_annualized = float(cost_rate.mean() * periods_per_year * 10000.0)
            elif initial > 0:
                total_cost_usd = float(pd.to_numeric(tr["total_cost_usd"], errors="coerce").sum())
                total_cost_bps_annualized = (total_cost_usd / initial) / years * 10000.0

        if "implementation_shortfall_bps" in trade_log.columns:
            avg_implementation_shortfall_bps = float(
                pd.to_numeric(trade_log["implementation_shortfall_bps"], errors="coerce").mean()
            )
        elif pd.notna(avg_trade_cost_bps):
            avg_implementation_shortfall_bps = float(avg_trade_cost_bps)

    avg_active_names = float(active_count.mean()) if active_count is not None and len(active_count) else np.nan
    active_breadth_pct = np.nan
    if active_count is not None and eligible_count is not None:
        denom = eligible_count.replace(0, np.nan)
        breadth = active_count / denom
        active_breadth_pct = float(breadth.mean()) if len(breadth.dropna()) else np.nan

    if exposure is not None and len(exposure):
        gross_exposure_mean = float(exposure.mean())
    else:
        gross_exposure_mean = float((positions.fillna(0).astype(int) == 1).mean())

    return {
        "cagr": float(cagr),
        "annualized_vol": float(annualized_vol) if pd.notna(annualized_vol) else np.nan,
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "sortino": float(sortino) if pd.notna(sortino) else np.nan,
        "max_drawdown": float(max_drawdown) if pd.notna(max_drawdown) else np.nan,
        "calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "hit_ratio": float(hit_ratio) if pd.notna(hit_ratio) else np.nan,
        "avg_win": float(avg_win) if pd.notna(avg_win) else np.nan,
        "avg_loss": float(avg_loss) if pd.notna(avg_loss) else np.nan,
        "avg_holding_period_weeks": float(avg_holding_period_weeks) if pd.notna(avg_holding_period_weeks) else np.nan,
        "total_trades": float(total_trades),
        "annual_turnover": float(annual_turnover) if pd.notna(annual_turnover) else np.nan,
        "pct_time_in_market": float(pct_time_in_market),
        "max_drawdown_duration_weeks": float(max_drawdown_duration_weeks),
        "avg_trade_cost_bps": float(avg_trade_cost_bps) if pd.notna(avg_trade_cost_bps) else np.nan,
        "total_cost_bps_annualized": float(total_cost_bps_annualized) if pd.notna(total_cost_bps_annualized) else np.nan,
        "avg_implementation_shortfall_bps": float(avg_implementation_shortfall_bps)
        if pd.notna(avg_implementation_shortfall_bps)
        else np.nan,
        "avg_active_names": float(avg_active_names) if pd.notna(avg_active_names) else np.nan,
        "active_breadth_pct": float(active_breadth_pct) if pd.notna(active_breadth_pct) else np.nan,
        "gross_exposure_mean": float(gross_exposure_mean) if pd.notna(gross_exposure_mean) else np.nan,
    }


def compute_rolling_sharpe(
    weekly_returns: pd.Series,
    window_weeks: int = 156,
    cash_rate: float = 0.02,
) -> pd.Series:
    """Compute rolling annualized Sharpe ratio.

    Args:
        weekly_returns: Period returns.
        window_weeks: Rolling window length in periods (name retained for compatibility).
        cash_rate: Annual cash rate.

    Returns:
        Rolling Sharpe series.
    """
    rets = weekly_returns.astype(float)
    periods_per_year = _infer_periods_per_year(rets.index)
    per_period_rf = (1.0 + cash_rate) ** (1.0 / periods_per_year) - 1.0

    excess = rets - per_period_rf
    mean_ann = excess.rolling(window_weeks).mean() * periods_per_year
    vol_ann = excess.rolling(window_weeks).std(ddof=1) * np.sqrt(periods_per_year)
    return mean_ann / vol_ann

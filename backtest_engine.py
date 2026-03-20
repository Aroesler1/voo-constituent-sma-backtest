"""Portfolio simulation engines for institutional constituent-level backtests."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import BacktestConfig

LOGGER = logging.getLogger(__name__)
EXTREME_PORTFOLIO_MOVE_THRESHOLD = 0.25


def _annual_to_period_cash_return(
    annual_yield: float,
    days: int,
    day_count: str,
    dynamic: bool,
) -> float:
    """Convert annualized cash yield to a period return.

    Args:
        annual_yield: Annualized yield in decimal form.
        days: Number of calendar days in period.
        day_count: Day count convention.
        dynamic: Whether annual_yield is market-implied (linear accrual) or flat fallback.

    Returns:
        Period cash return.
    """
    if days <= 0:
        return 0.0

    if dynamic:
        denom = 360.0 if day_count == "ACT/360" else 365.0
        return float(annual_yield) * (days / denom)

    return (1.0 + float(annual_yield)) ** (days / 365.0) - 1.0


def _build_cash_return_series(
    index: pd.DatetimeIndex,
    config: BacktestConfig,
    cash_curve: pd.Series | None,
) -> pd.Series:
    """Build period cash return series aligned to an index."""
    idx = pd.DatetimeIndex(index)
    day_steps = idx.to_series().diff().dt.days.fillna(1).clip(lower=1).astype(int)

    if cash_curve is not None and not cash_curve.empty:
        annual = cash_curve.reindex(idx).ffill().shift(1)
        annual = annual.fillna(config.CASH_RATE_ANNUAL)
        dynamic = True
    else:
        annual = pd.Series(config.CASH_RATE_ANNUAL, index=idx, dtype=float)
        dynamic = False

    out = pd.Series(index=idx, dtype=float)
    for dt in idx:
        out.loc[dt] = _annual_to_period_cash_return(
            annual_yield=float(annual.loc[dt]),
            days=int(day_steps.loc[dt]),
            day_count=config.CASH_RATE_DAY_COUNT,
            dynamic=dynamic,
        )
    return out


def _extract_feature_df(obj: Any, index: pd.DatetimeIndex, columns: list[str], default: float) -> pd.DataFrame:
    """Extract feature DataFrame from attrs or synthesize a constant matrix."""
    if isinstance(obj, pd.DataFrame):
        return obj.reindex(index=index, columns=columns)

    return pd.DataFrame(default, index=index, columns=columns, dtype=float)


def _trade_cost_components(
    *,
    trade_notional: float,
    modeled_impact_notional: float,
    price: float,
    adv_usd: float,
    sigma_20d: float,
    spread_bps_est: float,
    direction: str,
    use_open_execution: bool,
    config: BacktestConfig,
) -> dict[str, float]:
    """Compute one-way implementation cost components for a single trade leg."""
    if trade_notional <= 0:
        return {
            "arrival_gap_bps": 0.0,
            "half_spread_bps": 0.0,
            "impact_bps": 0.0,
            "explicit_fee_bps": 0.0,
            "commission_usd": 0.0,
            "commission_bps": 0.0,
            "regulatory_fee_usd": 0.0,
            "regulatory_fee_bps": 0.0,
            "total_cost_bps": 0.0,
            "total_cost_usd": 0.0,
            "participation": 0.0,
            "modeled_impact_notional": 0.0,
        }

    arrival_gap_bps = float(config.OPEN_AUCTION_SLIPPAGE_BPS if use_open_execution else config.SLIPPAGE_BPS)

    if config.ENABLE_ENHANCED_COST_MODEL:
        spread = float(spread_bps_est) if pd.notna(spread_bps_est) else np.nan
        if np.isnan(spread):
            spread = float(config.SPREAD_FLOOR_BPS)
        spread = max(spread, float(config.SPREAD_FLOOR_BPS))
        spread = min(spread, float(config.SPREAD_CAP_BPS))
        half_spread_bps = spread / 2.0

        adv = float(adv_usd) if pd.notna(adv_usd) and adv_usd > 0 else float(config.ASSUMED_ADV_USD_PER_NAME)
        adv = max(adv, 1.0)
        modeled_notional = max(float(modeled_impact_notional), 0.0)
        participation = min(modeled_notional / adv, 10.0)

        sigma = float(sigma_20d) if pd.notna(sigma_20d) and sigma_20d > 0 else 0.0
        if config.RETAIL_ACCOUNT_MODE and participation < float(config.MIN_PARTICIPATION_FOR_IMPACT):
            impact_bps = 0.0
        else:
            impact_bps = float(config.IMPACT_COEF) * (sigma * 10000.0) * np.sqrt(max(participation, 0.0))

        explicit_fee_bps = float(config.EXPLICIT_FEE_BPS)
        modeled_bps = arrival_gap_bps + half_spread_bps + impact_bps + explicit_fee_bps
    else:
        half_spread_bps = 0.0
        impact_bps = 0.0
        explicit_fee_bps = 0.0
        participation = 0.0
        modeled_bps = arrival_gap_bps

    px = float(price) if pd.notna(price) and price > 0 else 1.0
    shares = trade_notional / px
    if not config.USE_FRACTIONAL_SHARES:
        shares = float(np.floor(shares))
    commission_variable = float(config.COMMISSION_PER_SHARE) * shares
    commission_usd = float(config.COMMISSION_PER_TRADE) + max(float(config.MIN_COMMISSION_PER_ORDER), commission_variable)
    commission_bps = (commission_usd / trade_notional) * 10000.0 if trade_notional > 0 else 0.0

    regulatory_fee_usd = 0.0
    if config.INCLUDE_REGULATORY_FEES and str(direction).upper() == "SELL" and shares > 0:
        if px > float(config.FINRA_TAF_PER_SHARE):
            regulatory_fee_usd += min(float(config.FINRA_TAF_MAX_PER_TRADE), float(config.FINRA_TAF_PER_SHARE) * shares)
    regulatory_fee_bps = (regulatory_fee_usd / trade_notional) * 10000.0 if trade_notional > 0 else 0.0

    total_cost_bps = modeled_bps + commission_bps + regulatory_fee_bps
    total_cost_usd = trade_notional * modeled_bps / 10000.0 + commission_usd + regulatory_fee_usd

    return {
        "arrival_gap_bps": arrival_gap_bps,
        "half_spread_bps": half_spread_bps,
        "impact_bps": impact_bps,
        "explicit_fee_bps": explicit_fee_bps,
        "commission_usd": commission_usd,
        "commission_bps": commission_bps,
        "regulatory_fee_usd": regulatory_fee_usd,
        "regulatory_fee_bps": regulatory_fee_bps,
        "total_cost_bps": total_cost_bps,
        "total_cost_usd": total_cost_usd,
        "participation": participation,
        "modeled_impact_notional": max(float(modeled_impact_notional), 0.0),
    }


def run_constituent_backtest(
    price_df: pd.DataFrame,
    return_df: pd.DataFrame,
    membership_mask: pd.DataFrame,
    active_mask: pd.DataFrame,
    rebalance_calendar: pd.DatetimeIndex,
    config: BacktestConfig,
    cash_curve: pd.Series | None = None,
    store_weights: bool = True,
    store_cost_attribution: bool = True,
) -> dict[str, Any]:
    """Run constituent-level long/flat backtest with equal-weight active allocation.

    Args:
        price_df: Adjusted close matrix (date x ticker). Optional attrs may include
            ``open_df``, ``adv_usd``, ``sigma_20d``, ``spread_bps_est``.
        return_df: Simple close-to-close return matrix aligned to ``price_df``.
        membership_mask: Point-in-time membership boolean matrix.
        active_mask: Signal matrix ({1,0,NaN}) aligned to prices.
        rebalance_calendar: Execution dates for rebalance events.
        config: Runtime backtest config.
        cash_curve: Optional annualized cash yield series.
        store_weights: Whether to retain the full daily weights matrix.
        store_cost_attribution: Whether to retain per-trade cost attribution rows.

    Returns:
        Dictionary containing equity, returns, weights, exposure, and cost diagnostics.
    """
    if price_df.empty:
        raise ValueError("price_df is empty.")

    price_attrs = dict(price_df.attrs)
    price_df = price_df.copy(deep=False)
    price_df.attrs = {}
    return_df = return_df.copy(deep=False)
    return_df.attrs = {}
    membership_mask = membership_mask.copy(deep=False)
    membership_mask.attrs = {}
    active_mask = active_mask.copy(deep=False)
    active_mask.attrs = {}

    dates = pd.DatetimeIndex(price_df.index).sort_values().unique()
    common_cols = sorted(
        set(price_df.columns)
        & set(return_df.columns)
        & set(membership_mask.columns)
        & set(active_mask.columns)
    )
    if not common_cols:
        raise ValueError("No common ticker columns across price/return/membership/signal matrices.")

    close_px = price_df.reindex(index=dates, columns=common_cols).astype(float)
    returns_cc = return_df.reindex(index=dates, columns=common_cols).astype(float)
    membership = membership_mask.reindex(index=dates, columns=common_cols).fillna(False).astype(bool)
    active = active_mask.reindex(index=dates, columns=common_cols)

    open_attr = price_attrs.get("open_df")
    if isinstance(open_attr, pd.DataFrame):
        open_px = open_attr.reindex(index=dates, columns=common_cols).astype(float)
        use_open_execution = config.EXECUTION_TIMING == "next_open"
    else:
        open_px = close_px.copy()
        use_open_execution = False

    adv_df = _extract_feature_df(
        price_attrs.get("adv_usd"),
        dates,
        common_cols,
        float(config.ASSUMED_ADV_USD_PER_NAME),
    )
    sigma_df = _extract_feature_df(price_attrs.get("sigma_20d"), dates, common_cols, 0.0)
    spread_df = _extract_feature_df(price_attrs.get("spread_bps_est"), dates, common_cols, float(config.SPREAD_FLOOR_BPS))

    rebalance_set = set(pd.to_datetime(pd.DatetimeIndex(rebalance_calendar)).tolist())
    cash_returns = _build_cash_return_series(dates, config, cash_curve)

    n_assets = len(common_cols)
    w_close = np.zeros(n_assets, dtype=float)
    cash_w_close = 1.0

    equity_curve = pd.Series(np.nan, index=dates, dtype=float)
    equity_curve.iloc[0] = float(config.INITIAL_CAPITAL)

    period_returns = pd.Series(0.0, index=dates, dtype=float)
    implementation_shortfall_bps = pd.Series(0.0, index=dates, dtype=float)

    weights = pd.DataFrame(0.0, index=dates, columns=common_cols, dtype=float) if store_weights else None
    cash_weights = pd.Series(1.0, index=dates, dtype=float) if store_weights else None
    active_count = pd.Series(0, index=dates, dtype=int)
    eligible_count = pd.Series(0, index=dates, dtype=int)
    exposure = pd.Series(0.0, index=dates, dtype=float)

    trade_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] | None = [] if store_cost_attribution else None

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        date_i = dates[i]

        equity_prev_close = float(equity_curve.iloc[i - 1])
        if equity_prev_close <= 0:
            equity_curve.iloc[i] = 0.0
            period_returns.iloc[i] = 0.0
            if weights is not None:
                weights.iloc[i] = 0.0
            if cash_weights is not None:
                cash_weights.iloc[i] = 1.0
            exposure.iloc[i] = 0.0
            continue

        close_prev = close_px.loc[prev_date].to_numpy(dtype=float)
        open_i = open_px.loc[date_i].to_numpy(dtype=float)
        adv_prev = adv_df.loc[prev_date].to_numpy(dtype=float)

        if use_open_execution:
            with np.errstate(divide="ignore", invalid="ignore"):
                overnight = open_i / close_prev - 1.0
            overnight = np.where(np.isfinite(overnight), overnight, 0.0)
            asset_open_notional = equity_prev_close * w_close * (1.0 + overnight)
        else:
            asset_open_notional = equity_prev_close * w_close

        asset_open_notional = np.where(np.isfinite(asset_open_notional), asset_open_notional, 0.0)
        cash_open_notional = equity_prev_close * cash_w_close

        equity_open_before_cost = float(np.nansum(asset_open_notional) + cash_open_notional)

        member_prev = membership.loc[prev_date].to_numpy(dtype=bool)
        signal_prev = active.loc[prev_date].fillna(0.0).to_numpy(dtype=float) > 0.5

        tradable = member_prev & np.isfinite(close_prev)
        if use_open_execution:
            tradable &= np.isfinite(open_i) & (open_i > 0)
        if config.ENFORCE_INVESTABILITY_FILTER:
            tradable &= close_prev >= float(config.MIN_PRICE_TO_TRADE)
            tradable &= np.nan_to_num(adv_prev, nan=0.0) >= float(config.MIN_ADV_USD_TO_TRADE)

        desired_active = tradable & signal_prev
        active_count.iloc[i] = int(desired_active.sum())
        eligible_count.iloc[i] = int(tradable.sum())

        total_cost_usd = 0.0

        if date_i in rebalance_set:
            target_w = np.zeros(n_assets, dtype=float)
            if int(desired_active.sum()) >= int(config.MIN_ACTIVE_NAMES):
                target_w[desired_active] = 1.0 / float(desired_active.sum())

            if equity_open_before_cost > 0:
                current_w_open = asset_open_notional / equity_open_before_cost
            else:
                current_w_open = np.zeros(n_assets, dtype=float)

            delta_w = target_w - current_w_open
            if float(config.REBALANCE_BUFFER_BPS) > 0:
                buffer = float(config.REBALANCE_BUFFER_BPS) / 10000.0
                delta_w[np.abs(delta_w) < buffer] = 0.0

            if float(config.MIN_TRADE_NOTIONAL_USD) > 0 and equity_open_before_cost > 0:
                min_weight = float(config.MIN_TRADE_NOTIONAL_USD) / equity_open_before_cost
                delta_w[np.abs(delta_w) < min_weight] = 0.0

            target_w = np.clip(current_w_open + delta_w, 0.0, 1.0)
            target_cash_w = max(0.0, 1.0 - float(target_w.sum()))
            total_target = float(target_w.sum())
            if total_target > 1.0:
                target_w = target_w / total_target
                target_cash_w = 0.0

            turnover = np.abs(delta_w)

            exec_px = open_i if use_open_execution else close_prev
            sigma_prev = sigma_df.loc[prev_date].to_numpy(dtype=float)
            spread_prev = spread_df.loc[prev_date].to_numpy(dtype=float)

            for j in np.where(turnover > 1e-12)[0]:
                trade_notional = float(equity_open_before_cost * turnover[j])
                if trade_notional <= 0:
                    continue

                px = float(exec_px[j]) if pd.notna(exec_px[j]) and exec_px[j] > 0 else float(close_prev[j])
                if not np.isfinite(px) or px <= 0:
                    px = 1.0
                direction = "BUY" if delta_w[j] > 0 else "SELL"
                modeled_impact_notional = trade_notional
                if config.RETAIL_ACCOUNT_MODE:
                    modeled_nav = min(float(equity_open_before_cost), float(config.RETAIL_EXECUTION_ACCOUNT_SIZE_USD))
                    modeled_impact_notional = float(modeled_nav * turnover[j])

                costs = _trade_cost_components(
                    trade_notional=trade_notional,
                    modeled_impact_notional=modeled_impact_notional,
                    price=px,
                    adv_usd=float(adv_prev[j]),
                    sigma_20d=float(sigma_prev[j]),
                    spread_bps_est=float(spread_prev[j]),
                    direction=direction,
                    use_open_execution=use_open_execution,
                    config=config,
                )

                total_cost_usd += costs["total_cost_usd"]
                fill = px * (1.0 + costs["total_cost_bps"] / 10000.0) if direction == "BUY" else px * (1.0 - costs["total_cost_bps"] / 10000.0)

                trade_rows.append(
                    {
                        "date": date_i,
                        "ticker": common_cols[j],
                        "direction": direction,
                        "price": fill,
                        "slippage_cost": costs["total_cost_usd"],
                        "equity_before": equity_open_before_cost,
                        "equity_after": np.nan,
                        "trade_notional": trade_notional,
                        "modeled_trade_notional": costs["modeled_impact_notional"],
                        "delta_weight": float(delta_w[j]),
                        "turnover_fraction": float(turnover[j]),
                        "arrival_gap_bps": costs["arrival_gap_bps"],
                        "half_spread_bps": costs["half_spread_bps"],
                        "impact_bps": costs["impact_bps"],
                        "explicit_fee_bps": costs["explicit_fee_bps"],
                        "commission_usd": costs["commission_usd"],
                        "commission_bps": costs["commission_bps"],
                        "regulatory_fee_usd": costs["regulatory_fee_usd"],
                        "regulatory_fee_bps": costs["regulatory_fee_bps"],
                        "total_cost_bps": costs["total_cost_bps"],
                        "total_cost_usd": costs["total_cost_usd"],
                        "participation": costs["participation"],
                        "implementation_shortfall_bps": costs["total_cost_bps"],
                    }
                )

                if cost_rows is not None:
                    cost_rows.append(
                        {
                            "date": date_i,
                            "ticker": common_cols[j],
                            "direction": direction,
                            "trade_notional": trade_notional,
                            "turnover_fraction": float(turnover[j]),
                            **costs,
                        }
                    )

            equity_after_cost = max(equity_open_before_cost - total_cost_usd, 0.0)
            implementation_shortfall_bps.iloc[i] = (
                (total_cost_usd / equity_open_before_cost) * 10000.0 if equity_open_before_cost > 0 else 0.0
            )
            asset_post_open = equity_after_cost * target_w
            cash_post_open = equity_after_cost * target_cash_w
        else:
            asset_post_open = asset_open_notional
            cash_post_open = cash_open_notional

        if use_open_execution:
            close_i = close_px.loc[date_i].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                intraday = close_i / open_i - 1.0
            intraday = np.where(np.isfinite(intraday), intraday, 0.0)
            asset_close_notional = asset_post_open * (1.0 + intraday)
        else:
            cc_ret = returns_cc.loc[date_i].to_numpy(dtype=float)
            cc_ret = np.where(np.isfinite(cc_ret), cc_ret, 0.0)
            asset_close_notional = asset_post_open * (1.0 + cc_ret)

        asset_close_notional = np.where(np.isfinite(asset_close_notional), asset_close_notional, 0.0)

        cash_ret = float(cash_returns.loc[date_i])
        cash_close_notional = cash_post_open * (1.0 + cash_ret)

        equity_close = float(np.nansum(asset_close_notional) + cash_close_notional)
        equity_close = max(equity_close, 0.0)
        equity_curve.iloc[i] = equity_close

        if equity_prev_close > 0:
            period_returns.iloc[i] = equity_close / equity_prev_close - 1.0
        else:
            period_returns.iloc[i] = 0.0

        if abs(float(period_returns.iloc[i])) >= EXTREME_PORTFOLIO_MOVE_THRESHOLD:
            prev_asset_notional = equity_prev_close * w_close
            contrib = (asset_close_notional - prev_asset_notional) / equity_prev_close
            top_idx = np.argsort(np.abs(contrib))[::-1][:5]
            top_rows = ", ".join(
                f"{common_cols[j]}:{contrib[j]:+.4f}|w={w_close[j]:.4f}|r={((asset_close_notional[j] / prev_asset_notional[j]) - 1.0) if prev_asset_notional[j] > 0 else np.nan:+.4f}"
                for j in top_idx
                if np.isfinite(contrib[j]) and abs(float(contrib[j])) > 0
            )
            LOGGER.warning(
                "Extreme portfolio move on %s: return=%+.4f, top_contributors=%s",
                pd.Timestamp(date_i).date().isoformat(),
                float(period_returns.iloc[i]),
                top_rows or "n/a",
            )

        if equity_close > 0:
            w_close = asset_close_notional / equity_close
            cash_w_close = float(cash_close_notional / equity_close)
        else:
            w_close = np.zeros(n_assets, dtype=float)
            cash_w_close = 1.0

        if weights is not None:
            weights.iloc[i] = w_close
        if cash_weights is not None:
            cash_weights.iloc[i] = cash_w_close
        exposure.iloc[i] = float(np.clip(1.0 - cash_w_close, 0.0, 1.0))

        if total_cost_usd > 0 and trade_rows:
            for k in range(len(trade_rows) - 1, -1, -1):
                if pd.Timestamp(trade_rows[k]["date"]) != date_i:
                    break
                if pd.isna(trade_rows[k]["equity_after"]):
                    trade_rows[k]["equity_after"] = equity_open_before_cost - total_cost_usd

    if weights is not None:
        weights.iloc[0] = 0.0
    if cash_weights is not None:
        cash_weights.iloc[0] = 1.0
    exposure.iloc[0] = 0.0

    positions = (exposure > 0).astype("Int64")

    trade_log = pd.DataFrame(trade_rows)
    if not trade_log.empty:
        trade_log["date"] = pd.to_datetime(trade_log["date"])
        trade_log = trade_log.sort_values(["date", "ticker", "direction"]).reset_index(drop=True)

    cost_attribution = pd.DataFrame(cost_rows) if cost_rows is not None else pd.DataFrame()
    if not cost_attribution.empty:
        cost_attribution["date"] = pd.to_datetime(cost_attribution["date"])
        cost_attribution = cost_attribution.sort_values(["date", "ticker", "direction"]).reset_index(drop=True)

    LOGGER.info(
        "Constituent backtest complete: dates=%s, tickers=%s, trades=%s, final_equity=%.2f",
        len(dates),
        len(common_cols),
        len(trade_log),
        float(equity_curve.iloc[-1]),
    )

    return {
        "equity_curve": equity_curve,
        "trade_log": trade_log,
        "positions": positions,
        "weekly_returns": period_returns,
        "period_returns": period_returns,
        "weights": weights if weights is not None else pd.DataFrame(index=dates),
        "cash_weights": cash_weights if cash_weights is not None else pd.Series(index=dates, dtype=float),
        "active_count": active_count,
        "eligible_count": eligible_count,
        "exposure": exposure,
        "cost_attribution": cost_attribution,
        "implementation_shortfall_bps": implementation_shortfall_bps,
    }


def run_backtest(
    weekly_data: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig,
    cash_curve: pd.Series | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for legacy single-asset runs.

    Args:
        weekly_data: DataFrame with date and total_return_index/close columns.
        signals: Long/flat signal series indexed by date.
        config: Runtime config.
        cash_curve: Optional annualized cash yield series.

    Returns:
        Backtest dictionary in the same structure as ``run_constituent_backtest``.
    """
    if "date" in weekly_data.columns:
        df = weekly_data.copy().sort_values("date")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df = weekly_data.copy().sort_index()
        df.index = pd.to_datetime(df.index)

    if "total_return_index" in df.columns:
        close = df["total_return_index"].astype(float)
    elif "close" in df.columns:
        close = df["close"].astype(float)
    else:
        raise ValueError("weekly_data requires 'total_return_index' or 'close'.")

    price = pd.DataFrame({"ASSET": close})
    returns = price.pct_change().fillna(0.0)
    membership = pd.DataFrame(True, index=price.index, columns=price.columns)
    active = pd.DataFrame({"ASSET": signals.reindex(price.index).astype("Float64")})
    calendar = pd.DatetimeIndex(price.index[1:])

    if "open" in df.columns:
        price.attrs["open_df"] = pd.DataFrame({"ASSET": df["open"].astype(float).reindex(price.index)})

    return run_constituent_backtest(
        price_df=price,
        return_df=returns,
        membership_mask=membership,
        active_mask=active,
        rebalance_calendar=calendar,
        config=config,
        cash_curve=cash_curve,
    )


def run_buy_and_hold(
    weekly_data: pd.DataFrame | pd.Series,
    config: BacktestConfig,
    cash_curve: pd.Series | None = None,
) -> dict[str, Any]:
    """Run buy-and-hold benchmark.

    Args:
        weekly_data: Price series or DataFrame with ``date`` and ``close``/``total_return_index``.
        config: Runtime configuration.
        cash_curve: Unused for buy-and-hold; kept for interface compatibility.

    Returns:
        Backtest dictionary with benchmark equity and returns.
    """
    _ = cash_curve

    if isinstance(weekly_data, pd.Series):
        px = weekly_data.astype(float).dropna()
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()
    else:
        df = weekly_data.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
        else:
            df = df.sort_index()
            df.index = pd.to_datetime(df.index)

        if "total_return_index" in df.columns:
            px = df["total_return_index"].astype(float)
        elif "close" in df.columns:
            px = df["close"].astype(float)
        else:
            raise ValueError("weekly_data must contain total_return_index or close.")

    ret = px.pct_change().fillna(0.0)
    equity = (1.0 + ret).cumprod() * float(config.INITIAL_CAPITAL)

    positions = pd.Series(1, index=equity.index, dtype="Int64")
    weights = pd.DataFrame({"ASSET": 1.0}, index=equity.index)
    cash_weights = pd.Series(0.0, index=equity.index, dtype=float)
    exposure = pd.Series(1.0, index=equity.index, dtype=float)
    active_count = pd.Series(1, index=equity.index, dtype=int)
    eligible_count = pd.Series(1, index=equity.index, dtype=int)

    trade_log = pd.DataFrame(
        [
            {
                "date": equity.index[0],
                "ticker": "ASSET",
                "direction": "BUY",
                "price": float(px.iloc[0]),
                "slippage_cost": 0.0,
                "equity_before": float(config.INITIAL_CAPITAL),
                "equity_after": float(config.INITIAL_CAPITAL),
                "trade_notional": float(config.INITIAL_CAPITAL),
                "turnover_fraction": 1.0,
                "total_cost_usd": 0.0,
                "total_cost_bps": 0.0,
            }
        ]
    )

    cost_attribution = pd.DataFrame(
        [
            {
                "date": equity.index[0],
                "ticker": "ASSET",
                "direction": "BUY",
                "trade_notional": float(config.INITIAL_CAPITAL),
                "turnover_fraction": 1.0,
                "arrival_gap_bps": 0.0,
                "half_spread_bps": 0.0,
                "impact_bps": 0.0,
                "explicit_fee_bps": 0.0,
                "commission_usd": 0.0,
                "commission_bps": 0.0,
                "total_cost_bps": 0.0,
                "total_cost_usd": 0.0,
                "participation": 0.0,
            }
        ]
    )

    implementation_shortfall_bps = pd.Series(0.0, index=equity.index, dtype=float)

    return {
        "equity_curve": equity,
        "trade_log": trade_log,
        "positions": positions,
        "weekly_returns": ret,
        "period_returns": ret,
        "weights": weights,
        "cash_weights": cash_weights,
        "active_count": active_count,
        "eligible_count": eligible_count,
        "exposure": exposure,
        "cost_attribution": cost_attribution,
        "implementation_shortfall_bps": implementation_shortfall_bps,
    }

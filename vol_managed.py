"""Volatility-managed exposure, used here as a positive control.

Moreira & Muir, "Volatility-Managed Portfolios" (*The Journal of Finance* 72(4),
2017, 1611-1644, doi:10.1111/jofi.12513), scale a portfolio by the inverse of
its own recent realised variance and report that this "produce[s] large alphas,
increase[s] Sharpe ratios". The rule is a timing rule with no cross-sectional
component and almost no turnover, so it is the natural control for a repository
whose headline result is that a moving-average timing rule loses to the index:
if the pipeline cannot reproduce a timing effect that the literature says is
there, the negative headline is uninformative about timing and informative only
about this code.

It is a control, not an endorsement. Cederburg, O'Doherty, Wang & Yan, "On the
performance of volatility-managed portfolios" (*Journal of Financial Economics*
138(1), 2020, 95-117), find the strategy does not survive real-time
implementation for most factors, and this repository's own result is reported
against that backdrop rather than around it.

Implementation follows the paper: the overlay scales the *return series* rather
than reallocating inside the portfolio, so it composes with the constituent
backtest without changing the engine.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_WINDOW = 21
DEFAULT_CAP = 1.5


def realized_variance(returns: pd.Series, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Trailing realised variance of daily returns.

    Realised variance in the Moreira-Muir sense is the mean squared return over
    the window, not the demeaned sample variance: at a 21-day horizon the mean
    return is noise, and subtracting it adds estimation error without removing
    bias.

    Args:
        returns: Daily simple returns.
        window: Lookback in trading days.

    Returns:
        Variance series with NaN over the warm-up.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    r = returns.astype(float)
    return r.pow(2).rolling(window=window, min_periods=window).mean()


def calibrate_c(
    returns: pd.Series,
    *,
    train_end: pd.Timestamp,
    window: int = DEFAULT_WINDOW,
) -> float:
    """Choose the constant that makes the average uncapped weight one in training.

    The weight is ``c / RV``, so setting ``c = 1 / mean(1 / RV)`` over the
    training half makes the *uncapped* average weight exactly one there. The cap
    is applied afterwards, which pulls the realised average slightly below one;
    ``vol_managed_weights`` reports the realised average so this is visible
    rather than assumed away.

    Args:
        returns: Daily simple returns over the full sample.
        train_end: Last date of the training half, inclusive.
        window: Realised-variance lookback.

    Returns:
        The scaling constant.
    """
    rv = realized_variance(returns, window=window).shift(1)
    train = rv.loc[rv.index <= pd.Timestamp(train_end)].dropna()
    train = train[train > 0]
    if train.empty:
        raise ValueError("No usable training observations for the vol-managed constant.")
    return float(1.0 / (1.0 / train).mean())


def vol_managed_weights(
    returns: pd.Series,
    *,
    c: float,
    window: int = DEFAULT_WINDOW,
    cap: float = DEFAULT_CAP,
    update: str = "daily",
) -> pd.Series:
    """Implementable exposure weights: ``clip(c / RV_{t-1}, 0, cap)``.

    The variance is lagged one day so the weight applied to date *t* uses only
    returns observable through *t-1*. Warm-up dates, where no variance exists
    yet, get a weight of one (or the cap, if it is below one): the control must
    not get credit for sitting out the start of the sample.

    ``update`` sets how often the weight is actually traded. Moreira and Muir
    scale monthly returns, so ``"monthly"`` reproduces the paper: the weight is
    set on the first trading day of each month and held. ``"daily"`` retrades
    every session, which tracks the variance more closely and costs about
    twenty times as much in turnover. Both are reported because the difference
    between them is the whole question of whether the effect survives costs.

    Args:
        returns: Daily simple returns.
        c: Scaling constant from :func:`calibrate_c`.
        window: Realised-variance lookback.
        cap: Maximum exposure. 1.5 allows 50% leverage.
        update: ``"daily"`` or ``"monthly"``.

    Returns:
        Weight series aligned to ``returns``.
    """
    if cap <= 0:
        raise ValueError("cap must be positive.")
    if update not in {"daily", "monthly"}:
        raise ValueError("update must be 'daily' or 'monthly'.")

    rv = realized_variance(returns, window=window).shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = float(c) / rv
    weights = raw.clip(lower=0.0, upper=float(cap))

    if update == "monthly":
        idx = pd.DatetimeIndex(weights.index)
        month = idx.year * 100 + idx.month
        is_month_start = np.r_[True, month[1:] != month[:-1]]
        weights = weights.where(pd.Series(is_month_start, index=idx)).ffill()
        rebalance_mask = pd.Series(is_month_start, index=idx, dtype=bool)
    else:
        rebalance_mask = pd.Series(True, index=weights.index, dtype=bool)

    # Warm-up dates have no variance estimate yet. Fill them fully invested so
    # the control gets no credit for sitting out the start of the sample, but
    # never above the cap.
    weights = weights.fillna(min(1.0, float(cap)))
    weights.attrs["rebalance_mask"] = rebalance_mask
    return weights


def cash_returns_from_annual_yield(annual_yield: pd.Series) -> pd.Series:
    """Convert annual yields to interval returns using elapsed calendar days."""
    annual = annual_yield.astype(float)
    if (annual <= -1.0).any():
        raise ValueError("Annual cash yields must be greater than -100%.")
    idx = pd.DatetimeIndex(annual.index)
    elapsed = idx.to_series().diff().dt.days.fillna(1.0).clip(lower=1.0)
    return pd.Series(
        np.power(1.0 + annual.to_numpy(), elapsed.to_numpy() / 365.25) - 1.0,
        index=idx,
        dtype=float,
        name="cash_return",
    )


def _simulate_holdings(
    base: pd.Series,
    signal_weights: pd.Series,
    cash: pd.Series,
    *,
    rebalance_mask: pd.Series,
    cost_bps: float,
) -> pd.DataFrame:
    """Trade from drifted holdings, then evolve risky and cash sleeves."""
    rows: list[dict[str, float]] = []
    end_weight = 0.0
    rate = float(cost_bps) / 10_000.0

    for i, dt in enumerate(base.index):
        pretrade = end_weight
        rebalance = bool(rebalance_mask.loc[dt]) or i == 0
        target = float(signal_weights.loc[dt]) if rebalance else pretrade
        turnover = abs(target - pretrade)
        cost = turnover * rate
        risky_growth = target * (1.0 + float(base.loc[dt]))
        cash_growth = (1.0 - target) * (1.0 + float(cash.loc[dt]))
        gross_growth = risky_growth + cash_growth
        net_growth = gross_growth - cost
        if not np.isfinite(net_growth) or net_growth <= 0.0:
            raise ValueError("Vol-managed holdings produced nonpositive portfolio wealth.")
        gross_return = gross_growth - 1.0
        net_return = net_growth - 1.0
        end_weight = risky_growth / net_growth
        rows.append(
            {
                "signal_weight": float(signal_weights.loc[dt]),
                "pretrade_weight": pretrade,
                "weight": target,
                "turnover": turnover,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": net_return,
                "end_weight": end_weight,
            }
        )
    return pd.DataFrame(rows, index=base.index)


def apply_vol_management(
    base_returns: pd.Series,
    *,
    c: float,
    cash_returns: pd.Series | None = None,
    window: int = DEFAULT_WINDOW,
    cap: float = DEFAULT_CAP,
    update: str = "daily",
    cost_bps: float = 0.0,
    initial_capital: float = 1_000_000.0,
) -> dict[str, object]:
    """Overlay volatility management on a return series and charge its turnover.

    The unfunded part of the position earns (or, when levered, pays) the cash
    rate. Borrowing at the cash rate is an assumption favourable to the overlay:
    a retail account pays more than the three-month bill. The 1.5 cap bounds how
    much that assumption can flatter the result.

    Args:
        base_returns: Daily returns of the underlying portfolio, net of its own
            trading costs.
        c: Scaling constant from :func:`calibrate_c`.
        cash_returns: Daily cash returns; zero if omitted.
        window: Realised-variance lookback.
        cap: Maximum exposure.
        update: Weight-update frequency, ``"daily"`` or ``"monthly"``.
        cost_bps: One-way cost charged on the change in exposure, in bps of NAV.
        initial_capital: Starting equity for the returned curve.

    Returns:
        Dict with the weight series, gross and net return series, the equity
        curve, realised average weight, and annualised overlay turnover.
    """
    if cost_bps < 0:
        raise ValueError("cost_bps must be nonnegative.")
    base = base_returns.astype(float).fillna(0.0)
    signal_weights = vol_managed_weights(
        base, c=c, window=window, cap=cap, update=update
    )
    rebalance_mask = signal_weights.attrs["rebalance_mask"]

    if cash_returns is None:
        cash = pd.Series(0.0, index=base.index, dtype=float)
    else:
        cash = cash_returns.reindex(base.index).astype(float).fillna(0.0)

    holdings = _simulate_holdings(
        base,
        signal_weights,
        cash,
        rebalance_mask=rebalance_mask,
        cost_bps=cost_bps,
    )
    weights = holdings["weight"].rename("weight")
    gross = holdings["gross_return"].rename("gross_return")
    cost = holdings["cost"].rename("cost")
    net = holdings["net_return"].rename("net_return")
    turnover = holdings["turnover"].rename("turnover")

    equity = (1.0 + net).cumprod() * float(initial_capital)
    gross_equity = (1.0 + gross).cumprod() * float(initial_capital)
    elapsed_days = max((base.index[-1] - base.index[0]).days + 1, 1)
    years = elapsed_days / 365.25

    return {
        "weights": weights,
        "signal_weights": signal_weights,
        "pretrade_weights": holdings["pretrade_weight"],
        "end_weights": holdings["end_weight"],
        "turnover": turnover,
        "gross_returns": gross,
        "returns": net,
        "equity_curve": equity,
        "gross_equity_curve": gross_equity,
        "cost_drag": cost,
        "avg_weight": float(weights.mean()),
        "max_weight": float(weights.max()),
        "pct_time_levered": float((weights > 1.0).mean()),
        "annual_turnover": float(turnover.sum() / years),
    }

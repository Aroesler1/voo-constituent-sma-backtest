"""Point-in-time constituent panel construction.

Everything an entrypoint needs before the first backtest runs: the universe,
the CRSP price and return matrices, liquidity features, the default signal
mask and the cash curve. Extracted from ``main.py`` so that the timing-luck,
index-versus-stock, volatility-managed and CRSP-tape-comparison studies build
the identical panel rather than each reimplementing it.

The two price hooks (``constituent_fetcher``, ``benchmark_fetcher``) exist so a
different CRSP tape can be substituted without copying any of the assembly; see
``crsp_v2.py``.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from config import BacktestConfig
from data_loader import (
    build_snapshot_manifest_row,
    fetch_crsp_batch_prices,
    fetch_fred_cash_rate,
    fetch_sec_voo_holdings_proxy,
    fetch_sp500_membership_history_public,
)
from preprocessing import (
    build_point_in_time_constituent_universe,
    compute_daily_liquidity_feature_matrices,
)
from strategy import compute_sma_matrix, generate_active_mask

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestPanel:
    """Everything downstream of data loading and upstream of the first backtest."""

    trading_index: pd.DatetimeIndex
    close_df: pd.DataFrame
    close_returns: pd.DataFrame
    membership: pd.DataFrame
    base_weights: pd.DataFrame
    active_mask: pd.DataFrame
    sma_length: int
    valid_cols: list[str]
    cash_curve: pd.Series | None
    effective_cash_rate: float
    spy_close: pd.Series
    voo_close: pd.Series
    extended_voo_proxy: pd.Series
    coverage_ratio: pd.Series
    source_by_date: pd.Series
    provider_mix: dict[str, int]
    constituentsnapshot_rows: list[dict[str, Any]] = field(default_factory=list)
    voo_daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    spy_daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    voo_source: str = ""
    spy_source: str = ""
    sec_holdings: pd.DataFrame = field(default_factory=pd.DataFrame)
    sp500_events: pd.DataFrame = field(default_factory=pd.DataFrame)


def _normalize_ticker(value: str) -> str:
    return str(value).strip().upper().replace(".", "-")


def _to_date_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.sort_values("date").set_index("date")


def _extract_crsp_total_return_close(d: pd.DataFrame) -> pd.Series:
    """Construct a CRSP total-return price path from raw close and return fields.

    Vectorised. The previous implementation walked rows in Python with scalar
    .loc lookups, costing ~59us per row; across the universe that is ~8 minutes
    per call, and extract_adjusted_series calls it once per requested field, so
    a full run spent roughly three quarters of an hour inside this function
    alone. The cumulative product is now computed per segment with groupby.

    A segment break is a PERMNO change or a gap of more than seven days. Both
    reset the compounding, because a price path may not be carried across two
    different securities or across a listing gap.
    """
    raw_close = pd.to_numeric(d["close"], errors="coerce").abs()
    total_return = pd.to_numeric(d.get("total_return"), errors="coerce")
    if "permno" in d.columns:
        permno = pd.to_numeric(d["permno"], errors="coerce")
    else:
        permno = pd.Series(np.nan, index=d.index)

    gap_days = d.index.to_series().diff().dt.days
    # first row is always a break; a permno change or a >7d gap resets the path
    segment_break = (
        gap_days.isna()
        | (gap_days > 7)
        | ((permno != permno.shift()) & permno.notna() & permno.shift().notna())
    )
    segment = segment_break.cumsum()

    # missing vendor return falls back to the raw price change, then to zero,
    # matching the original row-wise precedence
    raw_ret = raw_close / raw_close.shift() - 1.0
    ret = total_return.fillna(raw_ret).fillna(0.0)

    # at a break the growth factor is 1 so the segment starts exactly at `base`
    growth = (1.0 + ret).where(~segment_break, 1.0)
    cumulative = growth.groupby(segment).cumprod()

    base = raw_close.where(segment_break).groupby(segment).transform("first")
    base = base.where(base.notna() & (base > 0), 1.0)

    return (base * cumulative).astype(float)


def extract_adjusted_series(df: pd.DataFrame, field: str) -> pd.Series:
    """Extract adjusted OHLCV series from normalized vendor OHLCV frames."""
    d = _to_date_index(df)
    is_crsp = "total_return" in d.columns and "permno" in d.columns

    if is_crsp:
        synthetic_close = _extract_crsp_total_return_close(d)
        legacy_adj_close = (
            pd.to_numeric(d["adjusted_close"], errors="coerce")
            if "adjusted_close" in d.columns
            else synthetic_close
        )
        scale = (synthetic_close / legacy_adj_close).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        if field == "close":
            return synthetic_close
        if field == "volume" and "volume" in d.columns:
            return pd.to_numeric(d["volume"], errors="coerce")
        if field == "open" and "open" in d.columns:
            return pd.to_numeric(d["open"], errors="coerce") * scale
        if field == "high" and "high" in d.columns:
            return pd.to_numeric(d["high"], errors="coerce") * scale
        if field == "low" and "low" in d.columns:
            return pd.to_numeric(d["low"], errors="coerce") * scale

    if field == "close":
        if "adjClose" in d.columns:
            close = pd.to_numeric(d["adjClose"], errors="coerce")
            if "adjusted_close" in d.columns:
                close = close.fillna(pd.to_numeric(d["adjusted_close"], errors="coerce"))
            elif "close" in d.columns:
                close = close.fillna(pd.to_numeric(d["close"], errors="coerce"))
            return close
        if "adjusted_close" in d.columns:
            return d["adjusted_close"].astype(float)
        if "close" in d.columns:
            return d["close"].astype(float)

    if field == "volume":
        if "adjVolume" in d.columns:
            vol = pd.to_numeric(d["adjVolume"], errors="coerce")
            if "volume" in d.columns:
                vol = vol.fillna(pd.to_numeric(d["volume"], errors="coerce"))
            return vol
        if "volume" in d.columns:
            return d["volume"].astype(float)

    # Adjust raw O/H/L using close adjustment factor when adjusted intraday fields are missing.
    if {"adjusted_close", "close"}.issubset(d.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = d["adjusted_close"].astype(float) / d["close"].astype(float)
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    elif {"adjClose", "close"}.issubset(d.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = d["adjClose"].astype(float) / d["close"].astype(float)
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        factor = pd.Series(1.0, index=d.index)

    if field == "open":
        if "adjOpen" in d.columns:
            adj_open = pd.to_numeric(d["adjOpen"], errors="coerce")
            if "open" in d.columns:
                adj_open = adj_open.fillna(pd.to_numeric(d["open"], errors="coerce") * factor)
            return adj_open
        if "open" in d.columns:
            return d["open"].astype(float) * factor

    if field == "high":
        if "adjHigh" in d.columns:
            adj_high = pd.to_numeric(d["adjHigh"], errors="coerce")
            if "high" in d.columns:
                adj_high = adj_high.fillna(pd.to_numeric(d["high"], errors="coerce") * factor)
            return adj_high
        if "high" in d.columns:
            return d["high"].astype(float) * factor

    if field == "low":
        if "adjLow" in d.columns:
            adj_low = pd.to_numeric(d["adjLow"], errors="coerce")
            if "low" in d.columns:
                adj_low = adj_low.fillna(pd.to_numeric(d["low"], errors="coerce") * factor)
            return adj_low
        if "low" in d.columns:
            return d["low"].astype(float) * factor

    raise ValueError(f"Cannot extract adjusted series '{field}' from columns {list(df.columns)}")


def _build_matrix(
    price_map: dict[str, pd.DataFrame],
    tickers: list[str],
    index: pd.DatetimeIndex,
    field: str,
) -> pd.DataFrame:
    """Build date x ticker matrix for one adjusted field."""
    mat = pd.DataFrame(index=index, columns=tickers, dtype=np.float32)
    for ticker in tickers:
        df = price_map.get(ticker)
        if df is None or df.empty:
            continue
        try:
            s = extract_adjusted_series(df, field)
            mat[ticker] = s.reindex(index).astype(np.float32)
        except Exception as exc:
            LOGGER.debug("Failed to extract %s for %s: %s", field, ticker, exc)
    return mat


def _build_return_matrix(
    price_map: dict[str, pd.DataFrame],
    tickers: list[str],
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build date x ticker simple-return matrix using vendor total return when available."""
    mat = pd.DataFrame(index=index, columns=tickers, dtype=float)
    for ticker in tickers:
        df = price_map.get(ticker)
        if df is None or df.empty:
            continue
        d = _to_date_index(df)
        close = extract_adjusted_series(df, "close")
        series = close.pct_change(fill_method=None)
        gap_days = d.index.to_series().diff().dt.days
        contiguous = gap_days.fillna(1).le(7)
        series = series.where(contiguous, 0.0)
        if "close" in d.columns:
            raw_close = pd.to_numeric(d["close"], errors="coerce")
            raw_ret = raw_close.pct_change(fill_method=None)
            with np.errstate(divide="ignore", invalid="ignore"):
                adj_factor = close / raw_close
            adj_factor = adj_factor.replace([np.inf, -np.inf], np.nan)
            factor_jump = adj_factor.pct_change(fill_method=None).abs().gt(0.25)
            suspicious_adjustment = factor_jump & raw_ret.abs().le(0.50) & series.abs().gt(0.50)
            series = series.mask(suspicious_adjustment, raw_ret)
        if "total_return" in d.columns:
            total_return = pd.to_numeric(d["total_return"], errors="coerce")
            fallback = series.copy()
            if "permno" in d.columns:
                permno = pd.to_numeric(d["permno"], errors="coerce")
                fallback = fallback.where(permno.eq(permno.shift(1)), 0.0)
            if "source_vendor" in d.columns:
                source_vendor = d["source_vendor"].astype(str).str.lower()
                fallback = fallback.where(source_vendor.eq(source_vendor.shift(1)), 0.0)
            series = pd.Series(
                np.where(total_return.notna(), total_return.to_numpy(dtype=float, copy=False), fallback.to_numpy(dtype=float, copy=False)),
                index=d.index,
                dtype=float,
            )
        mat[ticker] = series.reindex(index)
    return mat.fillna(0.0)


def _log_return_sanity(
    return_df: pd.DataFrame,
    membership: pd.DataFrame,
) -> None:
    """Log extreme constituent return diagnostics for PIT-eligible names."""
    aligned = return_df.reindex(index=membership.index, columns=membership.columns)
    eligible_returns = aligned.where(membership.fillna(False))
    abs_returns = eligible_returns.abs()
    if abs_returns.empty:
        return

    extreme_mask = abs_returns > 0.50
    extreme_count = int(extreme_mask.sum().sum())
    total_obs = int(eligible_returns.notna().sum().sum())
    max_abs = float(abs_returns.max().max()) if total_obs > 0 else np.nan
    LOGGER.info(
        "Return sanity: eligible_obs=%s, >50%% moves=%s, max_abs_return=%.4f",
        total_obs,
        extreme_count,
        max_abs if pd.notna(max_abs) else np.nan,
    )
    if extreme_count <= 0:
        return

    stacked = abs_returns.where(extreme_mask).stack().sort_values(ascending=False).head(10)
    for (dt, ticker), value in stacked.items():
        LOGGER.warning("Extreme eligible return: %s %s abs_return=%.4f", dt.date().isoformat(), ticker, float(value))


def _sanitize_open_matrix(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
) -> pd.DataFrame:
    """Repair clearly invalid adjusted opens caused by vendor anomalies.

    This is limited to impossible OHLC relationships or extreme open-only gaps
    that are not corroborated by the same day's close.
    """
    out = open_df.copy()
    hi = high_df.reindex_like(out)
    lo = low_df.reindex_like(out)
    prev_close = close_df.shift(1).reindex_like(out)
    close_now = close_df.reindex_like(out)

    invalid_bounds = out.notna() & (
        (hi.notna() & out.gt(hi)) |
        (lo.notna() & out.lt(lo))
    )
    if bool(invalid_bounds.to_numpy(dtype=bool).any()):
        clipped = out.where(~hi.notna(), np.minimum(out, hi))
        clipped = clipped.where(~lo.notna(), np.maximum(clipped, lo))
        out = out.where(~invalid_bounds, clipped)
        LOGGER.warning("Adjusted open QA clipped %s impossible OHLC rows.", int(invalid_bounds.sum().sum()))

    with np.errstate(divide="ignore", invalid="ignore"):
        overnight_gap = out / prev_close - 1.0
        same_day_move = close_now / prev_close - 1.0

    suspicious_gap = (
        out.notna()
        & prev_close.notna()
        & overnight_gap.abs().gt(0.75)
        & same_day_move.abs().lt(0.35)
    )
    if bool(suspicious_gap.to_numpy(dtype=bool).any()):
        replacement = prev_close.where(prev_close.notna(), close_now)
        out = out.where(~suspicious_gap, replacement)
        LOGGER.warning("Adjusted open QA replaced %s suspicious open-only gaps.", int(suspicious_gap.sum().sum()))

    return out


def _max_available_date(df: pd.DataFrame) -> pd.Timestamp | None:
    """Return the max date in a normalized price frame."""
    if df.empty or "date" not in df.columns:
        return None
    return pd.to_datetime(df["date"]).max()


def _needs_recent_tail(df: pd.DataFrame, requested_end: str, tolerance_days: int = 7) -> bool:
    """Return whether a frame is materially stale relative to the requested end date."""
    max_date = _max_available_date(df)
    if max_date is None:
        return True
    return max_date < (pd.Timestamp(requested_end) - pd.Timedelta(days=tolerance_days))


def _fetch_benchmark_series(
    ticker: str,
    start: str,
    end: str,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, str]:
    """Fetch one benchmark series from CRSP.

    Benchmark tickers are ETFs, so this path passes include_funds=True; the
    constituent path leaves funds excluded. There is no vendor fallback: CRSP is
    the only price source, and a missing benchmark is a hard failure rather than
    something to paper over with a second-best series.
    """
    if config.PRIMARY_PRICE_SOURCE != "crsp" or not config.has_crsp_credentials():
        raise ValueError(
            f"Cannot fetch benchmark {ticker}: CRSP is the only price source and "
            "its credentials are not configured."
        )
    crsp_map = fetch_crsp_batch_prices(
        tickers=[ticker],
        start=start,
        end=end,
        username=config.WRDS_USERNAME or config.CRSP_USERNAME,
        password=config.WRDS_PASSWORD,
        api_key=config.CRSP_API_KEY,
        include_funds=True,
    )
    df = crsp_map.get(ticker)
    if df is None or df.empty:
        raise ValueError(f"CRSP returned no rows for benchmark {ticker}.")
    if _needs_recent_tail(df, end):
        LOGGER.warning(
            "CRSP coverage for %s ends at %s, before the requested end %s; "
            "the effective sample is shorter than configured.",
            ticker,
            _max_available_date(df).date().isoformat(),
            end,
        )
    return df, "crsp"


def _fetch_constituent_prices(
    tickers: list[str],
    start: str,
    end: str,
    config: BacktestConfig,
    current_tickers: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Fetch constituent prices from CRSP.

    CRSP is the only price source. Tickers it cannot resolve are reported and
    left out rather than backfilled from a second vendor, so the universe stays
    a single consistent PERMNO-keyed panel and the coverage gate downstream sees
    the true shortfall.
    """
    requested = sorted({_normalize_ticker(t) for t in tickers if str(t).strip()})
    if config.PRIMARY_PRICE_SOURCE != "crsp" or not config.has_crsp_credentials():
        raise ValueError("CRSP is the only price source and its credentials are not configured.")

    crsp_prices = fetch_crsp_batch_prices(
        tickers=requested,
        start=start,
        end=end,
        username=config.WRDS_USERNAME or config.CRSP_USERNAME,
        password=config.WRDS_PASSWORD,
        api_key=config.CRSP_API_KEY,
    )
    prices = dict(crsp_prices)
    source_map = {ticker: "crsp" for ticker in crsp_prices}
    LOGGER.info("CRSP resolved %s/%s tickers.", len(prices), len(requested))

    unresolved = [ticker for ticker in requested if ticker not in prices]
    if unresolved:
        LOGGER.warning(
            "%s tickers did not resolve through CRSP and are excluded: %s",
            len(unresolved),
            ", ".join(unresolved[:20]) + ("..." if len(unresolved) > 20 else ""),
        )
    # CRSP coverage ends before the configured END_DATE, so current constituents
    # look stale. This used to trigger a vendor tail extension; it is now
    # reported, because the honest statement is that the sample ends early.
    stale = [
        ticker for ticker, df in prices.items()
        if current_tickers and ticker in current_tickers and _needs_recent_tail(df, end)
    ]
    if stale:
        LOGGER.warning(
            "%s current constituents have CRSP history ending before %s; "
            "the effective sample end is earlier than configured.",
            len(stale), end,
        )

    return prices, source_map


def _build_daily_cash_curve(
    config: BacktestConfig,
    trading_index: pd.DatetimeIndex,
) -> pd.Series | None:
    """Build daily time-varying cash yield curve (annualized decimal)."""
    if not config.USE_DYNAMIC_CASH_RATE:
        return None

    if not config.FRED_API_KEY:
        LOGGER.warning("USE_DYNAMIC_CASH_RATE=True but FRED_API_KEY missing; fallback to flat cash rate.")
        return None

    start = (trading_index.min() - pd.Timedelta(days=30)).date().isoformat()
    end = trading_index.max().date().isoformat()

    daily_curve: pd.DataFrame | None = None
    primary_error: Exception | None = None

    try:
        daily_curve = fetch_fred_cash_rate(
            start=start,
            end=end,
            series_id=config.CASH_RATE_SOURCE,
            api_key=config.FRED_API_KEY,
            as_of_date=config.CASH_RATE_AS_OF_DATE,
        )
    except Exception as exc:
        primary_error = exc
        LOGGER.warning("Primary cash source fetch failed (%s): %s", config.CASH_RATE_SOURCE, exc)

    if daily_curve is None and config.CASH_RATE_FALLBACK_SOURCE:
        try:
            daily_curve = fetch_fred_cash_rate(
                start=start,
                end=end,
                series_id=config.CASH_RATE_FALLBACK_SOURCE,
                api_key=config.FRED_API_KEY,
                as_of_date=config.CASH_RATE_AS_OF_DATE,
            )
            LOGGER.warning("Using fallback cash source: %s", config.CASH_RATE_FALLBACK_SOURCE)
        except Exception as exc:
            LOGGER.warning("Fallback cash source fetch failed (%s): %s", config.CASH_RATE_FALLBACK_SOURCE, exc)
            if primary_error is not None:
                LOGGER.debug("Primary cash source error detail: %s", primary_error, exc_info=True)
            return None

    if daily_curve is None or daily_curve.empty:
        return None

    curve = daily_curve.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.set_index("date").sort_index()

    out = curve["annual_yield"].reindex(trading_index).ffill().bfill()
    out.attrs["snapshot_id"] = daily_curve.attrs.get("snapshot_id")
    out.attrs["fetched_at_utc"] = daily_curve.attrs.get("fetched_at_utc")
    out.attrs["sha256"] = daily_curve.attrs.get("sha256")
    out.attrs["source_series"] = (
        curve["source_series"].iloc[-1] if "source_series" in curve.columns else config.CASH_RATE_SOURCE
    )
    out.attrs["requested_start"] = start
    out.attrs["requested_end"] = end

    LOGGER.info(
        "Daily cash curve ready: rows=%s, source=%s, mean=%.4f",
        len(out),
        out.attrs.get("source_series", config.CASH_RATE_SOURCE),
        float(out.mean()),
    )
    return out


def snapshot_row(vendor: str, ticker: str, df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    """Create one snapshot manifest row from a fetched DataFrame."""
    return build_snapshot_manifest_row(
        vendor=vendor,
        ticker=ticker,
        snapshot_id=df.attrs.get("snapshot_id"),
        fetched_at_utc=df.attrs.get("fetched_at_utc"),
        sha256=df.attrs.get("sha256"),
        requested_start=start,
        requested_end=end,
    )


def _first_universe_date(sp500_events: pd.DataFrame) -> pd.Timestamp:
    """Return the first usable date supported by the PIT membership proxy."""
    date_col = None
    for candidate in ("start_date", "date", "effective_date"):
        if candidate in sp500_events.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError("sp500_events must contain a usable date column.")

    starts = pd.to_datetime(sp500_events[date_col], errors="coerce").dropna()
    if starts.empty:
        raise ValueError("sp500_events contains no valid usable dates.")
    return pd.Timestamp(starts.min()).normalize()


def _build_extended_voo_proxy(
    spy_close: pd.Series,
    voo_close: pd.Series,
    voo_inception: str,
) -> pd.Series:
    """Build a total-return proxy that uses SPY pre-inception and VOO thereafter."""
    inception = pd.Timestamp(voo_inception)
    spy = spy_close.sort_index().astype(float)
    voo = voo_close.sort_index().astype(float)

    combined_ret = pd.concat(
        [
            spy.pct_change(fill_method=None).loc[spy.index < inception],
            voo.pct_change(fill_method=None).loc[voo.index >= inception],
        ],
        axis=0,
    ).sort_index()
    combined_ret = combined_ret[~combined_ret.index.duplicated(keep="last")].fillna(0.0)

    tri = (1.0 + combined_ret).cumprod()
    tri.name = "voo_proxy_total_return_index"
    return tri

def build_panel(
    config: BacktestConfig,
    *,
    sma_length: int | None = None,
    constituent_fetcher: Callable[..., tuple[dict[str, pd.DataFrame], dict[str, str]]] | None = None,
    benchmark_fetcher: Callable[[str, str, str, BacktestConfig], tuple[pd.DataFrame, str]] | None = None,
) -> BacktestPanel:
    """Build the point-in-time constituent panel for a run.

    Args:
        config: Runtime configuration.
        sma_length: Length for the default signal mask; defaults to
            ``config.SMA_LENGTH_DAYS``.
        constituent_fetcher: Override for constituent price retrieval, used to
            swap CRSP tapes. Must match ``_fetch_constituent_prices``.
        benchmark_fetcher: Override for benchmark retrieval. Must match
            ``_fetch_benchmark_series``.

    Returns:
        A populated :class:`BacktestPanel`.
    """
    sma_length = int(sma_length if sma_length is not None else config.SMA_LENGTH_DAYS)
    constituent_fetcher = constituent_fetcher or _fetch_constituent_prices
    benchmark_fetcher = benchmark_fetcher or _fetch_benchmark_series

    if not config.has_crsp_credentials():
        raise ValueError("Missing CRSP/WRDS credentials; CRSP is the only price source.")

    # 1) Universe proxy sources (point-in-time snapshots)
    sp500_events = fetch_sp500_membership_history_public(config.START_DATE, config.END_DATE)
    sec_holdings = fetch_sec_voo_holdings_proxy(config.PRE2019_PROXY_CUTOFF, config.END_DATE)

    # 2) Benchmarks + trading calendar anchor
    voo_daily, voo_source = benchmark_fetcher("VOO", config.START_DATE, config.END_DATE, config)
    spy_daily, spy_source = benchmark_fetcher("SPY", config.START_DATE, config.END_DATE, config)

    voo_close = extract_adjusted_series(voo_daily, "close").sort_index()
    spy_close = extract_adjusted_series(spy_daily, "close").sort_index()

    strategy_start = max(pd.Timestamp(spy_close.index.min()), _first_universe_date(sp500_events))
    trading_index = pd.DatetimeIndex(spy_close.loc[spy_close.index >= strategy_start].index).sort_values().unique()
    extended_voo_proxy = _build_extended_voo_proxy(spy_close, voo_close, config.VOO_INCEPTION)

    LOGGER.info(
        "Strategy trading calendar: %s..%s (%s days)",
        trading_index.min().date().isoformat(),
        trading_index.max().date().isoformat(),
        len(trading_index),
    )

    # 3) Build PIT universe matrix
    membership = build_point_in_time_constituent_universe(
        sec_holdings=sec_holdings,
        sp500_membership_events=sp500_events,
        trading_index=trading_index,
        cutoff_date=config.PRE2019_PROXY_CUTOFF,
        holdings_lag_business_days=config.HOLDINGS_LAG_BUSINESS_DAYS,
    )

    universe_tickers = sorted({_normalize_ticker(t) for t in membership.columns})
    current_tickers = {
        _normalize_ticker(ticker)
        for ticker, is_member in membership.iloc[-1].items()
        if bool(is_member)
    }
    LOGGER.info("Universe symbols identified: %s", len(universe_tickers))

    # 4) Fetch constituent prices
    constituent_prices, constituent_source_map = constituent_fetcher(
        tickers=universe_tickers,
        start=config.START_DATE,
        end=config.END_DATE,
        config=config,
        current_tickers=current_tickers,
    )

    fetched_tickers = sorted(set(constituent_prices.keys()))
    if not fetched_tickers:
        raise RuntimeError("No constituent price histories were fetched from any provider.")

    constituentsnapshot_rows = [
        snapshot_row(
            constituent_source_map.get(ticker, config.PRIMARY_PRICE_SOURCE),
            ticker,
            df,
            config.START_DATE,
            config.END_DATE,
        )
        for ticker, df in sorted(constituent_prices.items())
    ]

    # 5) Build aligned matrices
    close_df = _build_matrix(constituent_prices, fetched_tickers, trading_index, "close")
    open_df = _build_matrix(constituent_prices, fetched_tickers, trading_index, "open")
    high_df = _build_matrix(constituent_prices, fetched_tickers, trading_index, "high")
    low_df = _build_matrix(constituent_prices, fetched_tickers, trading_index, "low")
    volume_df = _build_matrix(constituent_prices, fetched_tickers, trading_index, "volume")
    close_returns = _build_return_matrix(constituent_prices, fetched_tickers, trading_index)
    del constituent_prices

    # Restrict universe to tickers with any usable pricing history.
    valid_cols = [c for c in fetched_tickers if close_df[c].notna().any()]
    close_df = close_df[valid_cols]
    open_df = open_df[valid_cols]
    high_df = high_df[valid_cols]
    low_df = low_df[valid_cols]
    volume_df = volume_df[valid_cols]
    open_df = _sanitize_open_matrix(open_df, high_df, low_df, close_df).astype(np.float32)

    membership = membership.reindex(columns=valid_cols, fill_value=False)
    base_weights = membership.attrs.get(
        "base_weight_matrix",
        pd.DataFrame(0.0, index=membership.index, columns=membership.columns, dtype=np.float32),
    )
    base_weights = base_weights.reindex(index=trading_index, columns=valid_cols, fill_value=0.0).astype(np.float32)
    source_by_date = membership.attrs.get("source_by_date", pd.Series("unknown", index=trading_index, dtype=object))
    membership.attrs = {}

    coverage_ratio = (
        (membership & close_df.notna()).sum(axis=1) / membership.sum(axis=1).replace(0, np.nan)
    ).fillna(0.0)

    LOGGER.info(
        "Constituent matrices ready: tickers=%s, avg daily coverage=%.2f%%",
        len(valid_cols),
        float(coverage_ratio.mean() * 100.0),
    )
    provider_mix = pd.Series([constituent_source_map.get(t, "unknown") for t in valid_cols]).value_counts().to_dict()
    LOGGER.info("Constituent price providers: %s", provider_mix)

    # 6) Features + signals
    close_returns = close_returns.reindex(index=trading_index, columns=valid_cols).fillna(0.0)

    LOGGER.info("Building daily liquidity features.")

    liq = compute_daily_liquidity_feature_matrices(
        price_df=close_df,
        volume_df=volume_df,
        high_df=high_df,
        low_df=low_df,
        adv_lookback=config.ADV_LOOKBACK_DAYS,
        vol_lookback=config.VOL_LOOKBACK_DAYS,
        spread_model=config.SPREAD_MODEL,
        open_df=open_df,
    )

    tradable_sanity_mask = membership.reindex(index=trading_index, columns=valid_cols).fillna(False)
    if config.ENFORCE_INVESTABILITY_FILTER:
        tradable_sanity_mask &= close_df.ge(float(config.MIN_PRICE_TO_TRADE))
        tradable_sanity_mask &= liq["adv_usd"].fillna(0.0).ge(float(config.MIN_ADV_USD_TO_TRADE))
    _log_return_sanity(close_returns, tradable_sanity_mask)

    close_df.attrs["open_df"] = open_df
    close_df.attrs["adv_usd"] = liq["adv_usd"]
    close_df.attrs["sigma_20d"] = liq["sigma_20d"]
    close_df.attrs["spread_bps_est"] = liq["spread_bps_est"]

    del liq
    del high_df
    del low_df
    del volume_df
    del open_df
    gc.collect()

    LOGGER.info("Computing default SMA matrix.")
    sma = compute_sma_matrix(close_df, int(sma_length))
    LOGGER.info("Generating default active mask.")
    active_mask = generate_active_mask(
        prices=close_df,
        sma=sma,
        signal_type=config.SIGNAL_TYPE,
        entry_band_bps=config.ENTRY_BAND_BPS,
        exit_band_bps=config.EXIT_BAND_BPS,
    )
    del sma
    gc.collect()

    # 7) Cash curve
    cash_curve = _build_daily_cash_curve(config, trading_index)
    effective_cash_rate = float(cash_curve.mean()) if cash_curve is not None else float(config.CASH_RATE_ANNUAL)

    return BacktestPanel(
        trading_index=trading_index,
        close_df=close_df,
        close_returns=close_returns,
        membership=membership,
        base_weights=base_weights,
        active_mask=active_mask,
        sma_length=sma_length,
        valid_cols=valid_cols,
        cash_curve=cash_curve,
        effective_cash_rate=effective_cash_rate,
        spy_close=spy_close,
        voo_close=voo_close,
        extended_voo_proxy=extended_voo_proxy,
        coverage_ratio=coverage_ratio,
        source_by_date=pd.Series(source_by_date),
        provider_mix=provider_mix,
        constituentsnapshot_rows=constituentsnapshot_rows,
        voo_daily=voo_daily,
        spy_daily=spy_daily,
        voo_source=voo_source,
        spy_source=spy_source,
        sec_holdings=sec_holdings,
        sp500_events=sp500_events,
    )

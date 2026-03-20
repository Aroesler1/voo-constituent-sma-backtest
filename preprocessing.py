"""Preprocessing and feature engineering for constituent-level backtests."""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
FLOAT_DTYPE = np.float32


def _normalize_ticker(value: str) -> str:
    return str(value).strip().upper().replace(".", "-")


def _prepare_adjusted_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize vendor daily schema into adjusted OHLCV columns."""
    df = daily_df.copy()
    if "date" not in df.columns:
        raise ValueError("daily_df must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").set_index("date")

    if {"open", "high", "low", "close", "adjusted_close", "volume"}.issubset(df.columns):
        adj_factor = (df["adjusted_close"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        normalized = pd.DataFrame(
            {
                "adj_open": df["open"] * adj_factor,
                "adj_high": df["high"] * adj_factor,
                "adj_low": df["low"] * adj_factor,
                "adj_close": df["adjusted_close"],
                "adj_volume": df["volume"],
            },
            index=df.index,
        )
    elif {"adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"}.issubset(df.columns):
        normalized = pd.DataFrame(
            {
                "adj_open": df["adjOpen"],
                "adj_high": df["adjHigh"],
                "adj_low": df["adjLow"],
                "adj_close": df["adjClose"],
                "adj_volume": df["adjVolume"],
            },
            index=df.index,
        )
    else:
        raise ValueError("Unsupported daily schema. Expected adjusted OHLCV fields.")

    normalized = normalized.dropna(subset=["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"])
    if normalized.empty:
        raise ValueError("No valid adjusted rows after normalization.")
    return normalized


def resample_to_weekly(daily_df: pd.DataFrame, price_col: str = "adjusted_close") -> pd.DataFrame:
    """Resample daily adjusted data to weekly bars (W-FRI)."""
    _ = price_col
    daily = _prepare_adjusted_daily(daily_df)

    weekly = pd.DataFrame(
        {
            "open": daily["adj_open"].resample("W-FRI").first(),
            "high": daily["adj_high"].resample("W-FRI").max(),
            "low": daily["adj_low"].resample("W-FRI").min(),
            "close": daily["adj_close"].resample("W-FRI").last(),
            "volume": daily["adj_volume"].resample("W-FRI").sum(),
            "trading_days": daily["adj_close"].resample("W-FRI").count(),
        }
    )

    weekly = weekly.dropna(subset=["open", "high", "low", "close", "volume"])
    weekly = weekly[weekly["trading_days"] >= 3].copy()
    weekly = weekly.drop(columns=["trading_days"])

    LOGGER.info(
        "Weekly resample complete: rows=%s, range=%s..%s",
        len(weekly),
        weekly.index.min().date().isoformat() if not weekly.empty else "n/a",
        weekly.index.max().date().isoformat() if not weekly.empty else "n/a",
    )
    return weekly


def build_total_return_series(weekly_df: pd.DataFrame) -> pd.Series:
    """Build a total-return index from adjusted close."""
    if "close" not in weekly_df.columns:
        raise ValueError("weekly_df must contain 'close'.")

    close = weekly_df["close"].astype(float)
    log_returns = np.log(close / close.shift(1))
    tri = np.exp(log_returns.fillna(0.0).cumsum())
    if tri.empty:
        raise ValueError("Cannot build total-return index from empty weekly data.")
    tri = tri / tri.iloc[0]
    tri.name = "total_return_index"
    return tri


def _safe_correlation(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(a.corr(b))


def splice_spy_voo(
    spy_weekly: pd.DataFrame,
    voo_weekly: pd.DataFrame,
    voo_inception: str,
) -> pd.DataFrame:
    """Splice SPY pre-inception and VOO post-inception into extended series."""
    spy = spy_weekly.copy().sort_index()
    voo = voo_weekly.copy().sort_index()

    inception = pd.Timestamp(voo_inception)
    overlap_end = inception + pd.Timedelta(weeks=52)

    spy_ret = spy["close"].pct_change()
    voo_ret = voo["close"].pct_change()

    overlap = pd.concat(
        [
            spy_ret.loc[(spy_ret.index >= inception) & (spy_ret.index <= overlap_end)].rename("spy_return"),
            voo_ret.loc[(voo_ret.index >= inception) & (voo_ret.index <= overlap_end)].rename("voo_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    corr = _safe_correlation(overlap["spy_return"], overlap["voo_return"]) if not overlap.empty else np.nan
    mad_bps = (
        float((overlap["spy_return"] - overlap["voo_return"]).abs().mean() * 10000.0)
        if not overlap.empty
        else np.nan
    )

    LOGGER.info("SPY/VOO overlap validation: corr=%.6f, mad_bps=%.4f, weeks=%s", corr, mad_bps, len(overlap))

    if (not np.isnan(corr) and corr <= 0.999) or (not np.isnan(mad_bps) and mad_bps >= 2.0):
        warnings.warn(
            "SPY/VOO overlap quality check below target thresholds.",
            RuntimeWarning,
            stacklevel=2,
        )

    pre_spy = spy[spy.index < inception].copy()
    post_voo = voo[voo.index >= inception].copy()
    combined = pd.concat([pre_spy, post_voo], axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    pre_returns = spy_ret.loc[spy_ret.index < inception]
    post_returns = voo_ret.loc[voo_ret.index >= inception]
    combined_returns = pd.concat([pre_returns, post_returns], axis=0).sort_index()
    combined_returns = combined_returns.reindex(combined.index).fillna(0.0)

    tri = (1.0 + combined_returns).cumprod()
    tri = tri / tri.iloc[0]

    out = combined[["open", "high", "low", "close", "volume"]].copy()
    out["total_return_index"] = tri
    out = out.reset_index().rename(columns={"index": "date", "date": "date"})
    return out


def cross_validate_vendors(
    eodhd_weekly: pd.DataFrame,
    secondary_weekly: pd.DataFrame,
    threshold_bps: float,
) -> pd.DataFrame:
    """Cross-validate weekly return consistency across vendors."""
    eod = eodhd_weekly.copy()
    secondary = secondary_weekly.copy()

    if "date" not in eod.columns:
        eod = eod.reset_index().rename(columns={eod.index.name or "index": "date"})
    if "date" not in secondary.columns:
        secondary = secondary.reset_index().rename(columns={secondary.index.name or "index": "date"})

    eod["date"] = pd.to_datetime(eod["date"])
    secondary["date"] = pd.to_datetime(secondary["date"])

    eod_ret = eod[["date", "close"]].sort_values("date")
    eod_ret["eodhd_return"] = eod_ret["close"].pct_change()

    secondary_ret = secondary[["date", "close"]].sort_values("date")
    secondary_ret["secondary_return"] = secondary_ret["close"].pct_change()

    out = pd.merge(
        eod_ret[["date", "eodhd_return"]],
        secondary_ret[["date", "secondary_return"]],
        on="date",
        how="inner",
    ).dropna(subset=["eodhd_return", "secondary_return"])

    out["abs_diff_bps"] = (out["eodhd_return"] - out["secondary_return"]).abs() * 10000.0
    out["flagged"] = out["abs_diff_bps"] > threshold_bps

    flagged_count = int(out["flagged"].sum())
    total = len(out)
    LOGGER.info(
        "Vendor cross-validation: flagged=%s/%s, max=%.3f bps, mean=%.3f bps, median=%.3f bps",
        flagged_count,
        total,
        out["abs_diff_bps"].max() if total else np.nan,
        out["abs_diff_bps"].mean() if total else np.nan,
        out["abs_diff_bps"].median() if total else np.nan,
    )

    if total > 0 and flagged_count / total > 0.02:
        warnings.warn(
            f"Vendor discrepancies exceed 2% of weeks ({flagged_count / total:.2%}).",
            RuntimeWarning,
            stacklevel=2,
        )

    return out


def _corwin_schultz_spread(adj_high: pd.Series, adj_low: pd.Series) -> pd.Series:
    """Estimate effective spread using Corwin-Schultz high-low model."""
    h = adj_high.astype(float)
    l = adj_low.astype(float)

    log_hl = np.log(h / l).replace([np.inf, -np.inf], np.nan)
    beta = (log_hl**2).rolling(2).sum()

    high_2d = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    low_2d = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    gamma = (np.log(high_2d / low_2d) ** 2).replace([np.inf, -np.inf], np.nan)

    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = ((np.sqrt(2.0 * beta) - np.sqrt(beta)) / k) - np.sqrt((gamma / k).clip(lower=0.0))
    alpha = alpha.clip(lower=0.0)

    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.replace([np.inf, -np.inf], np.nan)


def compute_weekly_liquidity_features(
    daily_df: pd.DataFrame,
    adv_lookback: int,
    vol_lookback: int,
    spread_model: str,
) -> pd.DataFrame:
    """Compute weekly liquidity and volatility features for trading costs."""
    daily = _prepare_adjusted_daily(daily_df)

    daily_ret = np.log(daily["adj_close"] / daily["adj_close"].shift(1))
    adv_usd = (daily["adj_close"] * daily["adj_volume"]).rolling(adv_lookback).mean()
    sigma = daily_ret.rolling(vol_lookback).std()

    if spread_model.lower() != "corwin_schultz":
        raise ValueError("Only spread_model='corwin_schultz' is supported.")

    spread = _corwin_schultz_spread(daily["adj_high"], daily["adj_low"]) * 10000.0

    daily_features = pd.DataFrame(
        {
            "adv_usd": adv_usd,
            "sigma_20d": sigma,
            "spread_bps_est": spread,
        },
        index=daily.index,
    )

    weekly_features = daily_features.resample("W-FRI").last().replace([np.inf, -np.inf], np.nan)
    LOGGER.info(
        "Weekly liquidity features computed: rows=%s, adv missing=%.2f%%, sigma missing=%.2f%%, spread missing=%.2f%%",
        len(weekly_features),
        100.0 * weekly_features["adv_usd"].isna().mean() if len(weekly_features) else np.nan,
        100.0 * weekly_features["sigma_20d"].isna().mean() if len(weekly_features) else np.nan,
        100.0 * weekly_features["spread_bps_est"].isna().mean() if len(weekly_features) else np.nan,
    )
    return weekly_features


def compute_daily_liquidity_feature_matrices(
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    adv_lookback: int,
    vol_lookback: int,
    spread_model: str,
) -> dict[str, pd.DataFrame]:
    """Compute daily liquidity feature matrices for multi-asset cost modeling."""
    common_cols = sorted(set(price_df.columns) & set(volume_df.columns) & set(high_df.columns) & set(low_df.columns))
    if not common_cols:
        raise ValueError("No common columns across price/volume/high/low matrices.")

    px = price_df[common_cols].astype(FLOAT_DTYPE)
    vol = volume_df[common_cols].astype(FLOAT_DTYPE)
    hi = high_df[common_cols].astype(FLOAT_DTYPE)
    lo = low_df[common_cols].astype(FLOAT_DTYPE)

    adv_usd = (px * vol).rolling(adv_lookback).mean().astype(FLOAT_DTYPE)
    sigma_20d = np.log(px / px.shift(1)).rolling(vol_lookback).std().astype(FLOAT_DTYPE)

    spread_bps = pd.DataFrame(index=px.index, columns=common_cols, dtype=FLOAT_DTYPE)
    if spread_model.lower() != "corwin_schultz":
        raise ValueError("Only spread_model='corwin_schultz' is supported.")

    for ticker in common_cols:
        spread_bps[ticker] = (_corwin_schultz_spread(hi[ticker], lo[ticker]) * 10000.0).astype(FLOAT_DTYPE)

    return {
        "adv_usd": adv_usd.astype(FLOAT_DTYPE),
        "sigma_20d": sigma_20d.astype(FLOAT_DTYPE),
        "spread_bps_est": spread_bps.astype(FLOAT_DTYPE),
    }


def _build_membership_from_events(
    events_df: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build daily membership matrix from add/remove event rows."""
    if events_df.empty:
        return pd.DataFrame(False, index=trading_index, columns=[])

    events = events_df.copy()
    events["date"] = pd.to_datetime(events["date"])
    events["ticker"] = events["ticker"].map(_normalize_ticker)
    events["delta"] = np.where(events["action"].str.lower() == "add", 1, -1)
    events = events.dropna(subset=["date", "ticker"]).sort_values("date")

    pivot = events.pivot_table(
        index="date",
        columns="ticker",
        values="delta",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.sort_index()

    start = pd.Timestamp(trading_index.min())
    initial = pivot.loc[pivot.index < start].sum(axis=0) if not pivot.empty else pd.Series(dtype=float)
    flow = pivot.loc[pivot.index >= start].reindex(trading_index, fill_value=0)

    states = flow.cumsum()
    if len(initial) > 0:
        initial_aligned = initial.reindex(states.columns).fillna(0.0)
        states = states + initial_aligned

    return states > 0


def _build_membership_from_sec_snapshots(
    sec_holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    lag_business_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build daily membership and weight matrices from SEC snapshot rows."""
    if sec_holdings.empty:
        empty = pd.DataFrame(False, index=trading_index, columns=[])
        return empty, empty.astype(float)

    sec = sec_holdings.copy()
    sec["date"] = pd.to_datetime(sec["date"]) + pd.offsets.BDay(lag_business_days)
    sec["ticker"] = sec["ticker"].map(_normalize_ticker)
    sec = sec.sort_values(["date", "ticker"]).reset_index(drop=True)

    dates = sorted(sec["date"].dropna().unique())
    tickers = sorted(sec["ticker"].dropna().unique())

    mem = pd.DataFrame(False, index=trading_index, columns=tickers)
    wt = pd.DataFrame(0.0, index=trading_index, columns=tickers)

    for i, eff in enumerate(dates):
        start = pd.Timestamp(eff)
        end = pd.Timestamp(dates[i + 1]) if i + 1 < len(dates) else trading_index.max() + pd.Timedelta(days=1)
        idx_slice = trading_index[(trading_index >= start) & (trading_index < end)]
        if idx_slice.empty:
            continue

        snap = sec[sec["date"] == eff][["ticker", "weight"]].copy()
        if snap.empty:
            continue
        snap = snap[snap["weight"] > 0]
        if snap.empty:
            continue
        snap["weight"] = snap["weight"] / snap["weight"].sum()

        mem.loc[idx_slice, snap["ticker"].tolist()] = True
        for t, w in zip(snap["ticker"], snap["weight"]):
            wt.loc[idx_slice, t] = float(w)

    return mem, wt


def build_point_in_time_constituent_universe(
    sec_holdings: pd.DataFrame,
    sp500_membership_events: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    cutoff_date: str,
    holdings_lag_business_days: int,
) -> pd.DataFrame:
    """Build point-in-time daily membership matrix for the strategy universe.

    Post-cutoff uses SEC holdings proxy where available. Pre-cutoff uses public
    S&P 500 membership events. Missing SEC days post-cutoff fall back to S&P 500 proxy.

    Returns:
        Boolean membership DataFrame indexed by trading date.
        Extra data stored in attrs:
          - ``base_weight_matrix``
          - ``source_by_date``
    """
    idx = pd.DatetimeIndex(pd.to_datetime(trading_index)).sort_values().unique()

    pre_mem = _build_membership_from_events(sp500_membership_events, idx)
    sec_mem, sec_wt = _build_membership_from_sec_snapshots(sec_holdings, idx, holdings_lag_business_days)

    all_cols = sorted(set(pre_mem.columns) | set(sec_mem.columns))
    if not all_cols:
        raise ValueError("Universe construction failed: no constituents available from either source.")

    pre_mem = pre_mem.reindex(columns=all_cols, fill_value=False)
    sec_mem = sec_mem.reindex(columns=all_cols, fill_value=False)
    sec_wt = sec_wt.reindex(columns=all_cols, fill_value=0.0)

    cutoff = pd.Timestamp(cutoff_date)
    final_mem = pre_mem.copy()
    source = pd.Series("sp500_public_history", index=idx, dtype=object)

    post_idx = idx[idx >= cutoff]
    if len(post_idx) > 0:
        has_sec = sec_mem.loc[post_idx].any(axis=1)
        final_mem.loc[post_idx] = np.where(
            has_sec.to_numpy()[:, None],
            sec_mem.loc[post_idx].to_numpy(),
            pre_mem.loc[post_idx].to_numpy(),
        )
        source.loc[post_idx[has_sec.to_numpy()]] = "sec_proxy"

    # Base weights are SEC where available, otherwise equal-weight over available membership.
    base_w = pd.DataFrame(0.0, index=idx, columns=all_cols)
    base_w.loc[:, :] = sec_wt.reindex(index=idx, columns=all_cols, fill_value=0.0).to_numpy()

    fallback_rows = (base_w.sum(axis=1) <= 0) & (final_mem.sum(axis=1) > 0)
    if fallback_rows.any():
        counts = final_mem.loc[fallback_rows].sum(axis=1).replace(0, np.nan)
        eq = final_mem.loc[fallback_rows].div(counts, axis=0).fillna(0.0)
        base_w.loc[fallback_rows] = eq

    final_mem.attrs["base_weight_matrix"] = base_w
    final_mem.attrs["source_by_date"] = source
    LOGGER.info(
        "Built PIT universe: days=%s, avg constituents=%.1f, sec days=%s, preproxy days=%s",
        len(final_mem),
        float(final_mem.sum(axis=1).mean()),
        int((source == "sec_proxy").sum()),
        int((source == "sp500_public_history").sum()),
    )

    return final_mem


def build_rebalance_calendar(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Build deterministic execution dates mapped to trading days."""
    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values().unique()
    if len(idx) < 2:
        return idx

    freq = frequency.lower()
    if freq == "daily":
        return idx[1:]

    idx_series = pd.Series(idx, index=idx)

    if freq == "weekly":
        key = [f"{d.isocalendar().year}-{d.isocalendar().week:02d}" for d in idx]
        out = idx_series.groupby(key).first().sort_values().to_numpy()
        out_idx = pd.DatetimeIndex(out)
        return out_idx[out_idx > idx[0]]

    if freq == "monthly":
        out = idx_series.groupby(idx.to_period("M")).first().sort_values().to_numpy()
        out_idx = pd.DatetimeIndex(out)
        return out_idx[out_idx > idx[0]]

    if freq == "semi_monthly":
        dates: list[pd.Timestamp] = []
        by_month = idx_series.groupby(idx.to_period("M"))
        for _, month_dates in by_month:
            month_dates = month_dates.sort_values()
            first = month_dates.iloc[0]
            dates.append(first)

            anchor = pd.Timestamp(first.year, first.month, 15)
            candidate = month_dates[month_dates >= anchor]
            if not candidate.empty:
                dates.append(candidate.iloc[0])

        out_idx = pd.DatetimeIndex(sorted(set(pd.to_datetime(dates))))
        return out_idx[out_idx > idx[0]]

    raise ValueError("frequency must be one of {'daily','weekly','semi_monthly','monthly'}")


def estimate_proxy_fidelity(
    proxy_returns: pd.Series,
    benchmark_returns: pd.Series,
    coverage: pd.Series | None = None,
    te_high: float = 0.03,
    te_medium: float = 0.07,
) -> pd.DataFrame:
    """Estimate proxy fidelity and assign a quality grade."""
    aligned = pd.concat(
        [proxy_returns.rename("proxy"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        return pd.DataFrame(
            [
                {
                    "tracking_error_ann": np.nan,
                    "correlation": np.nan,
                    "mean_abs_diff_bps": np.nan,
                    "coverage_mean": np.nan,
                    "coverage_min": np.nan,
                    "fidelity_grade": "low",
                }
            ]
        )

    diff = aligned["proxy"] - aligned["benchmark"]
    tracking_error_ann = float(diff.std(ddof=1) * np.sqrt(252.0)) if len(diff) > 1 else np.nan
    correlation = float(aligned["proxy"].corr(aligned["benchmark"])) if len(aligned) > 1 else np.nan
    mean_abs_diff_bps = float(diff.abs().mean() * 10000.0)

    coverage_mean = float(coverage.mean()) if coverage is not None and len(coverage) else np.nan
    coverage_min = float(coverage.min()) if coverage is not None and len(coverage) else np.nan

    if pd.isna(tracking_error_ann):
        grade = "low"
    elif tracking_error_ann <= te_high and (pd.isna(coverage_mean) or coverage_mean >= 0.95):
        grade = "high"
    elif tracking_error_ann <= te_medium and (pd.isna(coverage_mean) or coverage_mean >= 0.80):
        grade = "medium"
    else:
        grade = "low"

    return pd.DataFrame(
        [
            {
                "tracking_error_ann": tracking_error_ann,
                "correlation": correlation,
                "mean_abs_diff_bps": mean_abs_diff_bps,
                "coverage_mean": coverage_mean,
                "coverage_min": coverage_min,
                "fidelity_grade": grade,
            }
        ]
    )

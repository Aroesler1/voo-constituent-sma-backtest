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
from config import BacktestConfig, config_hash, get_periods, load_config
from data_loader import (
    build_snapshot_manifest_row,
    fetch_crsp_batch_prices,
    fetch_eodhd,
    fetch_fred_cash_rate,
    fetch_sec_voo_holdings_proxy,
    fetch_sp500_membership_history_public,
)
from metrics import compute_drawdown_series, compute_metrics, compute_rolling_sharpe
from preprocessing import (
    build_point_in_time_constituent_universe,
    build_rebalance_calendar,
    compute_daily_liquidity_feature_matrices,
    estimate_proxy_fidelity,
)
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


def _normalize_ticker(value: str) -> str:
    return str(value).strip().upper().replace(".", "-")


def _to_date_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.sort_values("date").set_index("date")


def _extract_crsp_total_return_close(d: pd.DataFrame) -> pd.Series:
    """Construct a CRSP total-return price path from raw close and return fields."""
    raw_close = pd.to_numeric(d["close"], errors="coerce").abs()
    total_return = pd.to_numeric(d.get("total_return"), errors="coerce")
    if "permno" in d.columns:
        permno = pd.to_numeric(d["permno"], errors="coerce")
    else:
        permno = pd.Series(np.nan, index=d.index)

    out = pd.Series(index=d.index, dtype=float)
    prev_val = np.nan
    prev_raw = np.nan
    prev_permno = np.nan
    prev_date: pd.Timestamp | None = None

    for dt in d.index:
        raw = float(raw_close.loc[dt]) if pd.notna(raw_close.loc[dt]) else np.nan
        ret = float(total_return.loc[dt]) if pd.notna(total_return.loc[dt]) else np.nan
        perm = float(permno.loc[dt]) if pd.notna(permno.loc[dt]) else np.nan
        segment_break = (
            prev_date is None
            or (pd.notna(perm) and pd.notna(prev_permno) and perm != prev_permno)
            or (prev_date is not None and (dt - prev_date).days > 7)
        )

        if segment_break or not np.isfinite(prev_val) or prev_val <= 0:
            out.loc[dt] = raw if np.isfinite(raw) and raw > 0 else 1.0
        else:
            if not np.isfinite(ret):
                if np.isfinite(raw) and np.isfinite(prev_raw) and raw > 0 and prev_raw > 0:
                    ret = raw / prev_raw - 1.0
                else:
                    ret = 0.0
            out.loc[dt] = prev_val * (1.0 + float(ret))

        prev_val = float(out.loc[dt])
        prev_raw = raw
        prev_permno = perm
        prev_date = pd.Timestamp(dt)

    return out


def _extract_adjusted_series(df: pd.DataFrame, field: str) -> pd.Series:
    """Extract adjusted OHLCV series from EODHD-like frames."""
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
            s = _extract_adjusted_series(df, field)
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
        close = _extract_adjusted_series(df, "close")
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


def _provider_series_agree(primary_df: pd.DataFrame, fallback_df: pd.DataFrame) -> bool:
    """Check whether two vendor series materially agree on recent overlap."""
    try:
        primary = _extract_adjusted_series(primary_df, "close").dropna()
        fallback = _extract_adjusted_series(fallback_df, "close").dropna()
    except Exception:
        return True

    overlap = pd.concat([primary.rename("primary"), fallback.rename("fallback")], axis=1, join="inner").dropna()
    if len(overlap) < 60:
        return True

    ratio_full = (overlap["primary"] / overlap["fallback"]).replace([np.inf, -np.inf], np.nan).dropna()
    ret_full = overlap.pct_change(fill_method=None).dropna()
    overlap_recent = overlap.tail(252)
    ratio_recent = (overlap_recent["primary"] / overlap_recent["fallback"]).replace([np.inf, -np.inf], np.nan).dropna()
    ret_recent = overlap_recent.pct_change(fill_method=None).dropna()

    if ratio_full.empty or ret_recent.empty:
        return True

    median_ratio_recent = float(ratio_recent.median()) if len(ratio_recent) else np.nan
    corr_recent = float(ret_recent["primary"].corr(ret_recent["fallback"])) if len(ret_recent) > 1 else np.nan
    median_abs_diff_recent_bps = float((ret_recent["primary"] - ret_recent["fallback"]).abs().median() * 10000.0)

    ratio_outlier_share = float(((ratio_full < 0.5) | (ratio_full > 2.0)).mean()) if len(ratio_full) else 0.0
    ratio_q10 = float(ratio_full.quantile(0.10)) if len(ratio_full) else np.nan
    ratio_q90 = float(ratio_full.quantile(0.90)) if len(ratio_full) else np.nan
    ratio_dispersion = (ratio_q90 / ratio_q10) if pd.notna(ratio_q10) and ratio_q10 > 0 else np.inf
    corr_full = float(ret_full["primary"].corr(ret_full["fallback"])) if len(ret_full) > 1 else np.nan
    median_abs_diff_full_bps = float((ret_full["primary"] - ret_full["fallback"]).abs().median() * 10000.0) if len(ret_full) else np.nan

    recent_ok = 0.80 <= median_ratio_recent <= 1.25 and (
        (pd.notna(corr_recent) and corr_recent >= 0.98) or median_abs_diff_recent_bps <= 25.0
    )
    full_ok = (
        ratio_outlier_share <= 0.02
        and ratio_dispersion <= 1.5
        and (
            (pd.notna(corr_full) and corr_full >= 0.95)
            or median_abs_diff_full_bps <= 40.0
        )
    )
    return recent_ok and full_ok


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


def _merge_price_frames(primary_df: pd.DataFrame, fallback_df: pd.DataFrame) -> pd.DataFrame:
    """Merge two normalized price frames, preferring primary rows on overlap."""
    primary = primary_df.copy()
    fallback = fallback_df.copy()
    primary["date"] = pd.to_datetime(primary["date"])
    fallback["date"] = pd.to_datetime(fallback["date"])

    merged = pd.concat([primary, fallback], axis=0, ignore_index=True, sort=False)
    merged = merged.sort_values(["date"]).drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    merged.attrs["snapshot_id"] = f"{primary_df.attrs.get('snapshot_id')}|{fallback_df.attrs.get('snapshot_id')}"
    merged.attrs["fetched_at_utc"] = max(
        str(primary_df.attrs.get("fetched_at_utc", "")),
        str(fallback_df.attrs.get("fetched_at_utc", "")),
    )
    merged.attrs["sha256"] = f"{primary_df.attrs.get('sha256')}|{fallback_df.attrs.get('sha256')}"
    return merged


def _fetch_constituent_price_batch_eodhd(
    tickers: list[str],
    start: str,
    end: str,
    api_key: str,
) -> dict[str, pd.DataFrame]:
    """Fetch constituent prices from EODHD with progress logging and skip-on-failure."""
    requested = sorted({_normalize_ticker(t) for t in tickers if str(t).strip()})
    prices: dict[str, pd.DataFrame] = {}
    failed = 0

    for i, ticker in enumerate(requested, start=1):
        try:
            df = fetch_eodhd(ticker=ticker, start=start, end=end, api_key=api_key)
            prices[ticker] = df
        except Exception as exc:
            failed += 1
            LOGGER.warning("Skipping %s after EODHD fetch failure: %s", ticker, exc)

        if i % 100 == 0 or i == len(requested):
            LOGGER.info(
                "Constituent fetch progress: %s/%s completed (success=%s, failed=%s).",
                i,
                len(requested),
                len(prices),
                failed,
            )

    if not prices:
        raise RuntimeError("No constituent price histories were fetched.")

    return prices


def _fetch_single_price_with_fallback(
    ticker: str,
    start: str,
    end: str,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, str]:
    """Fetch one benchmark series using CRSP first when configured, then EODHD."""
    benchmark_etfs = {"VOO", "SPY"}
    if ticker.upper() in benchmark_etfs:
        if not config.EODHD_API_KEY:
            raise ValueError(f"EODHD_API_KEY is required for benchmark ETF {ticker}.")
        return fetch_eodhd(ticker, start, end, config.EODHD_API_KEY), "eodhd"

    if config.PRIMARY_PRICE_SOURCE == "crsp" and config.has_crsp_credentials():
        try:
            crsp_map = fetch_crsp_batch_prices(
                tickers=[ticker],
                start=start,
                end=end,
                username=config.WRDS_USERNAME or config.CRSP_USERNAME,
                password=config.WRDS_PASSWORD,
                api_key=config.CRSP_API_KEY,
            )
            if ticker in crsp_map and not crsp_map[ticker].empty:
                if not _needs_recent_tail(crsp_map[ticker], end) or not config.EODHD_API_KEY:
                    return crsp_map[ticker], "crsp"
                LOGGER.warning(
                    "CRSP coverage for %s ends at %s; extending with EODHD.",
                    ticker,
                    _max_available_date(crsp_map[ticker]).date().isoformat(),
                )
                eod_df = fetch_eodhd(ticker, start, end, config.EODHD_API_KEY)
                return _merge_price_frames(crsp_map[ticker], eod_df), "crsp+eodhd"
        except Exception as exc:
            LOGGER.warning("CRSP benchmark fetch failed for %s: %s", ticker, exc)

    if not config.EODHD_API_KEY:
        raise ValueError(f"No fallback provider available for {ticker}; missing EODHD_API_KEY.")
    return fetch_eodhd(ticker, start, end, config.EODHD_API_KEY), "eodhd"


def _fetch_constituent_prices_with_fallback(
    tickers: list[str],
    start: str,
    end: str,
    config: BacktestConfig,
    current_tickers: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Fetch constituent prices using CRSP first when available, then EODHD fallback."""
    requested = sorted({_normalize_ticker(t) for t in tickers if str(t).strip()})
    prices: dict[str, pd.DataFrame] = {}
    source_map: dict[str, str] = {}

    if config.PRIMARY_PRICE_SOURCE == "crsp" and config.has_crsp_credentials():
        try:
            crsp_prices = fetch_crsp_batch_prices(
                tickers=requested,
                start=start,
                end=end,
                username=config.WRDS_USERNAME or config.CRSP_USERNAME,
                password=config.WRDS_PASSWORD,
                api_key=config.CRSP_API_KEY,
            )
            prices.update(crsp_prices)
            source_map.update({ticker: "crsp" for ticker in crsp_prices})
            LOGGER.info("Primary CRSP fetch resolved %s/%s tickers.", len(crsp_prices), len(requested))
        except Exception as exc:
            LOGGER.warning("CRSP batch fetch failed; falling back to EODHD for all missing tickers: %s", exc)

    remaining = [ticker for ticker in requested if ticker not in prices]
    stale_current = [
        ticker
        for ticker, df in prices.items()
        if current_tickers and ticker in current_tickers and _needs_recent_tail(df, end)
    ]
    if stale_current:
        LOGGER.warning("CRSP stale current constituents requiring EODHD tail extension: %s", len(stale_current))
        remaining = sorted(set(remaining) | set(stale_current))
    if remaining:
        if not config.EODHD_API_KEY:
            LOGGER.warning("EODHD fallback unavailable; unresolved tickers remain: %s", len(remaining))
        else:
            eod_prices = _fetch_constituent_price_batch_eodhd(
                tickers=remaining,
                start=start,
                end=end,
                api_key=config.EODHD_API_KEY,
            )
            for ticker, eod_df in eod_prices.items():
                if ticker in prices:
                    if current_tickers and ticker in current_tickers and not _provider_series_agree(prices[ticker], eod_df):
                        LOGGER.warning(
                            "Provider validation failed for %s; replacing CRSP history with EODHD for ticker consistency.",
                            ticker,
                        )
                        prices[ticker] = eod_df
                        source_map[ticker] = "eodhd_validated"
                    else:
                        prices[ticker] = _merge_price_frames(prices[ticker], eod_df)
                        source_map[ticker] = "crsp+eodhd"
                else:
                    prices[ticker] = eod_df
                    source_map[ticker] = "eodhd"

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


def _snapshot_row(vendor: str, ticker: str, df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
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


def main() -> None:
    """Run full constituent-level institutional backtest workflow."""
    started = time.perf_counter()

    config = load_config()
    setup_logging(config.OUTPUT_DIR)

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(config.CACHE_DIR).mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting institutional constituent-level VOO SMA backtest.")
    if not config.has_crsp_credentials() and not config.EODHD_API_KEY:
        raise ValueError("Missing both CRSP/WRDS credentials and EODHD_API_KEY.")

    # 1) Universe proxy sources (point-in-time snapshots)
    sp500_events = fetch_sp500_membership_history_public(config.START_DATE, config.END_DATE)
    sec_holdings = fetch_sec_voo_holdings_proxy(config.PRE2019_PROXY_CUTOFF, config.END_DATE)

    # 2) Benchmarks + trading calendar anchor
    voo_daily, voo_source = _fetch_single_price_with_fallback("VOO", config.START_DATE, config.END_DATE, config)
    spy_daily, spy_source = _fetch_single_price_with_fallback("SPY", config.START_DATE, config.END_DATE, config)

    voo_close = _extract_adjusted_series(voo_daily, "close").sort_index()
    spy_close = _extract_adjusted_series(spy_daily, "close").sort_index()

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
    constituent_prices, constituent_source_map = _fetch_constituent_prices_with_fallback(
        tickers=universe_tickers,
        start=config.START_DATE,
        end=config.END_DATE,
        config=config,
        current_tickers=current_tickers,
    )

    fetched_tickers = sorted(set(constituent_prices.keys()))
    if not fetched_tickers:
        raise RuntimeError("No constituent price histories were fetched from any provider.")

    constituent_snapshot_rows = [
        _snapshot_row(
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
    sma = compute_sma_matrix(close_df, config.SMA_LENGTH_DAYS)
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
        _snapshot_row(voo_source, "VOO", voo_daily, config.START_DATE, config.END_DATE),
        _snapshot_row(spy_source, "SPY", spy_daily, config.START_DATE, config.END_DATE),
        _snapshot_row("secproxy", "VOO", sec_holdings, config.PRE2019_PROXY_CUTOFF, config.END_DATE),
        _snapshot_row("sp500pub", "membership", sp500_events, config.START_DATE, config.END_DATE),
    ]
    snapshot_rows.extend(constituent_snapshot_rows)

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
            if k not in {"EODHD_API_KEY", "FRED_API_KEY"}
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

"""Signal-generation logic for constituent and single-asset SMA strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sma(prices: pd.Series, length: int) -> pd.Series:
    """Compute a simple moving average for a single series.

    Args:
        prices: Price series indexed by date.
        length: SMA lookback length.

    Returns:
        SMA series with NaN for warm-up rows.
    """
    if length <= 0:
        raise ValueError("length must be positive.")
    return prices.astype(float).rolling(window=length, min_periods=length).mean()


def _generate_signals_core(
    prices: pd.Series,
    sma: pd.Series,
    signal_type: str,
    entry_band_bps: float,
    exit_band_bps: float,
) -> pd.Series:
    """Generate integer long/flat signals for one ticker."""
    out = _generate_signal_array(
        px=prices.to_numpy(dtype=np.float32, copy=False),
        ma=sma.to_numpy(dtype=np.float32, copy=False),
        signal_type=signal_type,
        entry_band_bps=entry_band_bps,
        exit_band_bps=exit_band_bps,
    )
    return pd.Series(out, index=prices.index, dtype="Float32").astype("Int64")


def _generate_signal_array(
    px: np.ndarray,
    ma: np.ndarray,
    signal_type: str,
    entry_band_bps: float,
    exit_band_bps: float,
) -> np.ndarray:
    """Generate float32 signal array with NaN warm-up values."""
    if signal_type not in {"level", "cross"}:
        raise ValueError("signal_type must be 'level' or 'cross'.")

    entry_threshold = ma * (1.0 + entry_band_bps / 10000.0)
    exit_threshold = ma * (1.0 - exit_band_bps / 10000.0)

    out = np.full(len(px), np.nan, dtype=np.float32)
    position = 0

    for i in range(len(px)):
        p = float(px[i])
        m = float(ma[i])
        if not np.isfinite(p) or not np.isfinite(m):
            continue

        if signal_type == "level":
            if position == 0 and p > entry_threshold[i]:
                position = 1
            elif position == 1 and p < exit_threshold[i]:
                position = 0
            out[i] = np.float32(position)
            continue

        if i == 0:
            out[i] = np.float32(position)
            continue

        p_prev = px[i - 1]
        entry_prev = entry_threshold[i - 1]
        exit_prev = exit_threshold[i - 1]

        if np.isfinite(p_prev) and np.isfinite(entry_prev) and np.isfinite(exit_prev):
            crossed_up = p_prev <= entry_prev and p > entry_threshold[i]
            crossed_down = p_prev >= exit_prev and p < exit_threshold[i]
            if position == 0 and crossed_up:
                position = 1
            elif position == 1 and crossed_down:
                position = 0

        out[i] = np.float32(position)

    return out


def generate_signals(
    weekly_prices: pd.Series,
    sma: pd.Series,
    signal_type: str,
    entry_band_bps: float,
    exit_band_bps: float,
) -> pd.Series:
    """Generate long/cash signals for a single series.

    Args:
        weekly_prices: Price series.
        sma: SMA series aligned to ``weekly_prices``.
        signal_type: Either ``level`` or ``cross``.
        entry_band_bps: Entry band in basis points.
        exit_band_bps: Exit band in basis points.

    Returns:
        Integer signal series in {0, 1} with warm-up NaNs.
    """
    return _generate_signals_core(
        prices=weekly_prices,
        sma=sma,
        signal_type=signal_type,
        entry_band_bps=entry_band_bps,
        exit_band_bps=exit_band_bps,
    )


def compute_sma_matrix(prices: pd.DataFrame, length: int) -> pd.DataFrame:
    """Compute SMA for each ticker column.

    Args:
        prices: Adjusted close matrix indexed by date.
        length: SMA lookback length in trading days.

    Returns:
        DataFrame of SMAs with the same shape as input.
    """
    if length <= 0:
        raise ValueError("length must be positive.")
    clean = prices.astype(np.float32, copy=False)
    return clean.rolling(window=length, min_periods=length).mean().astype(np.float32)


def generate_active_mask(
    prices: pd.DataFrame,
    sma: pd.DataFrame,
    signal_type: str,
    entry_band_bps: float,
    exit_band_bps: float,
) -> pd.DataFrame:
    """Generate per-ticker long/flat mask from price-vs-SMA signals.

    Args:
        prices: Price matrix indexed by date.
        sma: SMA matrix aligned to ``prices``.
        signal_type: ``level`` or ``cross``.
        entry_band_bps: Entry band in basis points.
        exit_band_bps: Exit band in basis points.

    Returns:
        DataFrame with per-ticker values {0.0, 1.0, NaN}; warm-up stays NaN.
    """
    if not prices.index.equals(sma.index):
        raise ValueError("prices and sma must share the same index.")

    common_cols = [c for c in prices.columns if c in sma.columns]
    if not common_cols:
        raise ValueError("No overlapping ticker columns between prices and sma.")

    px = prices[common_cols].to_numpy(dtype=np.float32, copy=False)
    ma = sma[common_cols].to_numpy(dtype=np.float32, copy=False)
    out = np.full(px.shape, np.nan, dtype=np.float32)

    for j, ticker in enumerate(common_cols):
        out[:, j] = _generate_signal_array(
            px=px[:, j],
            ma=ma[:, j],
            signal_type=signal_type,
            entry_band_bps=entry_band_bps,
            exit_band_bps=exit_band_bps,
        )

    return pd.DataFrame(out, index=prices.index, columns=common_cols, dtype=np.float32)

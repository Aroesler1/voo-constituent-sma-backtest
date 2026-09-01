"""EDGE bid-ask spread estimation from OHLC prices.

Implements the efficient estimator of Ardia, Guidotti & Kroencke (2024),
"Efficient Estimation of Bid-Ask Spreads from Open, High, Low, and Close
Prices", Journal of Financial Economics 161, 103916.

The `edge` function is vendored from the authors' reference implementation
(https://github.com/eguidotti/bidask, MIT license) so the estimator matches
the published paper exactly. The rolling/stepped matrix wrappers are local.

EDGE replaces Corwin-Schultz as the default spread model: it is unbiased in
the presence of overnight returns and substantially more efficient on daily
OHLC data (see the paper's simulation results).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def edge(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, sign: bool = False) -> float:
    """Full-sample EDGE spread estimate (fraction; 0.01 means 1%).

    Vendored from https://github.com/eguidotti/bidask (MIT license).
    Inputs are price vectors sorted in ascending timestamp order.
    """
    nobs = len(open)
    if len(high) != nobs or len(low) != nobs or len(close) != nobs:
        raise ValueError("Open, high, low, and close prices must have the same length")

    if nobs < 3:
        return np.nan

    o = np.log(np.asarray(open, dtype=float))
    h = np.log(np.asarray(high, dtype=float))
    l = np.log(np.asarray(low, dtype=float))
    c = np.log(np.asarray(close, dtype=float))
    m = (h + l) / 2.0

    h1, l1, c1, m1 = h[:-1], l[:-1], c[:-1], m[:-1]
    o, h, l, c, m = o[1:], h[1:], l[1:], c[1:], m[1:]

    r1 = m - o
    r2 = o - m1
    r3 = m - c1
    r4 = c1 - m1
    r5 = o - c1

    tau = np.where(np.isnan(h) | np.isnan(l) | np.isnan(c1), np.nan, (h != l) | (l != c1))
    po1 = tau * np.where(np.isnan(o) | np.isnan(h), np.nan, o != h)
    po2 = tau * np.where(np.isnan(o) | np.isnan(l), np.nan, o != l)
    pc1 = tau * np.where(np.isnan(c1) | np.isnan(h1), np.nan, c1 != h1)
    pc2 = tau * np.where(np.isnan(c1) | np.isnan(l1), np.nan, c1 != l1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        pt = np.nanmean(tau)
        po = np.nanmean(po1) + np.nanmean(po2)
        pc = np.nanmean(pc1) + np.nanmean(pc2)

        if np.nansum(tau) < 2 or po == 0 or pc == 0:
            return np.nan

        d1 = r1 - np.nanmean(r1) / pt * tau
        d3 = r3 - np.nanmean(r3) / pt * tau
        d5 = r5 - np.nanmean(r5) / pt * tau

        x1 = -4.0 / po * d1 * r2 + -4.0 / pc * d3 * r4
        x2 = -4.0 / po * d1 * r5 + -4.0 / pc * d5 * r4

        e1 = np.nanmean(x1)
        e2 = np.nanmean(x2)

        v1 = np.nanmean(x1**2) - e1**2
        v2 = np.nanmean(x2**2) - e2**2

    vt = v1 + v2
    s2 = (v2 * e1 + v1 * e2) / vt if vt > 0 else (e1 + e2) / 2.0

    s = np.sqrt(np.abs(s2))
    if sign:
        s *= np.sign(s2)

    return float(s)


def edge_spread_series(
    open_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    close_s: pd.Series,
    window: int = 126,
    step: int = 21,
) -> pd.Series:
    """Stepped trailing-window EDGE estimates, forward-filled between steps.

    The estimate at date t uses only the `window` trading days ending at t
    (point-in-time safe). Estimates are refreshed every `step` days because
    daily-frequency spread level changes are slow relative to the estimator's
    sampling noise, and stepping keeps the panel computation tractable.

    Returns a spread series as a FRACTION (0.001 = 10 bps).
    """
    idx = close_s.index
    n = len(idx)
    out = pd.Series(np.nan, index=idx, dtype=float)
    if n < window:
        return out

    o = open_s.to_numpy(dtype=float)
    h = high_s.to_numpy(dtype=float)
    l = low_s.to_numpy(dtype=float)
    c = close_s.to_numpy(dtype=float)

    for end in range(window, n + 1, step):
        start = end - window
        # require reasonable coverage inside the window
        valid = np.isfinite(c[start:end])
        if valid.sum() < window // 2:
            continue
        est = edge(o[start:end], h[start:end], l[start:end], c[start:end])
        out.iloc[end - 1] = est

    return out.ffill()


def edge_spread_matrix(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    window: int = 126,
    step: int = 21,
) -> pd.DataFrame:
    """Per-ticker stepped-rolling EDGE spreads (fraction) for a price panel."""
    common_cols = [c for c in close_df.columns if c in open_df.columns and c in high_df.columns and c in low_df.columns]
    out = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns, dtype=float)
    for ticker in common_cols:
        out[ticker] = edge_spread_series(
            open_df[ticker],
            high_df[ticker],
            low_df[ticker],
            close_df[ticker],
            window=window,
            step=step,
        )
    LOGGER.info(
        "EDGE spread matrix computed: tickers=%s, window=%s, step=%s, missing=%.2f%%",
        len(common_cols),
        window,
        step,
        100.0 * out.isna().mean().mean() if len(out) else np.nan,
    )
    return out

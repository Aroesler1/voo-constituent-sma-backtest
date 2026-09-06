"""CRSP Flat File Format 2.0 (CIZ) price loader, for tape comparison.

In January 2025 CRSP shipped the last release of the legacy Flat File Format
1.0 (SIZ) tape and now updates only Format 2.0 (CIZ). Schwarz, Walter & Weiss,
"Rewriting CRSP's History: Impact of Altered Monthly Returns on Asset Pricing"
(*Journal of Financial and Quantitative Analysis*, 24 February 2026; SSRN
5074864), measure the consequence: the transition "rewrites 9.62% of monthly
returns by more than 1 basis point, primarily due to a change in the dividend
reinvestment assumption" -- payouts reinvest on the ex-date in CIZ against
month-end in SIZ -- with a 22 bp mean absolute difference among altered returns.

Their result also contains a prediction this repository can test directly. The
reinvestment change is a *monthly* artefact; the paper states that daily returns
"did not change materially" in CIZ, and uses that to rebuild the new monthly
returns from the old daily ones. A strategy evaluated on daily data should
therefore be close to tape-invariant. ``run_tape_compare.py`` checks whether it
is.

On WRDS the two tapes are ``crsp.dsf`` (legacy) and ``crsp.dsf_v2`` (CIZ).

Two deliberate choices keep the comparison clean:

1. The ticker-to-PERMNO resolution comes from the *legacy* name tables in both
   runs, via ``data_loader._resolve_crsp_name_history``. CIZ ships its own
   ``stocknames_v2``, and using it would mix a universe change into what is
   meant to be a price change.
2. Normalisation runs through ``data_loader._normalize_crsp_daily`` after the
   CIZ columns are mapped onto the legacy raw names, so adjustment factors,
   delisting handling and de-duplication are byte-identical between tapes and
   only the vendor numbers differ.

Delisting returns are a known open difference. The legacy path joins
``crsp.dsedelist`` and compounds ``dlret`` into a delisted name's final day;
``crsp.dsf_v2.dlyret`` does not carry it. Verified on Big Lots, 2024-09-06:
both tapes report a price return of +9.11%, the legacy tape reports a total
return of -76.28% after compounding the delisting return, and the CIZ tape
reports +9.11%. So the CIZ delisting return has to be joined from its own
security-info history, which this loader does not yet do. The consequence is
confined to one row per delisted security, and ``run_tape_compare.py`` reports
the comparison with and without those rows so the size of the gap is visible
rather than buried in the total.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd

from config import BacktestConfig, load_config
from data_loader import (
    _get_wrds_connection,
    _normalize_crsp_daily,
    _normalize_ticker,
    _resolve_crsp_name_history,
)

LOGGER = logging.getLogger(__name__)

V2_CACHE_DIRNAME = "crsp_v2"

# CIZ column -> the legacy raw name that _normalize_crsp_daily expects.
V2_COLUMN_MAP = {
    "dlycaldt": "date",
    "dlyopen": "open_raw",
    "dlyhigh": "high_raw",
    "dlylow": "low_raw",
    "dlyclose": "close_raw",
    "dlyvol": "volume_raw",
    "dlyret": "ret_raw",
    "dlyretx": "retx_raw",
    "dlycumfacpr": "cfacpr",
    "dlycumfacshr": "cfacshr",
}


def _v2_cache_dir(config: BacktestConfig) -> Path:
    path = Path(config.CACHE_DIR) / V2_CACHE_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(config: BacktestConfig, ticker: str, start: str, end: str) -> Path:
    return _v2_cache_dir(config) / f"crspv2_{_normalize_ticker(ticker)}_{start}_{end}.parquet"


def normalize_v2_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Map CIZ columns onto the legacy raw schema and normalise them.

    Args:
        raw: Rows straight out of ``crsp.dsf_v2``, plus a ``permno`` column.

    Returns:
        A frame in the project's adjusted-price schema, identical in shape and
        column semantics to the legacy loader's output.
    """
    if raw.empty:
        return _normalize_crsp_daily(raw)

    mapped = raw.rename(columns=V2_COLUMN_MAP).copy()
    # No delisting return is available here: CIZ keeps it out of DlyRet and in a
    # separate security-info history this loader does not yet read. The column
    # must still exist for the shared normaliser. See the module docstring.
    mapped["dlret_raw"] = np.nan
    return _normalize_crsp_daily(mapped)


def fetch_crsp_v2_batch_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    config: BacktestConfig | None = None,
    include_funds: bool = False,
    batch_size: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch daily prices from ``crsp.dsf_v2``, cached per ticker.

    Args:
        tickers: Requested tickers.
        start: ISO start date.
        end: ISO end date.
        config: Runtime config; loaded from the environment if omitted.
        include_funds: Admit ETF share codes, for benchmark requests.
        batch_size: PERMNOs per SQL round trip.

    Returns:
        Mapping of ticker to normalised daily frame.
    """
    cfg = config or load_config()
    requested = sorted({_normalize_ticker(t) for t in tickers if str(t).strip()})
    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for ticker in requested:
        path = _cache_path(cfg, ticker, start, end)
        if path.exists():
            cached = pd.read_parquet(path)
            if not cached.empty:
                out[ticker] = cached
                continue
        missing.append(ticker)

    if not missing:
        LOGGER.info("CRSP v2: served %s tickers from cache.", len(out))
        return out

    try:
        conn = _get_wrds_connection(
            username=cfg.WRDS_USERNAME or cfg.CRSP_USERNAME,
            password=cfg.WRDS_PASSWORD or cfg.CRSP_API_KEY,
        )
    except (RuntimeError, ImportError) as exc:
        # Degrade to the cache rather than losing an otherwise complete run,
        # matching the legacy loader. The tickers still missing are the ones
        # CRSP cannot resolve at all, so a connection would not have helped.
        LOGGER.warning(
            "CRSP v2: WRDS unavailable (%s); serving %s cached ticker(s), %s uncached.",
            exc, len(out), len(missing),
        )
        return out

    size = int(batch_size or cfg.CRSP_BATCH_SIZE)
    try:
        for begin in range(0, len(missing), size):
            chunk = missing[begin : begin + size]
            names = _resolve_crsp_name_history(conn, chunk, start, end, include_funds=include_funds)
            if names.empty:
                LOGGER.warning("CRSP v2: no legacy name history for chunk starting %s.", chunk[0])
                continue

            permnos = sorted(set(names["permno"].tolist()))
            permno_sql = ", ".join(str(p) for p in permnos)
            sql = f"""
                SELECT
                    d.dlycaldt,
                    d.permno,
                    d.dlyopen,
                    d.dlyhigh,
                    d.dlylow,
                    d.dlyclose,
                    d.dlyvol,
                    d.dlyret,
                    d.dlyretx,
                    d.dlycumfacpr,
                    d.dlycumfacshr
                FROM crsp.dsf_v2 AS d
                WHERE d.dlycaldt BETWEEN '{start}' AND '{end}'
                  AND d.permno IN ({permno_sql})
                ORDER BY d.permno, d.dlycaldt
            """
            raw_df = conn.raw_sql(sql, date_cols=["dlycaldt"])
            if raw_df.empty:
                LOGGER.warning("CRSP v2: no rows for chunk starting %s.", chunk[0])
                continue

            merged = raw_df.merge(names, on="permno", how="inner", suffixes=("", "_name"))
            merged["dlycaldt"] = pd.to_datetime(merged["dlycaldt"], errors="coerce")
            merged = merged[
                (merged["dlycaldt"] >= merged["namedt"]) & (merged["dlycaldt"] <= merged["nameenddt"])
            ].copy()
            if merged.empty:
                continue

            # Same share-class priority and same de-duplication rule as the
            # legacy path, so a dual-class ticker resolves to the same security.
            share_priority = {11: 0, 10: 1, 18: 2, 12: 3, 41: 4, 40: 5, 42: 6, 48: 7, 71: 8, 70: 9, 72: 10}
            merged["share_priority"] = merged["shrcd"].map(share_priority).fillna(99).astype(int)
            merged = merged.sort_values(
                ["ticker", "dlycaldt", "share_priority", "namedt", "nameenddt", "exchcd", "permno"],
                ascending=[True, True, True, False, False, True, False],
            ).drop_duplicates(subset=["ticker", "dlycaldt"], keep="first")

            for ticker, group in merged.groupby("ticker", sort=False):
                normalized = normalize_v2_daily(group)
                if normalized.empty:
                    continue
                normalized.to_parquet(_cache_path(cfg, ticker, start, end), index=False)
                out[ticker] = normalized

            LOGGER.info("CRSP v2: fetched %s/%s tickers.", len(out), len(requested))
    finally:
        with suppress(Exception):
            conn.close()

    unresolved = sorted(set(requested) - set(out))
    if unresolved:
        LOGGER.info("CRSP v2: %s tickers unresolved.", len(unresolved))
    return out


def fetch_constituent_prices_v2(
    tickers: list[str],
    start: str,
    end: str,
    config: BacktestConfig,
    current_tickers: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """``panel.build_panel`` constituent hook backed by ``crsp.dsf_v2``."""
    _ = current_tickers
    prices = fetch_crsp_v2_batch_prices(list(tickers), start, end, config=config)
    return prices, {ticker: "crsp_v2" for ticker in prices}


def fetch_benchmark_series_v2(
    ticker: str,
    start: str,
    end: str,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, str]:
    """``panel.build_panel`` benchmark hook backed by ``crsp.dsf_v2``."""
    prices = fetch_crsp_v2_batch_prices([ticker], start, end, config=config, include_funds=True)
    df = prices.get(_normalize_ticker(ticker))
    if df is None or df.empty:
        raise ValueError(f"CRSP v2 returned no rows for benchmark {ticker}.")
    return df, "crsp_v2"


def compare_return_panels(
    legacy: pd.DataFrame,
    v2: pd.DataFrame,
    *,
    legacy_valid: pd.DataFrame | None = None,
    v2_valid: pd.DataFrame | None = None,
    threshold_bps: float = 1.0,
) -> dict[str, float]:
    """Count constituent-days whose return differs between the two tapes.

    Args:
        legacy: Daily simple-return matrix (date x ticker) from ``crsp.dsf``.
        v2: The same matrix from ``crsp.dsf_v2``.
        legacy_valid: Boolean matrix marking cells the legacy tape actually
            covers. Needed because ``panel.build_panel`` fills uncovered cells
            with zero rather than NaN, so without a mask the denominator counts
            the millions of ticker-days on which neither tape has a security and
            both "agree" at zero. Defaults to ``legacy.notna()``.
        v2_valid: The same for the CIZ tape.
        threshold_bps: Difference above which a day counts as altered.

    Returns:
        Counts and shares of altered days, plus the mean and max absolute
        difference among the altered ones. Cells one tape covers and the other
        does not are reported separately: those are coverage differences, not
        return differences.
    """
    cols = sorted(set(legacy.columns) & set(v2.columns))
    idx = legacy.index.intersection(v2.index)
    a = legacy.loc[idx, cols].astype(float)
    b = v2.loc[idx, cols].astype(float)

    a_ok = (legacy_valid.loc[idx, cols].fillna(False) if legacy_valid is not None else a.notna()).to_numpy()
    b_ok = (v2_valid.loc[idx, cols].fillna(False) if v2_valid is not None else b.notna()).to_numpy()
    both = a_ok & b_ok

    abs_bps = ((b - a).abs() * 10_000.0).to_numpy()
    abs_bps = np.where(both, abs_bps, np.nan)

    altered = np.greater(abs_bps, float(threshold_bps), where=~np.isnan(abs_bps), out=np.zeros_like(abs_bps, dtype=bool))
    n_both = int(both.sum())
    n_altered = int(altered.sum())
    altered_vals = abs_bps[altered]

    return {
        "threshold_bps": float(threshold_bps),
        "n_comparable_days": n_both,
        "n_legacy_only_days": int((a_ok & ~b_ok).sum()),
        "n_v2_only_days": int((~a_ok & b_ok).sum()),
        "n_altered_days": n_altered,
        "pct_altered": 100.0 * n_altered / n_both if n_both else np.nan,
        "mean_abs_diff_bps_altered": float(np.mean(altered_vals)) if n_altered else 0.0,
        "median_abs_diff_bps_altered": float(np.median(altered_vals)) if n_altered else 0.0,
        "max_abs_diff_bps": float(np.nanmax(abs_bps)) if n_both else np.nan,
        "n_tickers_compared": len(cols),
    }


def largest_return_differences(
    legacy: pd.DataFrame,
    v2: pd.DataFrame,
    *,
    legacy_valid: pd.DataFrame | None = None,
    v2_valid: pd.DataFrame | None = None,
    top_n: int = 25,
) -> pd.DataFrame:
    """The ticker-days on which the two tapes disagree most.

    Worth writing out rather than summarising: the largest disagreements are
    where a mechanical explanation (a delisting, a distribution, an adjustment
    factor) is visible, and a summary statistic hides which.
    """
    cols = sorted(set(legacy.columns) & set(v2.columns))
    idx = legacy.index.intersection(v2.index)
    a = legacy.loc[idx, cols].astype(float)
    b = v2.loc[idx, cols].astype(float)

    a_ok = legacy_valid.loc[idx, cols].fillna(False) if legacy_valid is not None else a.notna()
    b_ok = v2_valid.loc[idx, cols].fillna(False) if v2_valid is not None else b.notna()

    diff = (b - a).where(a_ok & b_ok)
    stacked = diff.stack().rename("diff").reset_index()
    stacked.columns = ["date", "ticker", "diff"]
    stacked["abs_bps"] = stacked["diff"].abs() * 10_000.0
    stacked = stacked.sort_values("abs_bps", ascending=False).head(int(top_n))

    stacked["legacy_return"] = [a.loc[d, t] for d, t in zip(stacked["date"], stacked["ticker"])]
    stacked["v2_return"] = [b.loc[d, t] for d, t in zip(stacked["date"], stacked["ticker"])]
    return stacked.reset_index(drop=True)

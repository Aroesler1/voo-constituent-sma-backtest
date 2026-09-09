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

The frozen cache records the historical CIZ daily-table assumption. A new live
pull does not reuse that assumption: it requires the approved metadata query's
verified daily source name in the same provenance sidecar as terminal outcomes.

Two deliberate choices keep the comparison clean:

1. The ticker-to-PERMNO resolution comes from the *legacy* name tables in both
   runs, via ``data_loader._resolve_crsp_name_history``. CIZ ships its own
   ``stocknames_v2``, and using it would mix a universe change into what is
   meant to be a price change.
2. Normalisation runs through ``data_loader._normalize_crsp_daily`` after the
   CIZ columns are mapped onto the legacy raw names, so adjustment factors,
   de-duplication and return conventions stay aligned.

Terminal-return semantics are not inferred from a table name or one observed
security. CRSP documents ``StkDelists.DelDlyDt`` as the date ``DelRet`` is
stored in the main daily time series. The entitled sources were verified as
``crsp.dsf_v2`` and ``crsp.stkdelists``. ``DlyRet`` therefore already includes
the terminal outcome and must not be compounded into the prior day. Frozen
caches built through legacy name-end filtering can omit CRSP's post-delist
daily row, so the verified sidecar restores that row at ``DelDlyDt``. Unknown
semantics fail closed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from contextlib import suppress
from pathlib import Path
from typing import Any

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

LEGACY_V2_CACHE_DIRNAME = "crsp_v2"
V2_CACHE_DIRNAME = "crsp_v2_versioned"
V2_RAW_CACHE_VERSION = "raw-v2"
V2_TERMINAL_CACHE_VERSION = "terminal-v3"
TERMINAL_SEMANTICS = {"separate_from_dlyret", "included_in_dlyret"}

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


def _v2_cache_dir(config: BacktestConfig, version: str = V2_RAW_CACHE_VERSION) -> Path:
    path = Path(config.CACHE_DIR) / V2_CACHE_DIRNAME / version
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(
    config: BacktestConfig,
    ticker: str,
    start: str,
    end: str,
    *,
    version: str = V2_RAW_CACHE_VERSION,
) -> Path:
    return _v2_cache_dir(config, version) / (
        f"crspv2_{_normalize_ticker(ticker)}_{start}_{end}.parquet"
    )


def _legacy_cache_path(config: BacktestConfig, ticker: str, start: str, end: str) -> Path:
    """Read-only fallback for the frozen, unversioned CIZ cache."""
    return Path(config.CACHE_DIR) / LEGACY_V2_CACHE_DIRNAME / (
        f"crspv2_{_normalize_ticker(ticker)}_{start}_{end}.parquet"
    )


def _covering_cache_paths(
    config: BacktestConfig,
    ticker: str,
    start: str,
    end: str,
) -> list[Path]:
    """Return exact then frozen superset caches without modifying old files."""
    exact = [
        _cache_path(config, ticker, start, end),
        _legacy_cache_path(config, ticker, start, end),
    ]
    wanted_start = pd.Timestamp(start)
    wanted_end = pd.Timestamp(end)
    pattern = re.compile(
        r"^crspv2_.+_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$"
    )
    covering: list[tuple[pd.Timedelta, Path]] = []
    for directory in (
        _v2_cache_dir(config, V2_RAW_CACHE_VERSION),
        Path(config.CACHE_DIR) / LEGACY_V2_CACHE_DIRNAME,
    ):
        for path in directory.glob(f"crspv2_{_normalize_ticker(ticker)}_*.parquet"):
            if path in exact:
                continue
            match = pattern.match(path.name)
            if not match:
                continue
            cached_start, cached_end = map(pd.Timestamp, match.groups())
            if cached_start <= wanted_start and cached_end >= wanted_end:
                excess = (wanted_start - cached_start) + (cached_end - wanted_end)
                covering.append((excess, path))
    return [*exact, *[path for _, path in sorted(covering, key=lambda item: item[0])]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_terminal_outcomes(
    extract_path: str | Path,
    metadata_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a terminal-return extract only when its semantics are documented."""
    extract = Path(extract_path)
    sidecar = Path(metadata_path)
    with sidecar.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    semantics = metadata.get("return_semantics")
    if semantics not in TERMINAL_SEMANTICS:
        raise ValueError(
            "CIZ terminal metadata must state return_semantics as "
            "'separate_from_dlyret' or 'included_in_dlyret'."
        )
    if metadata.get("source_table_verified") is not True:
        raise ValueError("CIZ terminal source table has not been entitlement-verified.")
    if metadata.get("daily_source_table_verified") is not True:
        raise ValueError("CIZ daily source table has not been entitlement-verified.")
    for field in (
        "source_product",
        "source_table",
        "daily_source_product",
        "daily_source_table",
        "query_text_sha256",
    ):
        if not str(metadata.get(field, "")).strip():
            raise ValueError(f"CIZ terminal metadata is missing {field}.")
    if not re.fullmatch(
        r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+",
        metadata["daily_source_table"],
    ):
        raise ValueError("Verified CIZ daily source must be a schema-qualified identifier.")

    actual_hash = _sha256_file(extract)
    if metadata.get("extract_sha256") != actual_hash:
        raise ValueError("CIZ terminal extract hash does not match its provenance sidecar.")

    outcomes = pd.read_csv(extract)
    required = {"permno", "event_date", "terminal_return"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"CIZ terminal extract is missing columns: {sorted(missing)}")
    selected = ["permno", "event_date", "terminal_return"]
    if "legacy_event_date" in outcomes.columns:
        selected.append("legacy_event_date")
    outcomes = outcomes.loc[:, selected].copy()
    outcomes["permno"] = pd.to_numeric(outcomes["permno"], errors="raise").astype("int64")
    outcomes["event_date"] = pd.to_datetime(outcomes["event_date"], errors="raise")
    if "legacy_event_date" in outcomes.columns:
        outcomes["legacy_event_date"] = pd.to_datetime(
            outcomes["legacy_event_date"], errors="coerce"
        )
    outcomes["terminal_return"] = pd.to_numeric(
        outcomes["terminal_return"], errors="raise"
    ).astype(float)
    if outcomes.duplicated(["permno", "event_date"]).any():
        raise ValueError("CIZ terminal extract has duplicate PERMNO/event-date keys.")
    if (outcomes["terminal_return"] < -1.0).any():
        raise ValueError("CIZ terminal returns must be decimal simple returns no lower than -1.")
    return outcomes.sort_values(["permno", "event_date"]).reset_index(drop=True), metadata


def apply_terminal_outcomes(
    normalized: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    return_semantics: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Mark terminal sessions and apply or restore the outcome exactly once.

    CIZ stores its delisting return in the main daily series on ``DelDlyDt``,
    conventionally the trading day after delisting. Frozen caches built with
    legacy name-end filtering can omit that post-delist row. For
    ``included_in_dlyret``, a missing row is restored from ``StkDelists.DelRet``
    without compounding it into the prior day's return.
    """
    if return_semantics not in TERMINAL_SEMANTICS:
        raise ValueError("Unknown CIZ terminal-return semantics; refusing to infer them.")

    out = normalized.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["permno"] = pd.to_numeric(out["permno"], errors="raise").astype("int64")
    events = outcomes.rename(columns={"event_date": "date"}).copy()
    events["date"] = pd.to_datetime(events["date"], errors="raise")
    events["permno"] = pd.to_numeric(events["permno"], errors="raise").astype("int64")
    if events.duplicated(["permno", "date"]).any():
        raise ValueError("Terminal outcomes must be unique by PERMNO and date.")
    if start is not None:
        events = events[events["date"] >= pd.Timestamp(start)]
    if end is not None:
        events = events[events["date"] <= pd.Timestamp(end)]
    events = events[events["permno"].isin(out["permno"].unique())]
    events = events[events["terminal_return"].notna()].copy()
    missing_events = events.iloc[0:0].copy()

    if return_semantics == "included_in_dlyret" and not events.empty:
        existing = pd.MultiIndex.from_frame(out[["permno", "date"]])
        missing_events = events[
            ~pd.MultiIndex.from_frame(events[["permno", "date"]]).isin(existing)
        ]
        missing_events = missing_events[
            ~missing_events["date"].isin(out["date"])
        ]
        appended_by_date: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Series]] = {}
        for event in missing_events.itertuples(index=False):
            prior = out[
                (out["permno"] == event.permno) & (out["date"] < event.date)
            ].sort_values("date")
            if prior.empty:
                continue
            row = prior.iloc[-1].copy()
            prior_date = pd.Timestamp(row["date"])
            row["date"] = event.date
            row["volume"] = 0.0
            if "adjVolume" in row.index:
                row["adjVolume"] = 0.0
            row["total_return"] = event.terminal_return
            current = appended_by_date.get(event.date)
            if current is None or prior_date > current[0]:
                appended_by_date[event.date] = (prior_date, row)
        if appended_by_date:
            for _, row in appended_by_date.values():
                out.loc[len(out)] = row

    out = out.drop(
        columns=["is_terminal_session", "terminal_return_applied", "terminal_return"],
        errors="ignore",
    ).merge(events, on=["permno", "date"], how="left", validate="many_to_one")
    terminal = out["terminal_return"].notna()
    out["is_terminal_session"] = terminal
    out["terminal_return_applied"] = False
    if return_semantics == "separate_from_dlyret":
        daily = pd.to_numeric(out["total_return"], errors="coerce")
        event = out["terminal_return"]
        out.loc[terminal, "total_return"] = np.where(
            daily.loc[terminal].notna(),
            (1.0 + daily.loc[terminal]) * (1.0 + event.loc[terminal]) - 1.0,
            event.loc[terminal],
        )
        out.loc[terminal, "terminal_return_applied"] = True
    elif return_semantics == "included_in_dlyret":
        # Existing daily rows already carry DlyRet. Restored rows receive the
        # same DelRet value by the documented CIZ daily convention.
        restored_keys = pd.MultiIndex.from_frame(
            missing_events[["permno", "date"]]
        )
        row_keys = pd.MultiIndex.from_frame(out[["permno", "date"]])
        restored = terminal & row_keys.isin(restored_keys)
        out.loc[restored, "total_return"] = out.loc[restored, "terminal_return"]
        out.loc[restored, "terminal_return_applied"] = True
    return out.sort_values(["date", "permno"]).reset_index(drop=True)


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
    # Whether terminal outcomes are separate from DlyRet is a product-semantic
    # question, not something this daily table can answer. Keep the daily
    # normalisation unbridged until a verified sidecar is supplied.
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
    terminal_source: tuple[pd.DataFrame, dict[str, Any]] | None = None
    corrected_version: str | None = None
    if cfg.CIZ_TERMINAL_RETURNS_PATH and cfg.CIZ_TERMINAL_METADATA_PATH:
        terminal_source = load_verified_terminal_outcomes(
            cfg.CIZ_TERMINAL_RETURNS_PATH,
            cfg.CIZ_TERMINAL_METADATA_PATH,
        )
        source_hash = terminal_source[1]["extract_sha256"][:12]
        corrected_version = f"{V2_TERMINAL_CACHE_VERSION}-{source_hash}"

    for ticker in requested:
        if corrected_version is not None:
            corrected_path = _cache_path(
                cfg, ticker, start, end, version=corrected_version
            )
            if corrected_path.exists():
                cached = pd.read_parquet(corrected_path)
                if not cached.empty:
                    out[ticker] = cached
                    continue

        for path in _covering_cache_paths(cfg, ticker, start, end):
            if not path.exists():
                continue
            cached = pd.read_parquet(path)
            if not cached.empty:
                cached["date"] = pd.to_datetime(cached["date"], errors="raise")
                cached = cached[
                    (cached["date"] >= pd.Timestamp(start))
                    & (cached["date"] <= pd.Timestamp(end))
                ].copy()
            if not cached.empty:
                if terminal_source is not None:
                    cached = apply_terminal_outcomes(
                        cached,
                        terminal_source[0],
                        return_semantics=terminal_source[1]["return_semantics"],
                        start=start,
                        end=end,
                    )
                    cached.to_parquet(corrected_path, index=False)
                else:
                    cached["is_terminal_session"] = False
                    cached["terminal_return_applied"] = False
                out[ticker] = cached
                break
        if ticker in out:
            continue
        missing.append(ticker)

    if not missing:
        LOGGER.info("CRSP v2: served %s tickers from cache.", len(out))
        return out

    if cfg.OFFLINE_MODE:
        LOGGER.warning(
            "CRSP v2 offline mode: %s unresolved ticker(s) remain missing from cache; "
            "no connection was attempted.",
            len(missing),
        )
        return out
    if terminal_source is None:
        raise ValueError(
            "A new CIZ pull requires entitlement-verified daily and terminal source metadata."
        )

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
                FROM {terminal_source[1]["daily_source_table"]} AS d
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
                if terminal_source is not None:
                    normalized = apply_terminal_outcomes(
                        normalized,
                        terminal_source[0],
                        return_semantics=terminal_source[1]["return_semantics"],
                        start=start,
                        end=end,
                    )
                    normalized.to_parquet(
                        _cache_path(
                            cfg,
                            ticker,
                            start,
                            end,
                            version=str(corrected_version),
                        ),
                        index=False,
                    )
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
    legacy_permno: pd.DataFrame | None = None,
    v2_permno: pd.DataFrame | None = None,
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
    both_coverage = a_ok & b_ok
    if (legacy_permno is None) != (v2_permno is None):
        raise ValueError("Both PERMNO panels are required for an identity-matched comparison.")
    if legacy_permno is not None and v2_permno is not None:
        legacy_ids = legacy_permno.loc[idx, cols].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float, na_value=np.nan)
        v2_ids = v2_permno.loc[idx, cols].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float, na_value=np.nan)
        identity_match = (
            pd.notna(legacy_ids) & pd.notna(v2_ids) & (legacy_ids == v2_ids)
        )
    else:
        identity_match = np.ones_like(both_coverage, dtype=bool)
    both = both_coverage & identity_match

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
        "n_identity_mismatch_days": int((both_coverage & ~identity_match).sum()),
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
    legacy_permno: pd.DataFrame | None = None,
    v2_permno: pd.DataFrame | None = None,
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

    if (legacy_permno is None) != (v2_permno is None):
        raise ValueError("Both PERMNO panels are required for an identity-matched comparison.")
    same_identity = pd.DataFrame(True, index=idx, columns=cols)
    if legacy_permno is not None and v2_permno is not None:
        legacy_ids = legacy_permno.loc[idx, cols].apply(pd.to_numeric, errors="coerce")
        v2_ids = v2_permno.loc[idx, cols].apply(pd.to_numeric, errors="coerce")
        same_identity = legacy_ids.notna() & v2_ids.notna() & legacy_ids.eq(v2_ids)

    diff = (b - a).where(a_ok & b_ok & same_identity)
    stacked = diff.stack().rename("diff").reset_index()
    stacked.columns = ["date", "ticker", "diff"]
    stacked["abs_bps"] = stacked["diff"].abs() * 10_000.0
    stacked = stacked.sort_values("abs_bps", ascending=False).head(int(top_n))

    stacked["legacy_return"] = [a.loc[d, t] for d, t in zip(stacked["date"], stacked["ticker"])]
    stacked["v2_return"] = [b.loc[d, t] for d, t in zip(stacked["date"], stacked["ticker"])]
    return stacked.reset_index(drop=True)

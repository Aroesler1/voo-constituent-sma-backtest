"""Data fetching, caching, and snapshot management."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from config import BacktestConfig, load_config

LOGGER = logging.getLogger(__name__)
CRSP_SNAPSHOT_SCHEMA_VERSION = 3
CRSP_COMMON_SHARE_CODES = (10, 11, 12, 18, 40, 41, 42, 48, 70, 71, 72)


def _parse_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _request_json_with_retries(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = 3,
    backoff_base: int = 2,
) -> Any:
    """Issue HTTP GET with retry and JSON parsing."""
    session = requests.Session()
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            status = exc.response.status_code if isinstance(exc, requests.HTTPError) and exc.response is not None else None
            retryable = status in {408, 409, 425, 429} or (status is not None and status >= 500) or status is None
            if not retryable:
                break
            if attempt >= retries:
                break
            delay = backoff_base ** attempt
            LOGGER.warning(
                "Request failed (attempt %s/%s): %s. Retrying in %ss",
                attempt,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"Failed request after {retries} attempts: {url}") from last_err


def _request_text_with_retries(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = 3,
    backoff_base: int = 2,
) -> str:
    """Issue HTTP GET with retry and return text payload."""
    session = requests.Session()
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_err = exc
            status = exc.response.status_code if isinstance(exc, requests.HTTPError) and exc.response is not None else None
            retryable = status in {408, 409, 425, 429} or (status is not None and status >= 500) or status is None
            if not retryable:
                break
            if attempt >= retries:
                break
            delay = backoff_base ** attempt
            LOGGER.warning(
                "Text request failed (attempt %s/%s): %s. Retrying in %ss",
                attempt,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"Failed request after {retries} attempts: {url}") from last_err


def _ensure_cache_dirs(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "snapshots").mkdir(parents=True, exist_ok=True)


def _coverage_ok(meta: dict[str, Any], start: str, end: str) -> bool:
    req_start = _parse_date(start)
    req_end = _parse_date(end)
    snap_start = _parse_date(meta["requested_start"])
    snap_end = _parse_date(meta["requested_end"])
    return snap_start <= req_start and snap_end >= req_end


def _cache_filename(vendor: str, ticker: str, start: str, end: str) -> str:
    return f"{vendor.lower()}_{ticker.upper()}_{start}_{end}.parquet"


def _extract_end_from_cache_name(path: Path) -> pd.Timestamp | None:
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    try:
        return _parse_date(parts[-1])
    except Exception:
        return None


def _find_fresh_cache(
    cache_dir: Path,
    vendor: str,
    ticker: str,
    start: str,
    end: str,
) -> Path | None:
    pattern = f"{vendor.lower()}_{ticker.upper()}_{start}_*.parquet"
    req_end = _parse_date(end)
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for file_path in cache_dir.glob(pattern):
        cache_end = _extract_end_from_cache_name(file_path)
        if cache_end is not None and cache_end >= req_end:
            candidates.append((cache_end, file_path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_df(df: pd.DataFrame) -> str:
    if df.empty:
        return _sha256_bytes(b"")
    sort_cols = ["date"] if "date" in df.columns else list(df.columns)
    normalized = df.sort_values(sort_cols).to_csv(index=False).encode("utf-8")
    return _sha256_bytes(normalized)


def _serialize_json_bytes(raw_payload: Any) -> bytes:
    return json.dumps(raw_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _normalize_ticker(value: str) -> str:
    return str(value).strip().upper().replace(".", "-")


def build_snapshot_manifest_row(
    vendor: str,
    ticker: str,
    snapshot_id: str | None,
    fetched_at_utc: str | None,
    sha256: str | None,
    requested_start: str,
    requested_end: str,
) -> dict[str, Any]:
    """Build one manifest row for run metadata."""
    return {
        "vendor": vendor,
        "ticker": ticker,
        "snapshot_id": snapshot_id,
        "fetched_at_utc": fetched_at_utc,
        "sha256": sha256,
        "requested_start": requested_start,
        "requested_end": requested_end,
    }


def resolve_snapshot(
    vendor: str,
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | Path,
    as_of_utc: str,
    lock_id: str | None = None,
) -> dict[str, Any] | None:
    """Resolve a snapshot satisfying point-in-time constraints."""
    base = Path(cache_dir) / "snapshots" / vendor.lower() / ticker.upper()
    if not base.exists():
        return None

    if lock_id:
        meta_path = base / lock_id / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Snapshot lock id '{lock_id}' not found for {vendor}:{ticker}.")
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if not _coverage_ok(meta, start, end):
            raise ValueError(
                f"Snapshot lock id '{lock_id}' does not cover requested range {start}..{end}."
            )
        return meta

    cutoff = pd.Timestamp(as_of_utc)
    candidates: list[dict[str, Any]] = []
    for meta_path in base.glob("*/metadata.json"):
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if vendor.lower() == "crsp" and int(meta.get("schema_version", 0)) < CRSP_SNAPSHOT_SCHEMA_VERSION:
                continue
            if not _coverage_ok(meta, start, end):
                continue
            fetched = pd.Timestamp(meta["fetched_at_utc"])
            if fetched <= cutoff:
                candidates.append(meta)
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda m: pd.Timestamp(m["fetched_at_utc"]))
    return candidates[-1]


def write_snapshot(
    vendor: str,
    ticker: str,
    requested_start: str,
    requested_end: str,
    raw_payload: Any,
    normalized_df: pd.DataFrame,
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Persist immutable snapshot artifacts."""
    fetched_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    raw_bytes = _serialize_json_bytes(raw_payload)
    raw_sha256 = _sha256_bytes(raw_bytes)
    normalized_sha256 = _sha256_df(normalized_df)
    base_dir = Path(cache_dir) / "snapshots" / vendor.lower() / ticker.upper()
    if base_dir.exists():
        for meta_path in base_dir.glob("*/metadata.json"):
            try:
                with meta_path.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                if (
                    meta.get("requested_start") == requested_start
                    and meta.get("requested_end") == requested_end
                    and meta.get("raw_sha256") == raw_sha256
                    and meta.get("normalized_sha256") == normalized_sha256
                ):
                    return meta
            except Exception:
                continue

    snapshot_id = f"{pd.Timestamp(fetched_at_utc).strftime('%Y%m%dT%H%M%SZ')}_{raw_sha256[:12]}"

    snap_dir = base_dir / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=False)

    raw_path = snap_dir / "raw.json.gz"
    with gzip.open(raw_path, "wb") as gz:
        gz.write(raw_bytes)

    normalized_path = snap_dir / "normalized.parquet"
    normalized_df.to_parquet(normalized_path, index=False)

    min_date = normalized_df["date"].min() if "date" in normalized_df.columns else None
    max_date = normalized_df["date"].max() if "date" in normalized_df.columns else None

    meta = {
        "vendor": vendor,
        "ticker": ticker,
        "snapshot_id": snapshot_id,
        "schema_version": CRSP_SNAPSHOT_SCHEMA_VERSION if vendor.lower() == "crsp" else 1,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "fetched_at_utc": fetched_at_utc,
        "raw_sha256": raw_sha256,
        "normalized_sha256": normalized_sha256,
        "row_count": int(len(normalized_df)),
        "min_date": pd.Timestamp(min_date).date().isoformat() if pd.notna(min_date) else None,
        "max_date": pd.Timestamp(max_date).date().isoformat() if pd.notna(max_date) else None,
        "normalized_path": str(normalized_path),
        "raw_path": str(raw_path),
    }
    if vendor.lower() == "crsp":
        meta["permno_resolution"] = "date_aware_name_history_v2"

    with (snap_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    return meta


def load_snapshot(meta: dict[str, Any]) -> pd.DataFrame:
    """Load normalized DataFrame from snapshot metadata."""
    df = pd.read_parquet(meta["normalized_path"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        if _is_price_like_frame(df):
            df = _deduplicate_normalized_df(df)
        df = df.sort_values("date")
    if str(meta.get("vendor", "")).lower() == "crsp":
        if "adjOpen" not in df.columns and "open" in df.columns:
            df["adjOpen"] = df["open"]
        if "adjHigh" not in df.columns and "high" in df.columns:
            df["adjHigh"] = df["high"]
        if "adjLow" not in df.columns and "low" in df.columns:
            df["adjLow"] = df["low"]
        if "adjClose" not in df.columns and "adjusted_close" in df.columns:
            df["adjClose"] = df["adjusted_close"]
        if "adjVolume" not in df.columns and "volume" in df.columns:
            df["adjVolume"] = df["volume"]
    df = df.reset_index(drop=True)
    df.attrs["snapshot_id"] = meta.get("snapshot_id")
    df.attrs["fetched_at_utc"] = meta.get("fetched_at_utc")
    df.attrs["sha256"] = meta.get("normalized_sha256")
    return df


def _finalize_and_log(
    df: pd.DataFrame,
    *,
    vendor: str,
    ticker: str,
    source_label: str,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{vendor} returned empty data for {ticker}.")
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
        dmin = df["date"].min().date().isoformat()
        dmax = df["date"].max().date().isoformat()
    else:
        df = df.reset_index(drop=True)
        dmin, dmax = "n/a", "n/a"
    LOGGER.info("%s %s (%s): rows=%s, range=%s..%s", vendor, ticker, source_label, len(df), dmin, dmax)
    return df


def _deduplicate_normalized_df(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate normalized price rows on date with deterministic tie-breaking."""
    if "date" not in df.columns or not df["date"].duplicated().any():
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    else:
        out["volume"] = 0.0
    if "permno" in out.columns:
        out["permno"] = pd.to_numeric(out["permno"], errors="coerce")
    else:
        out["permno"] = np.nan

    return (
        out.sort_values(["date", "volume", "permno"], ascending=[True, False, False])
        .drop_duplicates(subset=["date"], keep="first")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _is_price_like_frame(df: pd.DataFrame) -> bool:
    """Return whether a normalized frame looks like price history."""
    price_cols = {"adjusted_close", "close", "open", "high", "low", "permno"}
    return bool(price_cols.intersection(df.columns))


def _fetch_or_load_with_snapshot(
    *,
    vendor: str,
    ticker: str,
    start: str,
    end: str,
    live_fetch_fn,
    normalize_fn,
) -> pd.DataFrame:
    cfg: BacktestConfig = load_config()
    cache_dir = Path(cfg.CACHE_DIR)
    _ensure_cache_dirs(cache_dir)

    meta = resolve_snapshot(
        vendor=vendor,
        ticker=ticker,
        start=start,
        end=end,
        cache_dir=cache_dir,
        as_of_utc=cfg.SNAPSHOT_AS_OF_UTC,
        lock_id=cfg.SNAPSHOT_LOCK_ID,
    )
    if meta is not None:
        snap_df = load_snapshot(meta)
        if "date" in snap_df.columns:
            snap_df = snap_df[snap_df["date"] <= pd.Timestamp(end)].copy()
        if not snap_df.empty:
            return _finalize_and_log(
                snap_df,
                vendor=vendor,
                ticker=ticker,
                source_label=f"snapshot:{meta['snapshot_id']}",
            )
        if cfg.FREEZE_SNAPSHOTS and not cfg.ALLOW_LIVE_FETCH_WHEN_NO_SNAPSHOT:
            raise ValueError(
                f"Snapshot {meta.get('snapshot_id')} for {vendor}:{ticker} is empty and live fetch is disabled."
            )
        LOGGER.warning(
            "Ignoring empty snapshot %s for %s:%s and attempting live fetch.",
            meta.get("snapshot_id"),
            vendor,
            ticker,
        )

    if not cfg.FREEZE_SNAPSHOTS:
        cached_file = _find_fresh_cache(cache_dir, vendor, ticker, start, end)
        if cached_file is not None:
            cached = pd.read_parquet(cached_file)
            if "date" in cached.columns:
                cached["date"] = pd.to_datetime(cached["date"])
                cached = cached[cached["date"] <= pd.Timestamp(end)].copy()
            if not cached.empty:
                cached = _finalize_and_log(
                    cached,
                    vendor=vendor,
                    ticker=ticker,
                    source_label=f"cache:{cached_file.name}",
                )
                cached.attrs["snapshot_id"] = None
                cached.attrs["fetched_at_utc"] = None
                cached.attrs["sha256"] = _sha256_df(cached)
                return cached
            LOGGER.warning(
                "Ignoring empty cache file %s for %s:%s and attempting live fetch.",
                cached_file.name,
                vendor,
                ticker,
            )

    if cfg.FREEZE_SNAPSHOTS and not cfg.ALLOW_LIVE_FETCH_WHEN_NO_SNAPSHOT:
        raise RuntimeError(
            f"No eligible snapshot found for {vendor}:{ticker} under as-of {cfg.SNAPSHOT_AS_OF_UTC}, "
            "and live fetch is disabled."
        )

    raw_payload = live_fetch_fn()
    normalized = normalize_fn(raw_payload)
    if normalized.empty:
        raise ValueError(f"{vendor} returned empty response for {ticker}.")

    if "date" in normalized.columns:
        normalized = normalized[normalized["date"] <= pd.Timestamp(end)].copy()
        if _is_price_like_frame(normalized):
            normalized = _deduplicate_normalized_df(normalized)
        normalized = normalized.sort_values("date").reset_index(drop=True)
    else:
        normalized = normalized.reset_index(drop=True)

    if not cfg.FREEZE_SNAPSHOTS:
        cache_path = cache_dir / _cache_filename(vendor, ticker, start, end)
        normalized.to_parquet(cache_path, index=False)

    meta = write_snapshot(
        vendor=vendor,
        ticker=ticker,
        requested_start=start,
        requested_end=end,
        raw_payload=raw_payload,
        normalized_df=normalized,
        cache_dir=cache_dir,
    )
    normalized.attrs["snapshot_id"] = meta["snapshot_id"]
    normalized.attrs["fetched_at_utc"] = meta["fetched_at_utc"]
    normalized.attrs["sha256"] = meta["normalized_sha256"]

    return _finalize_and_log(normalized, vendor=vendor, ticker=ticker, source_label="live_fetch")


def fetch_eodhd(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch daily OHLCV data from EODHD."""
    if not api_key:
        raise ValueError("EODHD API key is missing.")

    endpoint = f"https://eodhd.com/api/eod/{ticker.upper()}.US"

    def _live_fetch() -> Any:
        return _request_json_with_retries(
            endpoint,
            params={"from": start, "to": end, "fmt": "json", "api_token": api_key},
        )

    def _normalize(raw_payload: Any) -> pd.DataFrame:
        if not isinstance(raw_payload, list) or len(raw_payload) == 0:
            raise ValueError("Malformed/empty EODHD response.")
        df = pd.DataFrame(raw_payload)
        expected_cols = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"EODHD response missing columns: {missing}")
        df = df[expected_cols].copy()
        df["source_vendor"] = "eodhd"
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=expected_cols)
        return df

    return _fetch_or_load_with_snapshot(
        vendor="eodhd",
        ticker=ticker,
        start=start,
        end=end,
        live_fetch_fn=_live_fetch,
        normalize_fn=_normalize,
    )


def _get_wrds_connection(
    *,
    username: str | None,
    password: str | None,
):
    """Create a WRDS connection for CRSP access."""
    try:
        import wrds  # type: ignore
    except ImportError as exc:
        raise ImportError("wrds package is not installed. Add it to the environment to use CRSP.") from exc

    kwargs: dict[str, Any] = {"autoconnect": True, "verbose": False}
    if username:
        kwargs["wrds_username"] = username
    if password:
        kwargs["wrds_password"] = password
    return wrds.Connection(**kwargs)


def _normalize_crsp_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw CRSP daily rows into the project's adjusted price schema."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
                "total_return",
                "retx",
                "permno",
            ]
        )

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    numeric_cols = [
        "open_raw",
        "high_raw",
        "low_raw",
        "close_raw",
        "volume_raw",
        "ret_raw",
        "retx_raw",
        "dlret_raw",
        "cfacpr",
        "cfacshr",
        "permno",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["cfacpr"] = out["cfacpr"].replace(0, np.nan).fillna(1.0)
    out["cfacshr"] = out["cfacshr"].replace(0, np.nan).fillna(1.0)
    adj_factor = 1.0 / out["cfacpr"]

    out["close"] = out["close_raw"].abs()
    out["open"] = out["open_raw"].abs() * adj_factor
    out["high"] = out["high_raw"].abs() * adj_factor
    out["low"] = out["low_raw"].abs() * adj_factor
    out["adjusted_close"] = out["close"] * adj_factor
    out["volume"] = out["volume_raw"].fillna(0.0) * out["cfacshr"]
    out["adjOpen"] = out["open"]
    out["adjHigh"] = out["high"]
    out["adjLow"] = out["low"]
    out["adjClose"] = out["adjusted_close"]
    out["adjVolume"] = out["volume"]
    out["retx"] = out["retx_raw"]
    out["source_vendor"] = "crsp"

    ret = out["ret_raw"]
    dlret = out["dlret_raw"]
    out["total_return"] = np.where(
        ret.notna() & dlret.notna(),
        (1.0 + ret) * (1.0 + dlret) - 1.0,
        np.where(ret.notna(), ret, dlret),
    )

    out["open"] = out["open"].fillna(out["adjusted_close"])
    out["high"] = out["high"].fillna(out["adjusted_close"])
    out["low"] = out["low"].fillna(out["adjusted_close"])

    normalized = out[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "adjOpen",
            "adjHigh",
            "adjLow",
            "adjClose",
            "adjVolume",
            "total_return",
            "retx",
            "permno",
            "source_vendor",
        ]
    ].dropna(subset=["date", "adjusted_close"])
    normalized["volume"] = pd.to_numeric(normalized["volume"], errors="coerce").fillna(0.0)
    normalized["permno"] = pd.to_numeric(normalized["permno"], errors="coerce")
    normalized = (
        normalized.sort_values(["date", "volume", "permno"], ascending=[True, False, False])
        .drop_duplicates(subset=["date"], keep="first")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return normalized


def _resolve_crsp_name_history(
    conn: Any,
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Resolve date-aware CRSP name history rows for requested tickers."""
    tickers_sql = ", ".join(f"'{ticker}'" for ticker in tickers)
    share_codes_sql = ", ".join(str(code) for code in CRSP_COMMON_SHARE_CODES)
    variants = [
        """
        SELECT
            UPPER(REPLACE(COALESCE(ticker, ''), '.', '-')) AS ticker,
            permno,
            namedt,
            nameenddt,
            shrcd,
            exchcd,
            comnam
        FROM crsp.stocknames
        WHERE UPPER(REPLACE(COALESCE(ticker, ''), '.', '-')) IN ({tickers_sql})
          AND shrcd IN ({share_codes_sql})
          AND COALESCE(nameenddt, DATE '9999-12-31') >= DATE '{start}'
          AND namedt <= DATE '{end}'
        ORDER BY ticker, nameenddt DESC, namedt DESC, permno DESC
        """,
        """
        SELECT
            UPPER(REPLACE(COALESCE(ticker, ''), '.', '-')) AS ticker,
            permno,
            namedt,
            nameendt AS nameenddt,
            shrcd,
            exchcd,
            comnam
        FROM crsp.stocknames
        WHERE UPPER(REPLACE(COALESCE(ticker, ''), '.', '-')) IN ({tickers_sql})
          AND shrcd IN ({share_codes_sql})
          AND COALESCE(nameendt, DATE '9999-12-31') >= DATE '{start}'
          AND namedt <= DATE '{end}'
        ORDER BY ticker, nameendt DESC, namedt DESC, permno DESC
        """,
        """
        SELECT
            UPPER(REPLACE(COALESCE(ticker, ''), '.', '-')) AS ticker,
            permno,
            namedt,
            nameenddt,
            shrcd,
            exchcd,
            comnam
        FROM crsp.msenames
        WHERE UPPER(REPLACE(COALESCE(ticker, ''), '.', '-')) IN ({tickers_sql})
          AND shrcd IN ({share_codes_sql})
          AND COALESCE(nameenddt, DATE '9999-12-31') >= DATE '{start}'
          AND namedt <= DATE '{end}'
        ORDER BY ticker, nameenddt DESC, namedt DESC, permno DESC
        """,
        """
        SELECT
            UPPER(REPLACE(COALESCE(ticker, ''), '.', '-')) AS ticker,
            permno,
            namedt,
            nameendt AS nameenddt,
            shrcd,
            exchcd,
            comnam
        FROM crsp.msenames
        WHERE UPPER(REPLACE(COALESCE(ticker, ''), '.', '-')) IN ({tickers_sql})
          AND shrcd IN ({share_codes_sql})
          AND COALESCE(nameendt, DATE '9999-12-31') >= DATE '{start}'
          AND namedt <= DATE '{end}'
        ORDER BY ticker, nameendt DESC, namedt DESC, permno DESC
        """,
    ]

    names_df: pd.DataFrame | None = None
    last_err: Exception | None = None
    for sql in variants:
        try:
            names_df = conn.raw_sql(
                sql.format(
                    tickers_sql=tickers_sql,
                    share_codes_sql=share_codes_sql,
                    start=start,
                    end=end,
                ),
                date_cols=["namedt", "nameenddt"],
            )
            break
        except Exception as exc:
            last_err = exc
            continue

    if names_df is None:
        raise RuntimeError("Failed to resolve CRSP name history from name tables.") from last_err

    if names_df.empty:
        return pd.DataFrame(
            columns=["ticker", "permno", "namedt", "nameenddt", "shrcd", "exchcd", "comnam"]
        )

    names_df["ticker"] = names_df["ticker"].map(_normalize_ticker)
    names_df["nameenddt"] = pd.to_datetime(names_df["nameenddt"], errors="coerce").fillna(pd.Timestamp("2100-12-31"))
    names_df["namedt"] = pd.to_datetime(names_df["namedt"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    names_df["permno"] = pd.to_numeric(names_df["permno"], errors="coerce")
    names_df["shrcd"] = pd.to_numeric(names_df["shrcd"], errors="coerce")
    names_df["exchcd"] = pd.to_numeric(names_df["exchcd"], errors="coerce").fillna(99)
    names_df = names_df.dropna(subset=["ticker", "permno"]).copy()
    names_df["permno"] = names_df["permno"].astype(int)
    names_df = names_df.sort_values(
        ["ticker", "namedt", "nameenddt", "exchcd", "permno"],
        ascending=[True, True, False, True, False],
    ).drop_duplicates(subset=["ticker", "permno", "namedt", "nameenddt"], keep="first")
    return names_df.reset_index(drop=True)


def fetch_crsp_batch_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    username: str | None = None,
    password: str | None = None,
    api_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch CRSP daily data via WRDS in batches, with snapshot reuse and per-ticker caching."""
    cfg = load_config()
    cache_dir = Path(cfg.CACHE_DIR)
    _ensure_cache_dirs(cache_dir)

    requested = sorted({_normalize_ticker(t) for t in tickers if str(t).strip()})
    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for ticker in requested:
        meta = resolve_snapshot(
            vendor="crsp",
            ticker=ticker,
            start=start,
            end=end,
            cache_dir=cache_dir,
            as_of_utc=cfg.SNAPSHOT_AS_OF_UTC,
            lock_id=cfg.SNAPSHOT_LOCK_ID,
        )
        if meta is None:
            missing.append(ticker)
            continue
        snap_df = load_snapshot(meta)
        snap_df = snap_df[snap_df["date"] <= pd.Timestamp(end)].copy()
        if snap_df.empty:
            missing.append(ticker)
            continue
        out[ticker] = _finalize_and_log(
            snap_df,
            vendor="crsp",
            ticker=ticker,
            source_label=f"snapshot:{meta['snapshot_id']}",
        )

    if not missing:
        return out

    user = username or cfg.WRDS_USERNAME or cfg.CRSP_USERNAME
    secret = password or cfg.WRDS_PASSWORD or api_key or cfg.CRSP_API_KEY
    if not user and not Path.home().joinpath(".pgpass").exists():
        LOGGER.warning("CRSP requested but WRDS username is missing; falling back for %s tickers.", len(missing))
        return out

    conn = _get_wrds_connection(username=user, password=secret)
    try:
        for batch_start in range(0, len(missing), int(cfg.CRSP_BATCH_SIZE)):
            chunk = missing[batch_start : batch_start + int(cfg.CRSP_BATCH_SIZE)]
            name_history = _resolve_crsp_name_history(conn, chunk, start, end)
            if name_history.empty:
                LOGGER.warning("CRSP permno resolution returned no matches for chunk starting with %s.", chunk[0])
                continue

            permnos = sorted(set(name_history["permno"].tolist()))
            permno_sql = ", ".join(str(p) for p in permnos)
            sql = f"""
                SELECT
                    d.date,
                    d.permno,
                    d.openprc AS open_raw,
                    d.askhi AS high_raw,
                    d.bidlo AS low_raw,
                    d.prc AS close_raw,
                    d.vol AS volume_raw,
                    d.ret AS ret_raw,
                    d.retx AS retx_raw,
                    d.cfacpr,
                    d.cfacshr,
                    dl.dlret AS dlret_raw
                FROM crsp.dsf AS d
                LEFT JOIN crsp.dsedelist AS dl
                  ON d.permno = dl.permno
                 AND d.date = dl.dlstdt
                WHERE d.date BETWEEN '{start}' AND '{end}'
                  AND d.permno IN ({permno_sql})
                ORDER BY d.permno, d.date
            """
            raw_df = conn.raw_sql(sql, date_cols=["date"])

            if raw_df.empty:
                LOGGER.warning("CRSP returned no rows for chunk starting with %s.", chunk[0])
                continue

            merged = raw_df.merge(
                name_history,
                on="permno",
                how="inner",
                suffixes=("", "_name"),
            )
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged = merged[
                (merged["date"] >= merged["namedt"])
                & (merged["date"] <= merged["nameenddt"])
            ].copy()
            if merged.empty:
                LOGGER.warning("CRSP name-history merge produced no valid rows for chunk starting with %s.", chunk[0])
                continue

            share_priority = {
                11: 0,
                10: 1,
                18: 2,
                12: 3,
                41: 4,
                40: 5,
                42: 6,
                48: 7,
                71: 8,
                70: 9,
                72: 10,
            }
            merged["share_priority"] = merged["shrcd"].map(share_priority).fillna(99).astype(int)
            merged = merged.sort_values(
                ["ticker", "date", "share_priority", "namedt", "nameenddt", "exchcd", "permno"],
                ascending=[True, True, True, False, False, True, False],
            ).drop_duplicates(subset=["ticker", "date"], keep="first")

            for ticker, raw_group in merged.groupby("ticker", sort=False):
                normalized = _normalize_crsp_daily(raw_group)
                if normalized.empty:
                    continue

                if not cfg.FREEZE_SNAPSHOTS:
                    cache_path = cache_dir / _cache_filename("crsp", ticker, start, end)
                    normalized.to_parquet(cache_path, index=False)

                meta = write_snapshot(
                    vendor="crsp",
                    ticker=ticker,
                    requested_start=start,
                    requested_end=end,
                    raw_payload=raw_group.to_dict(orient="records"),
                    normalized_df=normalized,
                    cache_dir=cache_dir,
                )
                normalized.attrs["snapshot_id"] = meta["snapshot_id"]
                normalized.attrs["fetched_at_utc"] = meta["fetched_at_utc"]
                normalized.attrs["sha256"] = meta["normalized_sha256"]
                out[ticker] = _finalize_and_log(
                    normalized,
                    vendor="crsp",
                    ticker=ticker,
                    source_label="live_fetch",
                )
    finally:
        with suppress(Exception):
            conn.close()

    unresolved = sorted(set(missing) - set(out))
    if unresolved:
        LOGGER.info("CRSP unresolved tickers in requested range: %s", len(unresolved))

    return out


def _normalize_membership_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize membership CSV into event rows (date,ticker,action)."""
    cols = {c.lower(): c for c in df.columns}

    if {"date", "ticker", "action"}.issubset(set(cols.keys())):
        out = df[[cols["date"], cols["ticker"], cols["action"]]].copy()
        out.columns = ["date", "ticker", "action"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["ticker"] = out["ticker"].map(_normalize_ticker)
        out["action"] = out["action"].astype(str).str.lower().str.strip()
        out = out[out["action"].isin(["add", "remove"])]
        return out.dropna(subset=["date", "ticker", "action"]).sort_values("date").reset_index(drop=True)

    if {"start_date", "end_date", "ticker"}.issubset(set(cols.keys())):
        tmp = df[[cols["start_date"], cols["end_date"], cols["ticker"]]].copy()
        tmp.columns = ["start_date", "end_date", "ticker"]
        tmp["start_date"] = pd.to_datetime(tmp["start_date"], errors="coerce")
        tmp["end_date"] = pd.to_datetime(tmp["end_date"], errors="coerce")
        tmp["ticker"] = tmp["ticker"].map(_normalize_ticker)
        tmp = tmp.dropna(subset=["start_date", "ticker"])

        add_rows = tmp[["start_date", "ticker"]].rename(columns={"start_date": "date"})
        add_rows["action"] = "add"

        rem_rows = tmp.dropna(subset=["end_date"])[["end_date", "ticker"]].rename(columns={"end_date": "date"})
        rem_rows["date"] = rem_rows["date"] + pd.offsets.BDay(1)
        rem_rows["action"] = "remove"

        out = pd.concat([add_rows, rem_rows], axis=0).sort_values("date").reset_index(drop=True)
        return out

    raise ValueError(
        "Membership file must contain either (date,ticker,action) or (start_date,end_date,ticker)."
    )


def fetch_sp500_membership_history_public(start: str, end: str) -> pd.DataFrame:
    """Fetch S&P 500 membership history proxy as event rows.

    Priority:
      1) Local CSV path from config.
      2) Optional public URL from config.

    Returns:
      DataFrame with columns: date, ticker, action, source.
    """
    cfg = load_config()

    def _live_fetch() -> Any:
        # Local source first.
        local_path = Path(cfg.SP500_MEMBERSHIP_CSV)
        if local_path.exists():
            return {
                "kind": "csv_text",
                "payload": local_path.read_text(encoding="utf-8"),
                "source": f"local:{local_path}",
            }

        if cfg.SP500_MEMBERSHIP_URL:
            text = _request_text_with_retries(str(cfg.SP500_MEMBERSHIP_URL))
            return {
                "kind": "csv_text",
                "payload": text,
                "source": f"url:{cfg.SP500_MEMBERSHIP_URL}",
            }

        raise FileNotFoundError(
            "No public S&P 500 membership source reachable. Provide local CSV at "
            f"{cfg.SP500_MEMBERSHIP_CSV} or set SP500_MEMBERSHIP_URL."
        )

    def _normalize(raw_payload: Any) -> pd.DataFrame:
        if not isinstance(raw_payload, dict) or raw_payload.get("kind") != "csv_text":
            raise ValueError("Malformed membership source payload.")
        raw_df = pd.read_csv(io.StringIO(raw_payload["payload"]))
        events = _normalize_membership_csv(raw_df)
        events["source"] = str(raw_payload.get("source", "unknown"))
        events = events[(events["date"] >= pd.Timestamp(start)) & (events["date"] <= pd.Timestamp(end))].copy()
        return events.reset_index(drop=True)

    return _fetch_or_load_with_snapshot(
        vendor="sp500pub",
        ticker="membership",
        start=start,
        end=end,
        live_fetch_fn=_live_fetch,
        normalize_fn=_normalize,
    )


def fetch_sec_voo_holdings_proxy(start: str, end: str) -> pd.DataFrame:
    """Fetch free SEC-based VOO holdings proxy snapshots.

    Priority:
      1) Local CSV path from config.
      2) Optional URL from config.

    Expected columns in source CSV:
      - date (or report_date / filing_date)
      - ticker
      - weight

    Returns:
      DataFrame with columns: date, ticker, weight, source.
    """
    cfg = load_config()

    def _live_fetch() -> Any:
        local_path = Path(cfg.SEC_VOO_PROXY_CSV)
        if local_path.exists():
            return {
                "kind": "csv_text",
                "payload": local_path.read_text(encoding="utf-8"),
                "source": f"local:{local_path}",
            }

        if cfg.SEC_VOO_PROXY_URL:
            text = _request_text_with_retries(str(cfg.SEC_VOO_PROXY_URL))
            return {
                "kind": "csv_text",
                "payload": text,
                "source": f"url:{cfg.SEC_VOO_PROXY_URL}",
            }

        raise FileNotFoundError(
            "No SEC proxy holdings source reachable. Provide local CSV at "
            f"{cfg.SEC_VOO_PROXY_CSV} or set SEC_VOO_PROXY_URL."
        )

    def _normalize(raw_payload: Any) -> pd.DataFrame:
        if not isinstance(raw_payload, dict) or raw_payload.get("kind") != "csv_text":
            raise ValueError("Malformed SEC proxy source payload.")

        raw_df = pd.read_csv(io.StringIO(raw_payload["payload"]))
        cols = {c.lower(): c for c in raw_df.columns}

        date_col = None
        for candidate in ["date", "report_date", "filing_date", "as_of_date", "effective_date"]:
            if candidate in cols:
                date_col = cols[candidate]
                break
        if date_col is None or "ticker" not in cols or "weight" not in cols:
            raise ValueError("SEC holdings CSV must contain date/report_date, ticker, weight columns.")

        out = raw_df[[date_col, cols["ticker"], cols["weight"]]].copy()
        out.columns = ["date", "ticker", "weight"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["ticker"] = out["ticker"].map(_normalize_ticker)
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce")

        out = out.dropna(subset=["date", "ticker", "weight"])
        out = out[out["weight"] > 0].copy()
        if out.empty:
            raise ValueError("SEC proxy holdings normalization returned no valid rows.")

        out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))].copy()
        if out.empty:
            raise ValueError("SEC proxy holdings source does not overlap requested date range.")

        # Normalize weights by snapshot date.
        by_date_sum = out.groupby("date")["weight"].transform("sum")
        out["weight"] = np.where(by_date_sum > 0, out["weight"] / by_date_sum, np.nan)
        out = out.dropna(subset=["weight"]).copy()
        out["source"] = str(raw_payload.get("source", "unknown"))

        return out.sort_values(["date", "ticker"]).reset_index(drop=True)

    return _fetch_or_load_with_snapshot(
        vendor="secproxy",
        ticker="voo",
        start=start,
        end=end,
        live_fetch_fn=_live_fetch,
        normalize_fn=_normalize,
    )


def fetch_fred_cash_rate(
    start: str,
    end: str,
    series_id: str,
    api_key: str,
    as_of_date: str,
) -> pd.DataFrame:
    """Fetch point-in-time FRED/ALFRED cash rates."""
    if not api_key:
        raise ValueError("FRED API key is missing.")

    endpoint = "https://api.stlouisfed.org/fred/series/observations"

    def _live_fetch() -> Any:
        return _request_json_with_retries(
            endpoint,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start,
                "observation_end": end,
                "realtime_start": as_of_date,
                "realtime_end": as_of_date,
            },
        )

    def _normalize(raw_payload: Any) -> pd.DataFrame:
        if not isinstance(raw_payload, dict) or "observations" not in raw_payload:
            raise ValueError(f"Malformed FRED response for series {series_id}.")
        rows = raw_payload["observations"]
        if not isinstance(rows, list) or len(rows) == 0:
            raise ValueError(f"Empty FRED response for series {series_id}.")

        df = pd.DataFrame(rows)
        required = ["date", "value", "realtime_start", "realtime_end"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"FRED response missing columns: {missing}")

        df = df[required].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["annual_yield"] = pd.to_numeric(df["value"], errors="coerce") / 100.0
        df = df.dropna(subset=["date", "annual_yield"]).copy()
        df["source_series"] = series_id
        return df[["date", "annual_yield", "source_series", "realtime_start", "realtime_end"]]

    df = _fetch_or_load_with_snapshot(
        vendor="fred",
        ticker=series_id,
        start=start,
        end=end,
        live_fetch_fn=_live_fetch,
        normalize_fn=_normalize,
    )

    LOGGER.info(
        "FRED %s: rows=%s, range=%s..%s, vintage=%s",
        series_id,
        len(df),
        df["date"].min().date().isoformat(),
        df["date"].max().date().isoformat(),
        as_of_date,
    )
    return df

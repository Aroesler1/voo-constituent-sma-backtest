"""Point-in-time ticker to PERMNO resolution.

CRSP tickers are not stable identifiers. They are reassigned when a company
dies, and they collide across share classes of one issuer. `audit_universe.py`
measures the damage on the cached universe: 437 of 1,153 tickers resolve to more
than one PERMNO, 327 with gaps over a year between securities (SOLV: 26.7 years).

PERMNO is the permanent security identifier. The fix is to resolve ticker to
PERMNO *as of each date* using CRSP's name history (`crsp.dsenames`, with the
validity window `namedt` .. `nameendt`), and to treat each PERMNO as a distinct
instrument rather than concatenating them under a shared symbol.

This module is deliberately independent of a live WRDS connection: the resolver
operates on a name-history table however it is obtained, so it is unit-testable
against fixtures and against the cached parquet universe today, and only the
name-history fetch needs credentials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# CRSP share codes for ordinary common equity. ETFs (73) and other fund
# structures are deliberately excluded from a constituent universe; a benchmark
# ETF has to be requested explicitly rather than arriving through a ticker match.
COMMON_SHARE_CODES = (10, 11, 12, 18)

NAME_HISTORY_COLUMNS = ("permno", "ticker", "namedt", "nameendt", "shrcd", "exchcd")
# Optional columns used to resolve share-class tickers; absent in older extracts.
OPTIONAL_COLUMNS = ("shrcls", "tsymbol")


def normalise_name_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a dsenames-like table into the columns and dtypes used here."""
    missing = set(NAME_HISTORY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"name history missing columns: {sorted(missing)}")
    keep = list(NAME_HISTORY_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in frame.columns]
    out = frame.loc[:, keep].copy()
    for col in OPTIONAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(str).str.upper().str.strip().replace({"NAN": "", "NONE": ""})
        else:
            out[col] = ""
    # a missing ticker must stay empty, not become the string "<NA>" or "NAN"
    out["ticker"] = (
        out["ticker"].astype(str).str.upper().str.strip()
        .replace({"<NA>": "", "NAN": "", "NONE": ""})
    )
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    for col in ("namedt", "nameendt"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    # an open-ended name record has a null end date; treat it as still current
    out["nameendt"] = out["nameendt"].fillna(pd.Timestamp.max.normalize())
    for col in ("shrcd", "exchcd"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out.dropna(subset=["permno", "namedt"])


def split_share_class(ticker: str) -> tuple[str, str]:
    """Split a vendor ticker like BRK-B into its base symbol and share class.

    CRSP does not put the share class in `ticker`; BRK-B does not appear there
    at all, while BRK plus shrcls 'B' resolves to permno 83443. Data vendors and
    index files commonly use the hyphenated form, so a universe built from them
    silently loses every dual-class name unless the suffix is split off.
    """
    raw = str(ticker).upper().strip()
    for sep in ("-", "."):
        if sep in raw:
            base, _, suffix = raw.partition(sep)
            if base and suffix.isalpha() and len(suffix) <= 2:
                return base, suffix
    return raw, ""


def resolve_as_of(
    name_history: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
    share_codes: tuple[int, ...] | None = COMMON_SHARE_CODES,
) -> list[int]:
    """PERMNOs trading under `ticker` on `as_of`.

    Returns a LIST, not a scalar. A ticker with two live share classes genuinely
    maps to two securities on the same date, and collapsing that to one silently
    drops a security or mixes two. Callers decide the policy; this function
    refuses to guess.
    """
    as_of = pd.Timestamp(as_of)
    frame = name_history
    base, share_class = split_share_class(ticker)
    alive = (frame["namedt"] <= as_of) & (frame["nameendt"] >= as_of)

    hits = frame[alive & (frame["ticker"] == str(ticker).upper().strip())]
    if hits.empty and "tsymbol" in frame.columns:
        # some extracts carry the hyphenated form in tsymbol
        hits = frame[alive & (frame["tsymbol"] == str(ticker).upper().strip())]
    if hits.empty and share_class:
        # BRK-B -> ticker BRK with shrcls B
        hits = frame[alive & (frame["ticker"] == base) & (frame["shrcls"] == share_class)]
    if share_codes is not None:
        hits = hits[hits["shrcd"].isin(list(share_codes))]
    return sorted({int(p) for p in hits["permno"].dropna().tolist()})


def build_membership(
    name_history: pd.DataFrame,
    tickers: list[str],
    dates: pd.DatetimeIndex,
    share_codes: tuple[int, ...] | None = COMMON_SHARE_CODES,
) -> pd.DataFrame:
    """Long table of (date, ticker, permno) valid on that date.

    Rows are emitted per PERMNO, so a dual-class ticker produces two rows and
    downstream code is forced to confront the ambiguity rather than inherit a
    silent choice.
    """
    history = normalise_name_history(name_history)
    wanted = {str(t).upper().strip() for t in tickers}
    history = history[history["ticker"].isin(wanted)]
    if share_codes is not None:
        history = history[history["shrcd"].isin(list(share_codes))]

    records = []
    for row in history.itertuples(index=False):
        span = dates[(dates >= row.namedt) & (dates <= row.nameendt)]
        records.extend(
            {"date": d, "ticker": row.ticker, "permno": int(row.permno)} for d in span
        )
    if not records:
        return pd.DataFrame(columns=["date", "ticker", "permno"])
    return pd.DataFrame(records).sort_values(["date", "ticker", "permno"]).reset_index(drop=True)


def flag_ambiguous(membership: pd.DataFrame) -> pd.DataFrame:
    """Dates where a ticker maps to more than one PERMNO.

    These are the dual-class collisions that previously produced duplicate dates,
    raised inside the matrix builder, and dropped the name from the universe
    entirely. Surfacing them is the point: the caller must pick a policy (largest
    market cap, primary exchange, or keep both as separate instruments) and say
    which, rather than losing the security to a swallowed exception.
    """
    counts = membership.groupby(["date", "ticker"])["permno"].nunique()
    return (counts[counts > 1]
            .rename("n_permnos")
            .reset_index()
            .sort_values(["ticker", "date"]))


def select_primary(membership: pd.DataFrame, priority: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one PERMNO per (date, ticker) using an explicit priority.

    `priority` needs columns (permno, rank) with lower rank preferred -- for
    example ordering by market capitalisation or by exchange. Requiring it to be
    passed in keeps the tie-break a documented decision rather than an accident
    of row order.
    """
    if {"permno", "rank"} - set(priority.columns):
        raise ValueError("priority needs columns: permno, rank")
    merged = membership.merge(priority[["permno", "rank"]], on="permno", how="left")
    merged["rank"] = merged["rank"].fillna(float("inf"))
    merged = merged.sort_values(["date", "ticker", "rank", "permno"])
    return (merged.drop_duplicates(subset=["date", "ticker"], keep="first")
            .drop(columns="rank")
            .reset_index(drop=True))


# lower is preferred when two securities carry a ticker on the same day; the
# same order the CRSP loaders use to de-duplicate a ticker's rows
SHARE_PRIORITY = {11: 0, 10: 1, 18: 2, 12: 3, 41: 4, 40: 5, 42: 6, 48: 7, 71: 8, 70: 9, 72: 10}
INTERVAL_COLUMNS = ["ticker", "permno", "crsp_ticker", "start", "end", "method"]


def membership_intervals(membership: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of a boolean membership matrix as (ticker, start, end)."""
    rows = []
    index = pd.DatetimeIndex(membership.index)
    for ticker in membership.columns:
        flags = membership[ticker].fillna(False).to_numpy(dtype=bool)
        if not flags.any():
            continue
        edges = np.diff(np.r_[0, flags.astype(int), 0])
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
            rows.append({"ticker": ticker, "start": index[start], "end": index[stop]})
    return pd.DataFrame(rows, columns=["ticker", "start", "end"])


def _runs(days: pd.DatetimeIndex, labels: np.ndarray) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    """Compress a per-day label array (-1 = none) into (label, first, last) runs."""
    out = []
    if len(days) == 0:
        return out
    breaks = np.flatnonzero(np.diff(labels) != 0) + 1
    for lo, hi in zip(np.r_[0, breaks], np.r_[breaks, len(days)] - 1):
        out.append((int(labels[lo]), days[lo], days[hi]))
    return out


def _assign_days(days: pd.DatetimeIndex, candidates: pd.DataFrame, rank_columns: list[str]) -> np.ndarray:
    """Pick one candidate row per day; candidates carry namedt/nameendt windows."""
    best = np.full(len(days), -1, dtype=int)
    if candidates.empty:
        return best
    ordered = candidates.sort_values(rank_columns, kind="mergesort")
    values = days.values
    # walk from worst to best so the preferred candidate is written last
    for pos in range(len(ordered) - 1, -1, -1):
        row = ordered.iloc[pos]
        alive = (values >= np.datetime64(row["namedt"])) & (values <= np.datetime64(row["nameendt"]))
        best[alive] = int(row["_row"])
    return best


def resolve_membership_intervals(
    intervals: pd.DataFrame,
    name_history: pd.DataFrame,
    share_codes: tuple[int, ...] = COMMON_SHARE_CODES,
) -> pd.DataFrame:
    """Map each membership interval to the CRSP security that was the member.

    Index membership files key members by ticker, and they are not consistent
    about which ticker: some use the symbol of the day, some the company's
    current symbol for its whole history (WBA for Walgreens' years as WAG),
    some a hyphenated share class CRSP does not store (BRK-B). Matching the
    file's ticker against CRSP's ticker of the day silently drops every member
    the second and third conventions describe.

    Rows returned, one per resolved stretch of an interval:

    - ``direct``: CRSP carried the member ticker on those days.
    - ``share_class``: a hyphenated ticker matched on base symbol plus shrcls.
    - ``renamed``: no security carried the ticker then, but one that carried it
      at another time existed under a different ticker; that PERMNO is used,
      preferring the one whose matching record is closest in time.
    - ``unresolved``: nothing matches (for example post-bankruptcy "Q"
      symbols, which CRSP never lists); ``permno`` is missing.

    A PERMNO-day claimed directly by one member is never also handed to
    another member through a non-direct match, so a renamed match cannot
    double count a security.
    """
    history = normalise_name_history(name_history)
    if share_codes is not None:
        history = history[history["shrcd"].isin(list(share_codes))]
    history = history.reset_index(drop=True)
    history["_row"] = np.arange(len(history))
    history["_priority"] = history["shrcd"].map(SHARE_PRIORITY).fillna(99)

    segments: list[dict] = []
    for item in intervals.itertuples(index=False):
        ticker = str(item.ticker).upper().strip()
        start, end = pd.Timestamp(item.start), pd.Timestamp(item.end)
        days = pd.bdate_range(start, end)
        if len(days) == 0:
            continue
        base, share_class = split_share_class(ticker)
        matches = history[history["ticker"] == ticker]
        method = "direct"
        if matches.empty and share_class:
            matches = history[(history["ticker"] == base) & (history["shrcls"] == share_class)]
            method = "share_class"

        # 1) securities that carried the ticker on the day itself
        direct = matches.assign(_neg_permno=-matches["permno"].astype("int64"))
        labels = _assign_days(days, direct, ["_priority", "exchcd", "_neg_permno"])

        # 2) for days still open, a security that carried the ticker at another
        # time and existed then under some other ticker
        open_days = labels < 0
        if open_days.any() and not matches.empty:
            permnos = matches["permno"].astype("int64").unique()
            others = history[history["permno"].astype("int64").isin(permnos)].copy()
            # distance from each record to the nearest record with the ticker
            anchor = matches.groupby(matches["permno"].astype("int64"))
            mid = days[open_days][len(days[open_days]) // 2]
            nearest = {
                int(p): float(min(abs((g["namedt"] - mid).dt.days).min(), abs((g["nameendt"].clip(upper=pd.Timestamp("2100-01-01")) - mid).dt.days).min()))
                for p, g in anchor
            }
            others["_distance"] = others["permno"].astype("int64").map(nearest)
            others["_neg_permno"] = -others["permno"].astype("int64")
            renamed = _assign_days(days, others, ["_distance", "_priority", "_neg_permno"])
            labels = np.where(open_days, renamed, labels)
            renamed_rows = set(np.unique(renamed[open_days & (renamed >= 0)]).tolist())
        else:
            renamed_rows = set()

        for row_id, first, last in _runs(days, labels):
            if row_id < 0:
                segments.append({"ticker": ticker, "permno": pd.NA, "crsp_ticker": "",
                                 "start": first, "end": last, "method": "unresolved"})
                continue
            rec = history.iloc[row_id]
            segments.append({
                "ticker": ticker,
                "permno": int(rec["permno"]),
                "crsp_ticker": str(rec["ticker"]),
                "start": first,
                "end": last,
                "method": "renamed" if row_id in renamed_rows and str(rec["ticker"]) != base else method,
            })

    out = pd.DataFrame(segments, columns=INTERVAL_COLUMNS)
    return _drop_double_claims(out)


def _drop_double_claims(segments: pd.DataFrame) -> pd.DataFrame:
    """Trim non-direct stretches that another member already holds directly."""
    if segments.empty:
        return segments
    direct = segments[segments["method"].isin(["direct", "share_class"])]
    kept = []
    for seg in segments.itertuples(index=False):
        if seg.method not in ("renamed",):
            kept.append(seg._asdict())
            continue
        claims = direct[(direct["permno"] == seg.permno) & (direct["ticker"] != seg.ticker)
                        & (direct["start"] <= seg.end) & (direct["end"] >= seg.start)]
        days = pd.bdate_range(seg.start, seg.end)
        free = np.ones(len(days), dtype=bool)
        for claim in claims.itertuples(index=False):
            free &= ~((days >= claim.start) & (days <= claim.end))
        for flag, first, last in _runs(days, free.astype(int)):
            row = seg._asdict()
            row.update(start=first, end=last)
            if not flag:
                row.update(permno=pd.NA, crsp_ticker="", method="unresolved")
            kept.append(row)
    return pd.DataFrame(kept, columns=INTERVAL_COLUMNS).reset_index(drop=True)

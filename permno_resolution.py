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

import pandas as pd

# CRSP share codes for ordinary common equity. ETFs (73) and other fund
# structures are deliberately excluded from a constituent universe; a benchmark
# ETF has to be requested explicitly rather than arriving through a ticker match.
COMMON_SHARE_CODES = (10, 11, 12, 18)

NAME_HISTORY_COLUMNS = ("permno", "ticker", "namedt", "nameendt", "shrcd", "exchcd")


def normalise_name_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a dsenames-like table into the columns and dtypes used here."""
    missing = set(NAME_HISTORY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"name history missing columns: {sorted(missing)}")
    out = frame.loc[:, list(NAME_HISTORY_COLUMNS)].copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    for col in ("namedt", "nameendt"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    # an open-ended name record has a null end date; treat it as still current
    out["nameendt"] = out["nameendt"].fillna(pd.Timestamp.max.normalize())
    for col in ("shrcd", "exchcd"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out.dropna(subset=["permno", "namedt"])


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
    hits = frame[
        (frame["ticker"] == str(ticker).upper().strip())
        & (frame["namedt"] <= as_of)
        & (frame["nameendt"] >= as_of)
    ]
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

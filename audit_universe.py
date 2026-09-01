#!/usr/bin/env python3
"""Audit a cached CRSP price universe for ticker/PERMNO integrity.

CRSP tickers are not stable identifiers. They are reused when a company dies and
the symbol is reassigned, and they collide across share classes of the same
issuer. PERMNO is the permanent security identifier; a price history keyed on
ticker without a point-in-time PERMNO constraint can therefore splice unrelated
securities into one series.

Two failure modes, which have different consequences and are reported separately:

1. SEQUENTIAL REUSE - the same ticker maps to different PERMNOs over disjoint
   date ranges. Concatenating them fabricates a return at the boundary, and any
   trailing-window statistic spanning it (a 200-day SMA, a volatility estimate)
   mixes two different companies. A gap of more than a year between segments is
   strong evidence of genuine reuse rather than a share-class transition.

2. SIMULTANEOUS DUPLICATION - several PERMNOs trade under one ticker on the SAME
   dates, i.e. dual share classes. This yields duplicate dates, which makes the
   series non-reindexable; in this repo that raises inside the matrix builder,
   is swallowed by a broad except, and silently removes the name from the
   universe.

Usage:
    python audit_universe.py --cache-dir data_cache
    python audit_universe.py --cache-dir data_cache --json report.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

_TICKER_RE = re.compile(r"^crsp_(?P<ticker>[^_]+)_")
REUSE_GAP_DAYS = 365


def _ticker_from(path: Path) -> str | None:
    match = _TICKER_RE.match(path.name)
    return match.group("ticker") if match else None


def audit_file(path: Path) -> dict | None:
    try:
        frame = pd.read_parquet(path, columns=["date", "permno"])
    except Exception:
        return None

    ticker = _ticker_from(path)
    if ticker is None or frame.empty:
        return None

    frame = frame.dropna(subset=["permno"]).sort_values("date")
    n_permnos = int(frame["permno"].nunique())
    duplicate_dates = int(len(frame) - frame["date"].nunique())

    # contiguous runs of a single permno, in date order
    segments = []
    if n_permnos > 1 and duplicate_dates == 0:
        run = (frame["permno"] != frame["permno"].shift()).cumsum()
        bounds = frame.groupby(run).agg(
            permno=("permno", "first"), start=("date", "min"), end=("date", "max"), n=("date", "size")
        )
        segments = bounds.to_dict("records")

    max_gap_days = 0
    for prev, nxt in zip(segments, segments[1:]):
        max_gap_days = max(max_gap_days, (nxt["start"] - prev["end"]).days)

    return {
        "ticker": ticker,
        "rows": int(len(frame)),
        "n_permnos": n_permnos,
        "duplicate_dates": duplicate_dates,
        "n_segments": len(segments),
        "max_gap_days": max_gap_days,
        "likely_reuse": max_gap_days > REUSE_GAP_DAYS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--json", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    files = sorted(cache.glob("crsp_*.parquet"))
    if not files:
        print(f"no crsp_*.parquet files under {cache}")
        return 1

    records = [r for r in (audit_file(p) for p in files) if r is not None]
    if not records:
        print("no readable cache files")
        return 1

    frame = pd.DataFrame(records).drop_duplicates(subset="ticker", keep="last")
    total = len(frame)
    multi = frame[frame["n_permnos"] > 1]
    dupes = frame[frame["duplicate_dates"] > 0]
    reuse = frame[frame["likely_reuse"]]

    print(f"cached tickers audited: {total}")
    print(f"  more than one PERMNO      : {len(multi):>5}  ({100 * len(multi) / total:.1f}%)")
    print(f"  duplicate dates           : {len(dupes):>5}  ({100 * len(dupes) / total:.1f}%)  "
          f"-> dual share classes, silently dropped downstream")
    print(f"  likely sequential reuse   : {len(reuse):>5}  ({100 * len(reuse) / total:.1f}%)  "
          f"-> gap > {REUSE_GAP_DAYS}d between securities")

    if not reuse.empty:
        print("\nworst sequential reuse (gap between two securities sharing a ticker):")
        for row in reuse.nlargest(8, "max_gap_days").itertuples():
            print(f"  {row.ticker:<8s} {row.max_gap_days:>6,d} days  "
                  f"({row.max_gap_days / 365.25:.1f} years)  permnos={row.n_permnos}")

    if not dupes.empty:
        print("\nmost duplicate-date rows (dual share classes):")
        for row in dupes.nlargest(8, "duplicate_dates").itertuples():
            print(f"  {row.ticker:<8s} {row.duplicate_dates:>6,d} duplicate rows  "
                  f"permnos={row.n_permnos}")

    if args.json:
        args.json.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        print(f"\nreport -> {args.json}")

    print("\nFix: resolve ticker -> PERMNO as of each date via crsp.dsenames")
    print("(namedt/nameendt) and treat each PERMNO as a distinct instrument.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

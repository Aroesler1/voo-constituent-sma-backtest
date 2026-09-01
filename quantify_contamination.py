#!/usr/bin/env python3
"""Measure how much ticker/PERMNO contamination biases a constituent backtest.

`audit_universe.py` establishes that 38% of cached tickers resolve to more than
one PERMNO. This asks the question that actually matters: how much does that
change the answer?

Method. For each cached ticker the price history is rebuilt twice:

  CONTAMINATED  every row, as the ticker-keyed loader produced it. Where a
                ticker was reused, this concatenates unrelated securities.
  CLEAN         only rows whose PERMNO matches the one CRSP's name history
                (`crsp.dsenames`, namedt..nameendt) assigns to that ticker on
                that date, restricted to ordinary common equity.

The two panels are then compared at three levels, from mechanical to economic:

  1. Rows dropped, and how many returns were FABRICATED -- a "return" computed
     across a splice between two different companies is not a return at all.
  2. The size of those fabricated returns, since a splice between securities at
     different price levels can manufacture an enormous one.
  3. What a 200-day SMA signal -- the actual strategy signal in this repo --
     does differently on each panel. A trailing window that spans a splice
     averages two companies' prices, so the corruption reaches the signal, not
     just the return series.

Usage:
    python quantify_contamination.py --cache-dir data_cache --names data/universe/dsenames.parquet
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from permno_resolution import COMMON_SHARE_CODES, normalise_name_history

_TICKER_RE = re.compile(r"^crsp_(?P<ticker>[^_]+)_")
# A single-day move beyond this is treated as implausible for an ordinary US
# equity and flagged as likely fabricated rather than real.
IMPLAUSIBLE_RETURN = 0.5


def _clean_mask(frame: pd.DataFrame, history: pd.DataFrame, ticker: str) -> pd.Series:
    """True where a row's PERMNO is the one valid for that ticker on that date."""
    spans = history[
        (history["ticker"] == ticker)
        & (history["shrcd"].isin(list(COMMON_SHARE_CODES)))
    ]
    if spans.empty:
        return pd.Series(False, index=frame.index)

    keep = pd.Series(False, index=frame.index)
    for row in spans.itertuples(index=False):
        keep |= (
            (frame["permno"] == row.permno)
            & (frame["date"] >= row.namedt)
            & (frame["date"] <= row.nameendt)
        )
    return keep


def _sma_signal(prices: pd.Series, window: int = 200) -> pd.Series:
    """The repo's actual signal: is price above its trailing SMA."""
    sma = prices.rolling(window, min_periods=window).mean()
    return (prices > sma).where(sma.notna())


def analyse_ticker(path: Path, history: pd.DataFrame) -> dict | None:
    match = _TICKER_RE.match(path.name)
    if match is None:
        return None
    ticker = match.group("ticker").upper()

    try:
        frame = pd.read_parquet(path, columns=["date", "permno", "adjusted_close"])
    except Exception:
        return None
    if frame.empty:
        return None

    frame = frame.dropna(subset=["date", "permno"]).sort_values("date")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates(subset="date", keep="first")

    keep = _clean_mask(frame, history, ticker)
    clean = frame[keep]

    contaminated_px = frame.set_index("date")["adjusted_close"].astype(float)
    clean_px = clean.set_index("date")["adjusted_close"].astype(float)

    # a return is fabricated when consecutive rows come from different PERMNOs
    permno = frame["permno"].to_numpy()
    splice = np.zeros(len(frame), dtype=bool)
    splice[1:] = permno[1:] != permno[:-1]
    rets = contaminated_px.pct_change().to_numpy()
    fabricated = rets[splice & np.isfinite(rets)]

    # signal disagreement on the dates both panels cover
    sig_c = _sma_signal(contaminated_px)
    sig_k = _sma_signal(clean_px)
    common = sig_c.dropna().index.intersection(sig_k.dropna().index)
    disagree = int((sig_c.loc[common] != sig_k.loc[common]).sum()) if len(common) else 0

    return {
        "ticker": ticker,
        "rows_contaminated": int(len(frame)),
        "rows_clean": int(len(clean)),
        "rows_dropped": int(len(frame) - len(clean)),
        "n_permnos": int(frame["permno"].nunique()),
        "fabricated_returns": int(len(fabricated)),
        "max_abs_fabricated": float(np.abs(fabricated).max()) if len(fabricated) else 0.0,
        "implausible_fabricated": int((np.abs(fabricated) > IMPLAUSIBLE_RETURN).sum()),
        "signal_days_compared": int(len(common)),
        "signal_disagreements": disagree,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--names", type=Path, default=Path("data/universe/dsenames.parquet"))
    parser.add_argument("--out", type=Path, default=Path("output/contamination_bias.csv"))
    args = parser.parse_args()

    history = normalise_name_history(pd.read_parquet(args.names))
    files = sorted(args.cache_dir.glob("crsp_*.parquet"))
    if not files:
        print(f"no crsp_*.parquet under {args.cache_dir}")
        return 1

    seen: dict[str, dict] = {}
    for path in files:
        rec = analyse_ticker(path, history)
        if rec is not None:
            seen[rec["ticker"]] = rec  # last file per ticker wins
    table = pd.DataFrame(seen.values())
    if table.empty:
        print("no readable tickers")
        return 1

    n = len(table)
    affected = table[table["rows_dropped"] > 0]
    fabricated_total = int(table["fabricated_returns"].sum())
    implausible = int(table["implausible_fabricated"].sum())
    sig_total = int(table["signal_days_compared"].sum())
    sig_bad = int(table["signal_disagreements"].sum())

    print(f"tickers analysed: {n}\n")
    print("ROW-LEVEL CORRUPTION")
    print(f"  tickers losing rows to point-in-time filtering : {len(affected):>7,}  "
          f"({100*len(affected)/n:.1f}%)")
    print(f"  rows in contaminated panel                     : {int(table['rows_contaminated'].sum()):>7,}")
    print(f"  rows surviving PIT PERMNO resolution           : {int(table['rows_clean'].sum()):>7,}")
    print(f"  rows dropped                                   : {int(table['rows_dropped'].sum()):>7,}  "
          f"({100*table['rows_dropped'].sum()/table['rows_contaminated'].sum():.1f}%)")

    print("\nFABRICATED RETURNS (computed across a splice between two securities)")
    print(f"  fabricated returns in the contaminated panel    : {fabricated_total:>7,}")
    print(f"  of which exceed +/-{IMPLAUSIBLE_RETURN:.0%} in one day             : {implausible:>7,}")
    if fabricated_total:
        worst = table.nlargest(6, "max_abs_fabricated")[
            ["ticker", "n_permnos", "fabricated_returns", "max_abs_fabricated"]]
        print("\n  largest fabricated single-day moves:")
        for row in worst.itertuples(index=False):
            print(f"    {row.ticker:<8s} {row.max_abs_fabricated:>10.1%}  "
                  f"({row.n_permnos} permnos, {row.fabricated_returns} splices)")

    print("\nSIGNAL CORRUPTION (200-day SMA, the strategy's actual signal)")
    print(f"  ticker-days where both panels have a signal    : {sig_total:>7,}")
    print(f"  days the signal DISAGREES                      : {sig_bad:>7,}  "
          f"({100*sig_bad/sig_total:.2f}%)" if sig_total else "")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.sort_values("rows_dropped", ascending=False).to_csv(args.out, index=False)
    print(f"\nper-ticker detail -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

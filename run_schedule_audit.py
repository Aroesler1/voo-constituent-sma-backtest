"""Reproduce the headline and isolate calendar anchors from committed tables only."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from timing_luck import frequency_isolated_summary, timing_luck_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed outputs without writing")
    args = parser.parse_args(argv)
    reports = Path(__file__).resolve().parent / "reports"
    variants = pd.read_csv(reports / "timing_luck_variants.csv")
    original = pd.read_csv(reports / "timing_luck_summary.csv")
    implied_index = original["cagr_daily"] + original["gap_daily_to_index"]
    if float(implied_index.max() - implied_index.min()) > 1e-12:
        raise ValueError("Timing summary does not imply one common index CAGR.")
    index_cagr = float(implied_index.mean())
    rebuilt = timing_luck_summary(variants, index_cagr=index_cagr)
    pd.testing.assert_frame_equal(original, rebuilt, check_dtype=False, atol=1e-14, rtol=1e-12)
    frequency, decomposition = frequency_isolated_summary(variants)
    for name, frame in (("timing_luck_by_frequency.csv", frequency),
                        ("timing_luck_frequency_decomposition.csv", decomposition)):
        path = reports / name
        if args.check:
            pd.testing.assert_frame_equal(pd.read_csv(path), frame, check_dtype=False,
                                          atol=1e-14, rtol=1e-12)
        else:
            frame.to_csv(path, index=False)
    headline = variants.loc[variants.sma_length.eq(200) & variants.label.eq("semi_monthly")].iloc[0]
    print(f"Verified {len(variants)} committed variants and {len(rebuilt)} headline summaries.")
    print(f"SMA-200 semi-monthly CAGR: {headline.cagr:.12%}; Sharpe: {headline.sharpe:.12f}.")
    print("Frequency audit tables verified." if args.check else "Frequency audit tables written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for the CRSP CIZ (dsf_v2) loader and the tape comparison.

Two things have to hold for the tape comparison to mean anything. The CIZ rows
must land in exactly the schema the legacy rows land in, so that any difference
in the backtest is a difference in CRSP's numbers rather than in this code. And
the difference counter must separate an altered return from a day one tape
simply does not carry.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crsp_v2 import V2_COLUMN_MAP, compare_return_panels, normalize_v2_daily  # noqa: E402
from data_loader import _normalize_crsp_daily  # noqa: E402


def _legacy_raw(n=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "permno": 12345,
            "open_raw": close - 0.5,
            "high_raw": close + 0.8,
            "low_raw": close - 0.9,
            "close_raw": close,
            "volume_raw": rng.integers(1_000, 10_000, size=n).astype(float),
            "ret_raw": np.r_[np.nan, np.diff(close) / close[:-1]],
            "retx_raw": np.r_[np.nan, np.diff(close) / close[:-1]],
            "cfacpr": 2.0,
            "cfacshr": 2.0,
            "dlret_raw": np.nan,
        }
    )


def _v2_raw_from_legacy(legacy: pd.DataFrame) -> pd.DataFrame:
    """The same numbers, written with CIZ column names."""
    inverse = {v: k for k, v in V2_COLUMN_MAP.items()}
    out = legacy.drop(columns=["dlret_raw"]).rename(columns=inverse)
    return out


def test_v2_normalisation_matches_the_legacy_schema_and_numbers():
    legacy_raw = _legacy_raw()
    legacy = _normalize_crsp_daily(legacy_raw)
    v2 = normalize_v2_daily(_v2_raw_from_legacy(legacy_raw))

    assert list(v2.columns) == list(legacy.columns)
    pd.testing.assert_frame_equal(v2, legacy)


def test_v2_leaves_the_delisting_return_where_ciz_puts_it():
    """CIZ folds the delisting return into DlyRet, so nothing is compounded in.

    If the loader invented a delisting column the last day's total return would
    be compounded twice; this pins that it is not.
    """
    legacy_raw = _legacy_raw()
    v2_raw = _v2_raw_from_legacy(legacy_raw)
    v2_raw.loc[v2_raw.index[-1], "dlyret"] = -0.35  # a delisting day in CIZ

    out = normalize_v2_daily(v2_raw)
    assert out["total_return"].iloc[-1] == pytest.approx(-0.35)


def test_legacy_compounds_its_separate_delisting_return():
    """The counterpart on the legacy tape, to show the two paths differ."""
    legacy_raw = _legacy_raw()
    legacy_raw.loc[legacy_raw.index[-1], "ret_raw"] = -0.10
    legacy_raw.loc[legacy_raw.index[-1], "dlret_raw"] = -0.30

    out = _normalize_crsp_daily(legacy_raw)
    assert out["total_return"].iloc[-1] == pytest.approx((1 - 0.10) * (1 - 0.30) - 1.0)


def test_empty_input_returns_the_empty_schema():
    out = normalize_v2_daily(pd.DataFrame())
    assert out.empty
    assert "adjusted_close" in out.columns


def _panel(values, index, columns):
    return pd.DataFrame(values, index=index, columns=columns, dtype=float)


def test_difference_counter_flags_only_genuinely_altered_days():
    idx = pd.bdate_range("2020-01-02", periods=4)
    cols = ["AAA", "BBB"]
    legacy = _panel([[0.010, 0.020], [0.010, 0.020], [0.010, 0.020], [0.010, 0.020]], idx, cols)
    v2 = legacy.copy()
    v2.iloc[0, 0] += 0.0005  # 5 bps, altered
    v2.iloc[1, 1] += 0.00002  # 0.2 bps, below the 1 bp threshold

    out = compare_return_panels(legacy, v2, threshold_bps=1.0)
    assert out["n_comparable_days"] == 8
    assert out["n_altered_days"] == 1
    assert out["pct_altered"] == pytest.approx(100.0 / 8.0)
    assert out["mean_abs_diff_bps_altered"] == pytest.approx(5.0)
    assert out["max_abs_diff_bps"] == pytest.approx(5.0)
    assert out["n_tickers_compared"] == 2


def test_coverage_gaps_are_counted_apart_from_altered_returns():
    idx = pd.bdate_range("2020-01-02", periods=3)
    cols = ["AAA"]
    legacy = _panel([[0.01], [0.02], [np.nan]], idx, cols)
    v2 = _panel([[0.01], [np.nan], [0.03]], idx, cols)

    out = compare_return_panels(legacy, v2, threshold_bps=1.0)
    assert out["n_comparable_days"] == 1
    assert out["n_altered_days"] == 0
    assert out["n_legacy_only_days"] == 1
    assert out["n_v2_only_days"] == 1


def test_comparison_uses_only_the_shared_dates_and_tickers():
    legacy = _panel(
        np.zeros((3, 2)), pd.bdate_range("2020-01-02", periods=3), ["AAA", "BBB"]
    )
    v2 = _panel(
        np.zeros((3, 2)), pd.bdate_range("2020-01-03", periods=3), ["BBB", "CCC"]
    )

    out = compare_return_panels(legacy, v2)
    assert out["n_tickers_compared"] == 1
    assert out["n_comparable_days"] == 2
    assert out["n_altered_days"] == 0

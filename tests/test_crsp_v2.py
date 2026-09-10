"""Tests for the CRSP CIZ (dsf_v2) loader and the tape comparison.

Two things have to hold for the tape comparison to mean anything. The CIZ rows
must land in exactly the schema the legacy rows land in, so that any difference
in the backtest is a difference in CRSP's numbers rather than in this code. And
the difference counter must separate an altered return from a day one tape
simply does not carry.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crsp_v2 import (  # noqa: E402
    V2_COLUMN_MAP,
    _covering_cache_paths,
    apply_terminal_outcomes,
    compare_return_panels,
    load_verified_terminal_outcomes,
    normalize_v2_daily,
)
from config import BacktestConfig  # noqa: E402
from data_loader import _normalize_crsp_daily  # noqa: E402
from run_tape_compare import _terminal_mask_from_outcomes  # noqa: E402


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


def test_frozen_superset_cache_is_reused_for_a_pinned_subperiod(tmp_path):
    config = BacktestConfig(CACHE_DIR=str(tmp_path), OFFLINE_MODE=True)
    legacy = tmp_path / "crsp_v2"
    legacy.mkdir()
    superset = legacy / "crspv2_SPY_1993-01-29_2026-09-04.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2024-01-02"])}).to_parquet(superset)
    paths = _covering_cache_paths(
        config,
        "SPY",
        "1996-01-02",
        "2024-12-31",
    )
    assert superset in paths


def test_vendor_terminal_dates_map_by_permno_not_ticker_label():
    dates = pd.bdate_range("2020-01-02", periods=2)
    permnos = pd.DataFrame(
        {"AAA": pd.array([10001, 10001], dtype="Int64")},
        index=dates,
    )
    outcomes = pd.DataFrame(
        {
            "permno": [10001],
            "legacy_event_date": [dates[0]],
            "event_date": [dates[1]],
            "terminal_return": [-0.5],
        }
    )
    legacy = _terminal_mask_from_outcomes(
        permnos, outcomes, "legacy_event_date"
    )
    ciz = _terminal_mask_from_outcomes(permnos, outcomes, "event_date")
    assert legacy["AAA"].tolist() == [True, False]
    assert ciz["AAA"].tolist() == [False, True]


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


def test_separate_terminal_outcome_is_compounded_exactly_once():
    legacy_raw = _legacy_raw()
    v2_raw = _v2_raw_from_legacy(legacy_raw)
    v2_raw.loc[v2_raw.index[-1], "dlyret"] = -0.10
    daily = normalize_v2_daily(v2_raw)
    events = pd.DataFrame(
        {
            "permno": [12345],
            "event_date": [v2_raw["dlycaldt"].iloc[-1]],
            "terminal_return": [-0.30],
        }
    )

    out = apply_terminal_outcomes(
        daily, events, return_semantics="separate_from_dlyret"
    )
    assert out["total_return"].iloc[-1] == pytest.approx((1 - 0.10) * (1 - 0.30) - 1)
    assert out["is_terminal_session"].iloc[-1]
    assert out["terminal_return_applied"].iloc[-1]
    pd.testing.assert_series_equal(
        out["total_return"].iloc[:-1],
        daily["total_return"].iloc[:-1],
        check_names=False,
    )


def test_included_terminal_outcome_is_marked_but_not_double_counted():
    raw = _v2_raw_from_legacy(_legacy_raw())
    raw.loc[raw.index[-1], "dlyret"] = -0.35
    daily = normalize_v2_daily(raw)
    events = pd.DataFrame(
        {
            "permno": [12345],
            "event_date": [raw["dlycaldt"].iloc[-1]],
            "terminal_return": [-0.35],
        }
    )

    out = apply_terminal_outcomes(
        daily, events, return_semantics="included_in_dlyret"
    )
    assert out["total_return"].iloc[-1] == pytest.approx(-0.35)
    assert out["is_terminal_session"].iloc[-1]
    assert not out["terminal_return_applied"].iloc[-1]


def test_post_delist_ciz_daily_row_is_restored_without_prior_day_compounding():
    daily = normalize_v2_daily(_v2_raw_from_legacy(_legacy_raw()))
    prior_last_return = daily["total_return"].iloc[-1]
    event_date = daily["date"].iloc[-1] + pd.offsets.BDay(1)
    events = pd.DataFrame(
        {
            "permno": [12345],
            "event_date": [event_date],
            "terminal_return": [-0.60],
        }
    )

    out = apply_terminal_outcomes(
        daily,
        events,
        return_semantics="included_in_dlyret",
        start=daily["date"].min().date().isoformat(),
        end=event_date.date().isoformat(),
    )
    assert len(out) == len(daily) + 1
    assert out["total_return"].iloc[-2] == pytest.approx(prior_last_return)
    assert out["total_return"].iloc[-1] == pytest.approx(-0.60)
    assert out["terminal_return_applied"].iloc[-1]


def test_terminal_event_does_not_create_duplicate_ticker_date():
    daily = normalize_v2_daily(_v2_raw_from_legacy(_legacy_raw()))
    daily.loc[daily.index[:3], "permno"] = 99999
    event_date = daily["date"].iloc[-1] + pd.offsets.BDay(1)
    events = pd.DataFrame(
        {
            "permno": [99999, 12345],
            "event_date": [event_date, event_date],
            "terminal_return": [-0.40, -0.20],
        }
    )
    # A reused ticker can have two PERMNOs with the same terminal event date.
    # The ticker panel cannot represent both securities in one cell.
    out = apply_terminal_outcomes(
        daily,
        events,
        return_semantics="included_in_dlyret",
        end=event_date.date().isoformat(),
    )
    assert not out["date"].duplicated().any()
    assert len(out) == len(daily) + 1
    assert out["permno"].iloc[-1] == 12345
    assert out["total_return"].iloc[-1] == pytest.approx(-0.20)


def test_unknown_terminal_semantics_fail_closed():
    daily = normalize_v2_daily(_v2_raw_from_legacy(_legacy_raw()))
    events = pd.DataFrame(
        {"permno": [12345], "event_date": [daily["date"].iloc[-1]], "terminal_return": [-0.2]}
    )
    with pytest.raises(ValueError, match="Unknown"):
        apply_terminal_outcomes(daily, events, return_semantics="assumed")


def test_terminal_extract_requires_verified_hashed_provenance(tmp_path):
    extract = tmp_path / "terminal.csv"
    pd.DataFrame(
        {"permno": [12345], "event_date": ["2020-01-09"], "terminal_return": [-0.2]}
    ).to_csv(extract, index=False)
    metadata = {
        "source_product": "entitlement-reported product",
        "source_table": "entitlement-reported table",
        "source_table_verified": True,
        "daily_source_product": "entitlement-reported daily product",
        "daily_source_table": "verified_schema.verified_daily_table",
        "daily_source_table_verified": True,
        "return_semantics": "separate_from_dlyret",
        "query_text_sha256": "abc123",
        "extract_sha256": hashlib.sha256(extract.read_bytes()).hexdigest(),
    }
    sidecar = tmp_path / "terminal.metadata.json"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")

    out, loaded_metadata = load_verified_terminal_outcomes(extract, sidecar)
    assert len(out) == 1
    assert loaded_metadata["return_semantics"] == "separate_from_dlyret"

    metadata["source_table_verified"] = False
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="entitlement-verified"):
        load_verified_terminal_outcomes(extract, sidecar)


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


def test_comparison_excludes_days_with_different_permanent_identifiers():
    idx = pd.bdate_range("2020-01-02", periods=2)
    legacy = _panel([[0.01], [0.02]], idx, ["AAA"])
    v2 = _panel([[0.01], [0.50]], idx, ["AAA"])
    legacy_permno = _panel([[10001], [10001]], idx, ["AAA"]).astype("Int64")
    v2_permno = _panel([[10001], [20002]], idx, ["AAA"]).astype("Int64")

    out = compare_return_panels(
        legacy,
        v2,
        legacy_permno=legacy_permno,
        v2_permno=v2_permno,
    )
    assert out["n_comparable_days"] == 1
    assert out["n_identity_mismatch_days"] == 1
    assert out["n_altered_days"] == 0

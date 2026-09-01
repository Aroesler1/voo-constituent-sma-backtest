"""Tests for point-in-time ticker to PERMNO resolution.

Each test encodes one of the two failure modes the universe audit found:
sequential ticker reuse across unrelated companies, and simultaneous dual share
classes. Both silently corrupted the cached universe.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from permno_resolution import (  # noqa: E402
    build_membership,
    flag_ambiguous,
    normalise_name_history,
    resolve_as_of,
    select_primary,
)


def _history() -> pd.DataFrame:
    """Fixture mirroring the real collisions found in the cache.

    VOO: sequential reuse -- an unrelated company held the ticker until 2010,
         then the Vanguard ETF (share code 73, a fund) took it.
    TAP: two common share classes trading simultaneously.
    """
    return pd.DataFrame([
        # ticker reused years apart by different securities
        {"permno": 86379, "ticker": "VOO", "namedt": "1998-10-16",
         "nameendt": "2007-06-29", "shrcd": 11, "exchcd": 1},
        {"permno": 12305, "ticker": "VOO", "namedt": "2010-09-09",
         "nameendt": None, "shrcd": 73, "exchcd": 1},
        # dual share classes alive at the same time
        {"permno": 89495, "ticker": "TAP", "namedt": "2002-01-02",
         "nameendt": None, "shrcd": 11, "exchcd": 1},
        {"permno": 89346, "ticker": "TAP", "namedt": "2002-01-02",
         "nameendt": None, "shrcd": 12, "exchcd": 1},
    ])


def test_open_ended_name_records_stay_current():
    hist = normalise_name_history(_history())
    assert hist["nameendt"].notna().all()
    assert (hist.loc[hist["permno"] == 12305, "nameendt"] > pd.Timestamp("2030-01-01")).all()


def test_sequential_reuse_resolves_to_the_right_era():
    """The whole point: the same ticker is different securities at different dates."""
    hist = normalise_name_history(_history())
    assert resolve_as_of(hist, "VOO", "2005-01-03") == [86379]
    # in the gap the ticker belongs to nobody
    assert resolve_as_of(hist, "VOO", "2009-01-05") == []
    # post-2010 VOO is a fund, excluded from a common-equity universe
    assert resolve_as_of(hist, "VOO", "2015-01-05") == []
    # ...but findable when funds are explicitly admitted
    assert resolve_as_of(hist, "VOO", "2015-01-05", share_codes=None) == [12305]


def test_dual_share_classes_return_both_not_one():
    """Collapsing to a scalar here is what silently dropped TAP from the universe."""
    hist = normalise_name_history(_history())
    assert resolve_as_of(hist, "TAP", "2010-06-01") == [89346, 89495]


def test_membership_never_concatenates_across_permnos():
    dates = pd.to_datetime(["2005-01-03", "2015-01-05"])
    m = build_membership(_history(), ["VOO"], pd.DatetimeIndex(dates))
    # only the pre-2010 common-equity security qualifies; the ETF era does not
    assert set(m["permno"]) == {86379}
    assert set(m["date"]) == {pd.Timestamp("2005-01-03")}


def test_flag_ambiguous_surfaces_dual_class_dates():
    dates = pd.DatetimeIndex(pd.to_datetime(["2010-06-01", "2010-06-02"]))
    m = build_membership(_history(), ["TAP"], dates)
    amb = flag_ambiguous(m)
    assert len(amb) == 2
    assert set(amb["n_permnos"]) == {2}


def test_select_primary_requires_an_explicit_priority():
    dates = pd.DatetimeIndex(pd.to_datetime(["2010-06-01"]))
    m = build_membership(_history(), ["TAP"], dates)
    with pytest.raises(ValueError):
        select_primary(m, pd.DataFrame({"permno": [89495]}))

    chosen = select_primary(m, pd.DataFrame({"permno": [89495, 89346], "rank": [1, 2]}))
    assert len(chosen) == 1
    assert int(chosen.iloc[0]["permno"]) == 89495


def test_unranked_permnos_sort_last_rather_than_winning_by_accident():
    dates = pd.DatetimeIndex(pd.to_datetime(["2010-06-01"]))
    m = build_membership(_history(), ["TAP"], dates)
    chosen = select_primary(m, pd.DataFrame({"permno": [89346], "rank": [5]}))
    assert int(chosen.iloc[0]["permno"]) == 89346


def _history_with_class() -> pd.DataFrame:
    """CRSP keeps share class in `shrcls`, not in the ticker.

    BRK-B does not appear in `ticker` at all; BRK with shrcls 'B' is the Class B
    security. A universe built from vendor files, which use the hyphenated form,
    loses every dual-class name unless the suffix is split off.
    """
    return pd.DataFrame([
        {"permno": 17778, "ticker": "BRK", "namedt": "1996-05-09", "nameendt": "2002-01-01",
         "shrcd": 11, "exchcd": 1, "shrcls": "A", "tsymbol": "BRKA"},
        {"permno": 83443, "ticker": "BRK", "namedt": "1996-05-09", "nameendt": "2002-01-01",
         "shrcd": 11, "exchcd": 1, "shrcls": "B", "tsymbol": "BRKB"},
        {"permno": 14593, "ticker": "AAPL", "namedt": "1980-12-12", "nameendt": None,
         "shrcd": 11, "exchcd": 3, "shrcls": "", "tsymbol": "AAPL"},
    ])


def test_split_share_class():
    from permno_resolution import split_share_class
    assert split_share_class("BRK-B") == ("BRK", "B")
    assert split_share_class("BF.B") == ("BF", "B")
    assert split_share_class("AAPL") == ("AAPL", "")
    # a hyphen that is not a share class must not be split away
    assert split_share_class("SOME-LONGSUFFIX") == ("SOME-LONGSUFFIX", "")


def test_hyphenated_share_class_resolves_to_the_right_permno():
    hist = normalise_name_history(_history_with_class())
    assert resolve_as_of(hist, "BRK-B", "2000-06-01") == [83443]
    assert resolve_as_of(hist, "BRK-A", "2000-06-01") == [17778]
    # the bare ticker still returns BOTH classes, since both are alive
    assert resolve_as_of(hist, "BRK", "2000-06-01") == [17778, 83443]


def test_plain_tickers_are_unaffected_by_share_class_handling():
    hist = normalise_name_history(_history_with_class())
    assert resolve_as_of(hist, "AAPL", "2015-06-01") == [14593]


def test_missing_optional_columns_do_not_break_resolution():
    """Older extracts have no shrcls/tsymbol; resolution must still work."""
    frame = _history_with_class().drop(columns=["shrcls", "tsymbol"])
    hist = normalise_name_history(frame)
    assert resolve_as_of(hist, "AAPL", "2015-06-01") == [14593]
    assert resolve_as_of(hist, "BRK-B", "2000-06-01") == []

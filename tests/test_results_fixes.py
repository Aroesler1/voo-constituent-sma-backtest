"""Regression tests for the fixes that changed published results (2026-10).

Each test fails on the code before the fix: SMA warm-up inside the sample,
the absolute rebalance band, holiday-dated membership events, the $5 filter
and ADV on a total-return level, the spliced VOO proxy, the gap reset of the
price path, ticker-keyed membership, and the smaller metric fixes.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_engine import run_constituent_backtest  # noqa: E402
from config import BacktestConfig  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from panel import (  # noqa: E402
    _build_extended_voo_proxy,
    _extract_crsp_total_return_close,
    active_mask_for_length,
    assemble_member_prices,
)
from permno_resolution import (  # noqa: E402
    membership_intervals,
    resolve_membership_intervals,
)
from preprocessing import (  # noqa: E402
    _build_membership_from_events,
    compute_daily_liquidity_feature_matrices,
)


def _plain_config(**overrides) -> BacktestConfig:
    """Costless, unfiltered engine config so only the behaviour under test moves."""
    base = dict(
        ENABLE_ENHANCED_COST_MODEL=False,
        SLIPPAGE_BPS=0.0,
        OPEN_AUCTION_SLIPPAGE_BPS=0.0,
        INCLUDE_REGULATORY_FEES=False,
        ENFORCE_INVESTABILITY_FILTER=False,
        EXECUTION_TIMING="same_close",
        CASH_RATE_ANNUAL=0.0,
    )
    base.update(overrides)
    return BacktestConfig(**base)


def _engine_inputs(n_names: int, n_days: int = 6):
    idx = pd.bdate_range("2021-03-01", periods=n_days)
    cols = [f"S{i:03d}" for i in range(n_names)]
    price = pd.DataFrame(100.0, index=idx, columns=cols)
    returns = pd.DataFrame(0.0, index=idx, columns=cols)
    member = pd.DataFrame(True, index=idx, columns=cols)
    active = pd.DataFrame(1.0, index=idx, columns=cols)
    return idx, cols, price, returns, member, active


# --- rebalance band -------------------------------------------------------


def test_equal_weights_below_the_band_are_still_bought():
    # 450 names -> 0.222% each, below the 25 bp band; the old band left 100% cash
    idx, cols, price, returns, member, active = _engine_inputs(450)
    res = run_constituent_backtest(
        price, returns, member, active, idx[1:2], _plain_config(), store_weights=True
    )
    assert res["exposure"].iloc[-1] == pytest.approx(1.0)
    assert res["weights"].iloc[-1].min() == pytest.approx(1 / 450)


def test_small_position_is_sold_when_its_signal_turns_off():
    idx, cols, price, returns, member, active = _engine_inputs(300)
    # name 0 falls 30% after entry, so its weight drops below the band
    price.loc[idx[2]:, cols[0]] = 70.0
    returns.loc[idx[2], cols[0]] = -0.30
    active.loc[idx[3]:, cols[0]] = 0.0
    res = run_constituent_backtest(
        price, returns, member, active, idx[[1, 4]], _plain_config(), store_weights=True
    )
    assert res["weights"].loc[idx[3], cols[0]] < 0.0025
    assert res["weights"].loc[idx[4], cols[0]] == 0.0


def test_turnover_is_charged_on_the_renormalised_weights():
    # 40 held names sit inside the band when a 41st enters; their weights are
    # rescaled to fund it, and that rescaling is traded notional
    idx, cols, price, returns, member, active = _engine_inputs(41)
    active.loc[:idx[1], cols[40]] = 0.0
    res = run_constituent_backtest(price, returns, member, active, idx[[1, 3]], _plain_config())
    trades = res["trade_log"]
    traded = trades.loc[trades["date"] == idx[3], "turnover_fraction"].sum()
    scale = 1.0 / (1.0 + 1.0 / 41.0 / 1.0)
    expected = 40 * (1 / 40) * (1 - scale) + (1 / 41) * scale
    assert traded == pytest.approx(expected, rel=1e-9)
    # the old accounting charged only the new name's pre-scaling weight
    assert traded > 1 / 41 * 1.5


def test_excess_is_funded_by_the_names_that_trade_not_by_every_holding():
    # 40 held names inside the band, one exit frees cash, two entries need more
    # than that: only the entries are scaled, the 39 kept names do not trade
    idx, cols, price, returns, member, active = _engine_inputs(42)
    active.loc[:idx[1], cols[40:]] = 0.0
    active.loc[idx[2]:, cols[0]] = 0.0
    res = run_constituent_backtest(
        price, returns, member, active, idx[[1, 3]], _plain_config(), store_weights=True
    )
    trades = res["trade_log"]
    day = trades[trades["date"] == idx[3]]
    assert sorted(day["ticker"]) == [cols[0], cols[40], cols[41]]
    assert day["turnover_fraction"].sum() == pytest.approx(2 / 40, rel=1e-9)
    assert res["weights"].loc[idx[3], cols[40]] == pytest.approx(1 / 80)


# --- raw price filter and ADV ----------------------------------------------


def test_price_filter_uses_the_raw_price_not_the_total_return_level():
    idx, cols, price, returns, member, active = _engine_inputs(2)
    price.loc[:, cols[0]] = 8.0  # total-return level after a 2:1 split
    raw = pd.DataFrame(8.0, index=idx, columns=cols)
    raw.loc[:, cols[0]] = 4.0  # the stock actually trades at $4
    raw.loc[:, cols[1]] = 50.0
    price.loc[:, cols[1]] = 50.0
    price.attrs["raw_close"] = raw
    cfg = _plain_config(ENFORCE_INVESTABILITY_FILTER=True, MIN_ADV_USD_TO_TRADE=0.0)
    res = run_constituent_backtest(price, returns, member, active, idx[1:2], cfg, store_weights=True)
    assert res["weights"].loc[idx[-1], cols[0]] == 0.0
    assert res["weights"].loc[idx[-1], cols[1]] == pytest.approx(1.0)


def test_adv_uses_traded_dollar_volume_when_supplied():
    idx = pd.bdate_range("2021-01-04", periods=25)
    level = pd.DataFrame({"A": 8.0}, index=idx)  # total-return level
    volume = pd.DataFrame({"A": 1_000_000.0}, index=idx)
    dollars = pd.DataFrame({"A": 4_000_000.0}, index=idx)  # $4 x 1m shares traded
    liq = compute_daily_liquidity_feature_matrices(
        level, volume, level, level, adv_lookback=20, vol_lookback=20,
        spread_model="corwin_schultz", dollar_volume_df=dollars,
    )
    assert liq["adv_usd"]["A"].iloc[-1] == pytest.approx(4_000_000.0)


# --- membership --------------------------------------------------------------


def test_removal_dated_on_a_market_holiday_still_removes_the_member():
    trading = pd.bdate_range("2020-12-21", "2021-01-08").drop(
        [pd.Timestamp("2020-12-25"), pd.Timestamp("2021-01-01")]
    )
    events = pd.DataFrame(
        {
            "date": ["2020-12-21", "2020-12-25", "2021-01-01"],
            "ticker": ["AAA", "AAA", "BBB"],
            "action": ["add", "remove", "add"],
        }
    )
    mem = _build_membership_from_events(events, trading)
    assert mem.loc["2020-12-24", "AAA"]
    assert not mem.loc["2020-12-28":, "AAA"].any()
    assert mem.loc["2021-01-04":, "BBB"].all()


def _names() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Walgreens: WAG until 2014, WBA after, one PERMNO
            {"permno": 19502, "ticker": "WAG", "shrcls": "", "namedt": "1990-01-01",
             "nameendt": "2014-12-30", "shrcd": 11, "exchcd": 1},
            {"permno": 19502, "ticker": "WBA", "shrcls": "", "namedt": "2014-12-31",
             "nameendt": None, "shrcd": 11, "exchcd": 3},
            # two Berkshire classes under one CRSP ticker
            {"permno": 17778, "ticker": "BRK", "shrcls": "A", "namedt": "1990-01-01",
             "nameendt": None, "shrcd": 11, "exchcd": 1},
            {"permno": 83443, "ticker": "BRK", "shrcls": "B", "namedt": "1996-05-09",
             "nameendt": None, "shrcd": 11, "exchcd": 1},
            # a company that holds WAG directly in another member row
            {"permno": 55555, "ticker": "XYZ", "shrcls": "", "namedt": "1990-01-01",
             "nameendt": None, "shrcd": 11, "exchcd": 1},
        ]
    )


def test_current_ticker_applied_backwards_resolves_to_the_same_permno():
    intervals = pd.DataFrame({"ticker": ["WBA"], "start": ["2000-01-03"], "end": ["2020-12-31"]})
    out = resolve_membership_intervals(intervals, _names())
    assert set(out["permno"].astype(int)) == {19502}
    renamed = out[out["method"] == "renamed"].iloc[0]
    assert renamed["crsp_ticker"] == "WAG"
    assert renamed["end"] <= pd.Timestamp("2014-12-30")
    assert (out["method"] == "direct").any()


def test_hyphenated_class_resolves_on_ticker_and_share_class():
    intervals = pd.DataFrame({"ticker": ["BRK-B"], "start": ["2010-02-16"], "end": ["2024-12-31"]})
    out = resolve_membership_intervals(intervals, _names())
    assert out["permno"].astype(int).tolist() == [83443]
    assert out["method"].tolist() == ["share_class"]
    assert out["crsp_ticker"].tolist() == ["BRK"]


def test_symbol_crsp_never_carried_is_reported_unresolved():
    intervals = pd.DataFrame({"ticker": ["ENRNQ"], "start": ["1996-01-02"], "end": ["2001-11-30"]})
    out = resolve_membership_intervals(intervals, _names())
    assert out["method"].tolist() == ["unresolved"]
    assert out["permno"].isna().all()


def test_renamed_match_never_double_counts_a_direct_member():
    intervals = pd.DataFrame(
        {"ticker": ["WAG", "WBA"], "start": ["2000-01-03", "2000-01-03"],
         "end": ["2014-12-30", "2020-12-31"]}
    )
    out = resolve_membership_intervals(intervals, _names())
    wba = out[out["ticker"] == "WBA"]
    # 2000-2014 belongs to the WAG row; WBA keeps only its own era
    assert wba.loc[wba["permno"].notna(), "start"].min() >= pd.Timestamp("2014-12-31")


def test_membership_intervals_are_contiguous_runs():
    idx = pd.bdate_range("2021-01-04", periods=6)
    mem = pd.DataFrame({"A": [True, True, False, True, True, True]}, index=idx)
    out = membership_intervals(mem)
    assert list(zip(out["start"], out["end"])) == [(idx[0], idx[1]), (idx[3], idx[5])]


def test_member_history_is_rebuilt_from_the_resolved_permno():
    dates = pd.bdate_range("2014-12-24", periods=8)
    wag = pd.DataFrame({"date": dates[:4], "permno": 19502, "close": [10.0, 11, 12, 13]})
    wba = pd.DataFrame({"date": dates[4:], "permno": 19502, "close": [14.0, 15, 16, 17]})
    resolution = pd.DataFrame(
        [
            {"ticker": "WBA", "permno": 19502, "crsp_ticker": "WAG", "start": dates[0],
             "end": dates[3], "method": "renamed"},
            {"ticker": "WBA", "permno": 19502, "crsp_ticker": "WBA", "start": dates[4],
             "end": dates[-1], "method": "direct"},
        ]
    )
    out, report = assemble_member_prices({"WAG": wag, "WBA": wba}, resolution)
    assert out["WBA"]["close"].tolist() == [10.0, 11, 12, 13, 14, 15, 16, 17]
    assert bool(report.loc[0, "rebuilt"])
    assert int(report.loc[0, "missing_stretches"]) == 0


# --- price path and benchmark proxy ---------------------------------------------


def test_trading_gap_does_not_restart_the_total_return_path():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-21", "2020-01-22"])
    frame = pd.DataFrame(
        {"close": [100.0, 100.0, 50.0, 50.0], "total_return": [np.nan, 0.0, 0.0, 0.0],
         "permno": 1}, index=dates
    )
    # a 2:1 split during the halt: the raw price halves, the holder's wealth does not
    path = _extract_crsp_total_return_close(frame)
    assert path.tolist() == [100.0, 100.0, 100.0, 100.0]


def test_voo_proxy_does_not_splice_the_earlier_ticker_owner():
    spy = pd.Series(100.0, index=pd.bdate_range("2003-01-01", "2010-12-31"))
    spy *= np.cumprod(np.full(len(spy), 1.0001))
    old_owner = pd.Series(1.0, index=pd.bdate_range("2003-01-01", "2003-06-30"))
    etf = pd.Series(110.0, index=pd.bdate_range("2010-09-20", "2010-12-31"))
    proxy = _build_extended_voo_proxy(spy, pd.concat([old_owner, etf]), "2010-09-09")
    daily = proxy.pct_change().dropna()
    assert daily.abs().max() < 0.01
    # SPY keeps earning until VOO's first priced day, then VOO (flat) takes over
    assert proxy.loc["2010-09-17"] > proxy.loc["2010-09-09"]
    assert proxy.loc["2010-12-31"] == pytest.approx(proxy.loc["2010-09-20"])


# --- warm-up --------------------------------------------------------------------


def test_signal_is_defined_on_the_first_sample_day():
    history = pd.bdate_range("1995-01-02", "1996-03-29")
    trading = history[history >= "1996-01-02"]
    close = pd.DataFrame({"A": np.linspace(50, 80, len(history))}, index=history)
    panel = SimpleNamespace(history_close_df=close, trading_index=trading)
    mask = active_mask_for_length(panel, 200, BacktestConfig())
    assert mask.index.equals(trading)
    assert mask.iloc[0]["A"] == 1.0
    # the old mask, computed on the trading calendar alone, was empty here
    from strategy import compute_sma_matrix  # noqa: PLC0415

    assert compute_sma_matrix(close.loc[trading], 200).iloc[0].isna().all()


# --- smaller metric fixes ------------------------------------------------------------


def test_sortino_uses_downside_deviation_over_all_periods():
    idx = pd.bdate_range("2021-01-04", periods=6)
    rets = pd.Series([0.0, 0.02, -0.01, -0.03, 0.01, 0.0], index=idx)
    equity = (1 + rets).cumprod() * 100.0
    met = compute_metrics(equity, rets, pd.DataFrame(), pd.Series(1, index=idx, dtype="Int64"), 0.0)
    downside = np.sqrt(np.mean(np.minimum(rets.to_numpy(), 0) ** 2)) * np.sqrt(252)
    assert met["sortino"] == pytest.approx(met["cagr"] / downside)


def test_run_pbo_deflated_sharpe_matches_main_conventions():
    from run_pbo import sweep_deflated_sharpe  # noqa: PLC0415
    from statistics_mt import deflated_sharpe, per_period_sharpe  # noqa: PLC0415

    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2010-01-04", periods=600)
    panel = pd.DataFrame(rng.normal(3e-4, 0.01, size=(600, 3)), index=idx, columns=["a", "b", "c"])
    got = sweep_deflated_sharpe(panel, cash_rate=0.02, n_trials=9)
    excess = [panel[c] - 0.02 / 252.0 for c in panel.columns]
    trial = [per_period_sharpe(s) for s in excess]
    want = deflated_sharpe(excess[0], n_trials=9, trial_sharpes=trial)
    assert got["a"] == pytest.approx(want)


def test_bond_equivalent_cash_yield_accrues_on_a_365_day_year():
    from backtest_engine import _annual_to_period_cash_return  # noqa: PLC0415

    cfg = BacktestConfig()
    assert cfg.CASH_RATE_DAY_COUNT == "ACT/365"
    one_day = _annual_to_period_cash_return(0.0365, 1, cfg.CASH_RATE_DAY_COUNT, dynamic=True)
    assert one_day == pytest.approx(0.0001)


def test_clean_panel_signal_never_spans_two_permnos(tmp_path):
    from quantify_contamination import analyse_ticker  # noqa: PLC0415

    dates = pd.bdate_range("2000-01-03", periods=500)
    frame = pd.DataFrame(
        {
            "date": dates,
            "permno": [1] * 250 + [2] * 250,
            # an unrelated security at a very different price takes the ticker over
            "adjusted_close": [100.0] * 250 + [2.0] * 250,
        }
    )
    path = tmp_path / "crsp_ABC_2000-01-03_2001-12-31.parquet"
    frame.to_parquet(path)
    names = pd.DataFrame(
        [
            {"permno": 1, "ticker": "ABC", "namedt": "1990-01-01", "nameendt": dates[249],
             "shrcd": 11, "exchcd": 1},
            {"permno": 2, "ticker": "ABC", "namedt": dates[250], "nameendt": None,
             "shrcd": 11, "exchcd": 1},
        ]
    )
    from permno_resolution import normalise_name_history  # noqa: PLC0415

    rec = analyse_ticker(path, normalise_name_history(names))
    # the contaminated SMA straddles the handover and disagrees; the clean
    # signal restarts on permno 2 and only covers days with its own history
    assert rec["signal_disagreements"] == 0
    assert rec["signal_days_compared"] == (250 - 199) + (250 - 199)


def test_held_out_metrics_compound_the_first_held_out_return():
    from run_timing_luck import run_vol_managed_control  # noqa: PLC0415

    dates = pd.bdate_range("2010-01-04", periods=120)
    rets = pd.Series(0.001, index=dates)
    rets.iloc[61] = 0.05  # first held-out day
    panel = SimpleNamespace(trading_index=dates, cash_curve=None, effective_cash_rate=0.0)
    cfg = BacktestConfig(CASH_RATE_ANNUAL=0.0)
    table, _, _ = run_vol_managed_control(
        panel, cfg, {"x": {"returns": rets, "cost_bps": 0.0, "description": "x"}}
    )
    held = rets.iloc[61:]
    row = table[table["variant"] == "buy_and_hold"].iloc[0]
    assert row["cagr"] == pytest.approx(np.prod(1 + held) ** (252 / len(held)) - 1)

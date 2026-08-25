"""
Tests for the 2026-08 review additions:
- EDGE spread estimator (Ardia-Guidotti-Kroencke 2024) recovers known spreads
  and stays point-in-time safe
- deflated Sharpe ratio properties
- trade-level profit concentration diagnostics
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import _matched_round_trips  # noqa: E402
from spread_edge import edge, edge_spread_series  # noqa: E402
from statistics_mt import (  # noqa: E402
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
)


def _simulate_ohlc(n_days: int, spread: float, seed: int = 0):
    """Random-walk efficient price observed with a full bid-ask bounce."""
    rng = np.random.default_rng(seed)
    steps_per_day = 20
    log_mid = np.cumsum(rng.normal(0.0, 0.02 / np.sqrt(steps_per_day), size=n_days * steps_per_day))
    mid = 100.0 * np.exp(log_mid).reshape(n_days, steps_per_day)
    side = rng.choice([-1.0, 1.0], size=(n_days, steps_per_day))
    obs = mid * (1.0 + side * spread / 2.0)

    open_px = obs[:, 0]
    close_px = obs[:, -1]
    high_px = obs.max(axis=1)
    low_px = obs.min(axis=1)
    return open_px, high_px, low_px, close_px


def test_edge_recovers_simulated_spread():
    true_spread = 0.01  # 100 bps
    o, h, l, c = _simulate_ohlc(4000, true_spread)
    est = edge(o, h, l, c)
    assert 0.5 * true_spread < est < 1.5 * true_spread


def test_edge_orders_spreads_correctly():
    # frictionless prices must estimate well below a 100 bps market
    o0, h0, l0, c0 = _simulate_ohlc(4000, 0.0)
    o1, h1, l1, c1 = _simulate_ohlc(4000, 0.01)
    est0 = edge(o0, h0, l0, c0)
    est1 = edge(o1, h1, l1, c1)
    assert est0 < 0.5 * est1


def test_edge_spread_series_is_point_in_time():
    o, h, l, c = _simulate_ohlc(600, 0.01)
    idx = pd.bdate_range("2020-01-01", periods=600)
    kwargs = dict(window=126, step=21)

    base = edge_spread_series(
        pd.Series(o, idx), pd.Series(h, idx), pd.Series(l, idx), pd.Series(c, idx), **kwargs
    )

    # perturb the FUTURE beyond day 400; estimates through day 400 must not move
    h2, c2 = h.copy(), c.copy()
    h2[401:] *= 1.5
    c2[401:] *= 1.4
    bumped = edge_spread_series(
        pd.Series(o, idx), pd.Series(h2, idx), pd.Series(l, idx), pd.Series(c2, idx), **kwargs
    )

    pd.testing.assert_series_equal(base.iloc[:400], bumped.iloc[:400])


def test_dsr_properties():
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.normal(0.0, 0.01, size=5000))
    noise = noise - noise.mean()  # exactly zero Sharpe
    assert abs(probabilistic_sharpe(noise) - 0.5) < 0.02

    vals = [expected_max_sharpe(n, 0.02**2) for n in (1, 5, 25)]
    assert vals[0] == 0.0 and vals == sorted(vals)

    good = pd.Series(rng.normal(0.001, 0.01, size=500))
    psr = probabilistic_sharpe(good)
    dsr = deflated_sharpe(good, n_trials=10, trial_sharpes=[0.0, 0.03, -0.02, 0.05, 0.01])
    assert dsr < psr


def test_round_trips_carry_pnl_for_concentration():
    rows = []
    d0 = pd.Timestamp("2020-01-02")
    # 25 round trips: 24 flat-ish, one large winner
    for i in range(25):
        entry_px = 100.0
        exit_px = 300.0 if i == 0 else 100.5
        rows.append(
            {
                "date": d0 + pd.Timedelta(days=2 * i),
                "ticker": f"T{i}",
                "direction": "BUY",
                "price": entry_px,
                "trade_notional": 1000.0,
            }
        )
        rows.append(
            {
                "date": d0 + pd.Timedelta(days=2 * i + 1),
                "ticker": f"T{i}",
                "direction": "SELL",
                "price": exit_px,
                "trade_notional": 1000.0 * exit_px / entry_px,
            }
        )
    trades = _matched_round_trips(pd.DataFrame(rows))
    assert "pnl_usd" in trades.columns
    pnl = trades["pnl_usd"]
    top_share = pnl.sort_values(ascending=False).iloc[:2].sum() / pnl[pnl > 0].sum()
    # the single big winner dominates gross profit
    assert top_share > 0.9

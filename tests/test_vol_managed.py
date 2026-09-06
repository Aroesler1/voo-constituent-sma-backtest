"""Tests for the Moreira-Muir volatility-managed overlay.

The overlay is used as a positive control, so what has to be pinned is that it
is implementable: no lookahead in the weight, the calibration does what it
claims on the training half only, and the cap and the turnover charge bite.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vol_managed import (  # noqa: E402
    apply_vol_management,
    calibrate_c,
    realized_variance,
    vol_managed_weights,
)


def _returns(n=1500, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(pd.bdate_range("2010-01-04", periods=n))
    return pd.Series(rng.normal(0.0003, sigma, size=n), index=idx)


def test_realized_variance_is_the_mean_squared_return():
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=pd.bdate_range("2020-01-01", periods=5))
    rv = realized_variance(r, window=3)
    assert np.isnan(rv.iloc[0]) and np.isnan(rv.iloc[1])
    assert rv.iloc[2] == pytest.approx((0.01**2 + 0.02**2 + 0.03**2) / 3.0)
    assert rv.iloc[4] == pytest.approx((0.03**2 + 0.01**2 + 0.02**2) / 3.0)


def test_realized_variance_rejects_degenerate_window():
    with pytest.raises(ValueError):
        realized_variance(_returns(50), window=1)


def test_calibration_makes_the_uncapped_weight_average_one_in_training():
    r = _returns()
    train_end = r.index[len(r) // 2]
    c = calibrate_c(r, train_end=train_end)

    rv = realized_variance(r).shift(1)
    raw = (c / rv).loc[rv.index <= train_end].dropna()
    assert raw.mean() == pytest.approx(1.0, rel=1e-9)


def test_calibration_ignores_the_test_half():
    """Doubling volatility after the split must not move the constant."""
    r = _returns()
    train_end = r.index[len(r) // 2]
    bumped = r.copy()
    bumped.loc[bumped.index > train_end] *= 5.0

    assert calibrate_c(r, train_end=train_end) == pytest.approx(
        calibrate_c(bumped, train_end=train_end), rel=1e-12
    )


def test_weight_uses_no_information_from_its_own_day():
    """Perturbing the return on day t must not change the weight applied at t."""
    r = _returns(n=400)
    c = calibrate_c(r, train_end=r.index[200])
    base = vol_managed_weights(r, c=c)

    bumped = r.copy()
    bumped.iloc[300] = 0.25  # a huge shock on one day
    shocked = vol_managed_weights(bumped, c=c)

    pd.testing.assert_series_equal(base.iloc[:301], shocked.iloc[:301])
    # and the shock must show up from the next day onwards
    assert shocked.iloc[301] < base.iloc[301]


def test_cap_binds_and_warmup_is_neutral():
    r = _returns()
    c = calibrate_c(r, train_end=r.index[len(r) // 2])
    w = vol_managed_weights(r, c=c, cap=1.5)
    assert w.max() <= 1.5 + 1e-12
    assert (w >= 0.0).all()
    # The 21-day window needs 21 observations and the weight is lagged a day,
    # so the first 21 dates carry no estimate; the control must sit fully
    # invested there rather than sitting out.
    assert (w.iloc[:21] == 1.0).all()
    assert w.iloc[21] != 1.0


def test_constant_volatility_leaves_returns_untouched():
    """With flat volatility the weight is one, so the overlay is a no-op."""
    idx = pd.DatetimeIndex(pd.bdate_range("2010-01-04", periods=400))
    r = pd.Series(np.where(np.arange(400) % 2 == 0, 0.01, -0.01), index=idx, dtype=float)
    c = calibrate_c(r, train_end=idx[200])
    out = apply_vol_management(r, c=c, cost_bps=0.0)

    assert out["avg_weight"] == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(out["returns"].to_numpy(), r.to_numpy(), atol=1e-12)


def test_cash_sleeve_is_paid_on_the_unfunded_share():
    idx = pd.DatetimeIndex(pd.bdate_range("2010-01-04", periods=60))
    r = pd.Series(0.0, index=idx)
    cash = pd.Series(0.001, index=idx)
    # A zero-variance path would divide by zero, so give the first half a live
    # variance and hold the weight down with a tight cap. The cap also bounds
    # the warm-up fill, so every weight stays at or below 0.5.
    r.iloc[:30] = 0.01
    c = calibrate_c(r, train_end=idx[40])
    out = apply_vol_management(r, c=c, cash_returns=cash, cap=0.5, cost_bps=0.0)

    w = out["weights"]
    expected = w * r + (1.0 - w) * cash
    np.testing.assert_allclose(out["gross_returns"].to_numpy(), expected.to_numpy(), atol=1e-15)
    assert (w <= 0.5 + 1e-12).all()


def test_turnover_is_charged_and_reduces_the_net_return():
    r = _returns(n=800, seed=3)
    c = calibrate_c(r, train_end=r.index[400])
    free = apply_vol_management(r, c=c, cost_bps=0.0)
    charged = apply_vol_management(r, c=c, cost_bps=25.0)

    assert charged["annual_turnover"] > 0
    assert charged["equity_curve"].iloc[-1] < free["equity_curve"].iloc[-1]
    # The charge is exactly turnover times the rate, including establishing the
    # first position.
    delta = free["weights"].diff()
    delta.iloc[0] = free["weights"].iloc[0]
    expected_cost = delta.abs() * 25.0 / 10_000.0
    np.testing.assert_allclose(charged["cost_drag"].to_numpy(), expected_cost.to_numpy(), atol=1e-15)


def test_leverage_is_bounded_by_the_cap():
    r = _returns(n=1200, seed=5)
    # Quiet second half: the inverse-variance weight wants to lever up.
    r.iloc[600:] *= 0.2
    c = calibrate_c(r, train_end=r.index[600])
    out = apply_vol_management(r, c=c, cap=1.5, cost_bps=0.0)

    assert out["max_weight"] == pytest.approx(1.5, abs=1e-9)
    assert 0.0 < out["pct_time_levered"] < 1.0


def test_rejects_nonpositive_cap():
    r = _returns(n=100)
    with pytest.raises(ValueError):
        vol_managed_weights(r, c=1e-4, cap=0.0)

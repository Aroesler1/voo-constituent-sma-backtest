"""Tests for the signal-evaluation schedule option and the timing-luck sweep.

The pinned property is the one the sweep exists to measure: dispersion across
evaluation days must come from the price path, and must vanish when the path
carries no information about when to trade.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_engine import run_constituent_backtest  # noqa: E402
from config import BacktestConfig  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from timing_luck import (  # noqa: E402
    EvaluationSchedule,
    build_evaluation_calendar,
    enumerate_schedules,
    schedule_diagnostics,
    timing_luck_summary,
)


def _calendar(start="2021-01-04", periods=520):
    return pd.DatetimeIndex(pd.bdate_range(start, periods=periods))


def test_enumeration_covers_the_twenty_seven_variants():
    schedules = enumerate_schedules()
    assert len(schedules) == 27
    assert sum(s.frequency == "daily" for s in schedules) == 1
    assert sum(s.frequency == "weekly" for s in schedules) == 5
    assert sum(s.frequency == "monthly" for s in schedules) == 21
    assert len({s.label for s in schedules}) == 27


def test_schedule_rejects_bad_anchors():
    with pytest.raises(ValueError):
        EvaluationSchedule("weekly", 6)
    with pytest.raises(ValueError):
        EvaluationSchedule("monthly", 22)
    with pytest.raises(ValueError):
        EvaluationSchedule("daily", 1)
    with pytest.raises(ValueError):
        EvaluationSchedule("weekly")


def test_daily_calendar_is_every_day_after_the_first():
    idx = _calendar()
    cal = build_evaluation_calendar(idx, EvaluationSchedule("daily"))
    pd.testing.assert_index_equal(cal, idx[1:])


def test_weekly_anchor_selects_the_kth_trading_day_and_executes_next_session():
    idx = _calendar("2021-01-04", 20)  # four clean Mon-Fri weeks
    for k in range(1, 6):
        cal = build_evaluation_calendar(idx, EvaluationSchedule("weekly", k))
        # In a full week the k-th trading day is the k-th weekday, and execution
        # is the following session.
        first_eval = idx[k - 1]
        assert cal[0] == idx[idx.get_loc(first_eval) + 1]
        # One rebalance per week, less any that would execute past the sample.
        assert 3 <= len(cal) <= 4


def test_monthly_anchor_falls_back_in_short_months():
    idx = _calendar("2021-01-04", 300)
    long_anchor = EvaluationSchedule("monthly", 21)
    diag = schedule_diagnostics(idx, long_anchor)
    # Business-day months here run 20-23 days, so a 21st-day anchor must fall
    # back at least once, and must still fire once per month.
    assert diag["n_short_period_fallbacks"] >= 1
    assert diag["n_rebalances"] >= diag["n_periods"] - 1


def test_every_variant_rebalances_once_per_period():
    idx = _calendar("2015-01-05", 1500)
    for schedule in enumerate_schedules():
        if schedule.frequency == "daily":
            continue
        diag = schedule_diagnostics(idx, schedule)
        # Exactly one execution per period, except the final period whose
        # execution date can fall outside the sample.
        assert diag["n_periods"] - 1 <= diag["n_rebalances"] <= diag["n_periods"]


def test_calendars_are_a_subset_of_the_trading_index():
    idx = _calendar("2018-01-02", 800)
    for schedule in enumerate_schedules():
        cal = build_evaluation_calendar(idx, schedule)
        assert cal.isin(idx).all()
        assert idx[0] not in cal


# ---------------------------------------------------------------------------
# The property the sweep is for
# ---------------------------------------------------------------------------


def _flat_cost_config(daily_rate: float) -> BacktestConfig:
    """Costless config whose cash sleeve earns exactly the asset's daily return.

    When cash and the asset pay the same thing, being in the market or out of it
    is the same trade, so no evaluation day can be luckier than another. The
    config is built directly rather than through ``load_config`` because
    ``validate`` demands WRDS credentials the tests do not have.
    """
    return BacktestConfig(
        ENABLE_ENHANCED_COST_MODEL=False,
        SLIPPAGE_BPS=0.0,
        OPEN_AUCTION_SLIPPAGE_BPS=0.0,
        COMMISSION_PER_TRADE=0.0,
        COMMISSION_PER_SHARE=0.0,
        MIN_COMMISSION_PER_ORDER=0.0,
        EXPLICIT_FEE_BPS=0.0,
        INCLUDE_REGULATORY_FEES=False,
        ENFORCE_INVESTABILITY_FILTER=False,
        REBALANCE_BUFFER_BPS=0.0,
        MIN_TRADE_NOTIONAL_USD=0.0,
        EXECUTION_TIMING="same_close",
        CASH_RATE_DAY_COUNT="ACT/360",
        # _annual_to_period_cash_return accrues annual * days / 360 on a dynamic
        # curve, so this annual figure pays `daily_rate` on each one-day step.
        CASH_RATE_ANNUAL=daily_rate * 360.0,
    )


def _calendar_day_steps(idx: pd.DatetimeIndex) -> pd.Series:
    """Calendar days per step, matching the engine's cash accrual exactly."""
    return idx.to_series().diff().dt.days.fillna(1).clip(lower=1).astype(float)


def _constant_return_panel(n_days=420, n_tickers=4, daily_rate=0.0004):
    """Every name accrues the same simple rate per calendar day.

    Accrual is per calendar day, not per trading day, because the cash sleeve
    accrues that way: a Friday-to-Monday step pays three days of cash. Matching
    the convention is what makes cash and the asset exactly interchangeable.
    """
    idx = pd.DatetimeIndex(pd.bdate_range("2019-01-02", periods=n_days))
    cols = [f"T{i}" for i in range(n_tickers)]
    step_return = daily_rate * _calendar_day_steps(idx)
    step_return.iloc[0] = 0.0
    returns = pd.DataFrame(np.tile(step_return.to_numpy()[:, None], (1, n_tickers)), index=idx, columns=cols)
    close = 100.0 * (1.0 + returns).cumprod()
    membership = pd.DataFrame(True, index=idx, columns=cols)
    return idx, close, returns, membership


def test_constant_return_path_gives_zero_timing_spread():
    """A path with no information about timing must produce no timing luck.

    The prices rise at a fixed rate, so the SMA signal is on for every name from
    the end of the warm-up, and the cash sleeve pays exactly what the names pay.
    Every one of the 27 evaluation schedules must then produce the identical
    equity curve. Any spread here would be an artefact of the schedule
    machinery rather than a measurement of the market.
    """
    daily_rate = 0.0004
    idx, close, returns, membership = _constant_return_panel(daily_rate=daily_rate)
    config = _flat_cost_config(daily_rate)
    cash_curve = pd.Series(daily_rate * 360.0, index=idx, dtype=float)

    sma = close.rolling(window=200, min_periods=200).mean()
    active = (close > sma).astype(float).where(sma.notna())

    rows = []
    curves = {}
    for schedule in enumerate_schedules():
        result = run_constituent_backtest(
            price_df=close,
            return_df=returns,
            membership_mask=membership,
            active_mask=active,
            rebalance_calendar=build_evaluation_calendar(idx, schedule),
            config=config,
            cash_curve=cash_curve,
            store_weights=False,
            store_cost_attribution=False,
        )
        met = compute_metrics(
            result["equity_curve"],
            result["weekly_returns"],
            result["trade_log"],
            result["positions"],
            0.0,
        )
        curves[schedule.label] = result["equity_curve"]
        rows.append({"sma_length": 200, "label": schedule.label, "cagr": met["cagr"], "sharpe": met["sharpe"]})

    variants = pd.DataFrame(rows)
    assert len(variants) == 27
    assert variants["cagr"].max() - variants["cagr"].min() == pytest.approx(0.0, abs=1e-12)

    reference = curves["daily"]
    for label, curve in curves.items():
        assert np.allclose(curve.to_numpy(), reference.to_numpy(), rtol=0, atol=1e-6), label

    summary = timing_luck_summary(variants, index_cagr=float(variants["cagr"].iloc[0]) + 0.01)
    assert summary.loc[0, "cagr_range"] == pytest.approx(0.0, abs=1e-12)
    assert summary.loc[0, "cagr_std"] == pytest.approx(0.0, abs=1e-12)
    assert summary.loc[0, "n_variants"] == 27


def test_volatile_path_does_produce_a_spread():
    """The complement: with a real price path the schedules must disagree.

    Without this the zero-spread test above would also pass on a broken
    implementation that ignored the schedule entirely.
    """
    rng = np.random.default_rng(7)
    n_days, n_tickers = 420, 4
    idx = pd.DatetimeIndex(pd.bdate_range("2019-01-02", periods=n_days))
    cols = [f"T{i}" for i in range(n_tickers)]
    shocks = rng.normal(0.0004, 0.02, size=(n_days, n_tickers))
    shocks[0] = 0.0
    returns = pd.DataFrame(shocks, index=idx, columns=cols)
    close = 100.0 * (1.0 + returns).cumprod()
    membership = pd.DataFrame(True, index=idx, columns=cols)

    config = _flat_cost_config(0.0004)
    cash_curve = pd.Series(0.0004 * 360.0, index=idx, dtype=float)
    sma = close.rolling(window=200, min_periods=200).mean()
    active = (close > sma).astype(float).where(sma.notna())

    cagrs = []
    for schedule in enumerate_schedules():
        result = run_constituent_backtest(
            price_df=close,
            return_df=returns,
            membership_mask=membership,
            active_mask=active,
            rebalance_calendar=build_evaluation_calendar(idx, schedule),
            config=config,
            cash_curve=cash_curve,
            store_weights=False,
            store_cost_attribution=False,
        )
        met = compute_metrics(
            result["equity_curve"],
            result["weekly_returns"],
            result["trade_log"],
            result["positions"],
            0.0,
        )
        cagrs.append(met["cagr"])

    assert max(cagrs) - min(cagrs) > 1e-4


def test_summary_scales_the_range_against_the_gap_to_the_index():
    variants = pd.DataFrame(
        {
            "sma_length": [200] * 3,
            "label": ["daily", "weekly_01", "monthly_07"],
            "cagr": [0.06, 0.05, 0.08],
            "sharpe": [0.30, 0.25, 0.40],
        }
    )
    out = timing_luck_summary(variants, index_cagr=0.10)
    row = out.iloc[0]
    assert row["cagr_daily"] == pytest.approx(0.06)
    assert row["cagr_range"] == pytest.approx(0.03)
    assert row["gap_daily_to_index"] == pytest.approx(0.04)
    assert row["range_over_daily_gap"] == pytest.approx(0.75)
    assert row["sharpe_range"] == pytest.approx(0.15)
    # No headline row present, so its columns are absent rather than invented.
    assert pd.isna(row["cagr_headline"])
    assert pd.isna(row["range_over_headline_gap"])


def test_summary_requires_the_expected_columns():
    with pytest.raises(ValueError):
        timing_luck_summary(pd.DataFrame({"sma_length": [200], "cagr": [0.1]}), index_cagr=0.1)


# ---------------------------------------------------------------------------
# The headline schedule, carried alongside the 27
# ---------------------------------------------------------------------------


def test_headline_schedule_is_not_one_of_the_anchors():
    from timing_luck import HEADLINE_SCHEDULE  # noqa: PLC0415

    assert HEADLINE_SCHEDULE.label == "semi_monthly"
    assert HEADLINE_SCHEDULE.is_anchor_variant is False
    assert HEADLINE_SCHEDULE not in enumerate_schedules()
    assert all(s.is_anchor_variant for s in enumerate_schedules())


def test_headline_calendar_matches_the_pipeline_builder():
    """The headline row must be the headline, not a reimplementation of it."""
    from preprocessing import build_rebalance_calendar  # noqa: PLC0415
    from timing_luck import HEADLINE_SCHEDULE  # noqa: PLC0415

    idx = _calendar("2015-01-05", 1500)
    pd.testing.assert_index_equal(
        build_evaluation_calendar(idx, HEADLINE_SCHEDULE),
        build_rebalance_calendar(idx, "semi_monthly"),
    )


def test_headline_schedule_rejects_an_anchor():
    with pytest.raises(ValueError):
        EvaluationSchedule("semi_monthly", 3)


def test_summary_excludes_non_anchor_rows_from_the_dispersion():
    """Adding the headline row must not move the range it is compared against."""
    anchors = pd.DataFrame(
        {
            "sma_length": [200] * 3,
            "label": ["daily", "weekly_01", "monthly_07"],
            "is_anchor_variant": [True, True, True],
            "cagr": [0.06, 0.05, 0.08],
            "sharpe": [0.30, 0.25, 0.40],
        }
    )
    base = timing_luck_summary(anchors, index_cagr=0.10).iloc[0]

    # A headline row far outside the anchor range, which would widen it if the
    # summary counted it.
    with_headline = pd.concat(
        [
            anchors,
            pd.DataFrame(
                {
                    "sma_length": [200],
                    "label": ["semi_monthly"],
                    "is_anchor_variant": [False],
                    "cagr": [0.20],
                    "sharpe": [0.90],
                }
            ),
        ],
        ignore_index=True,
    )
    out = timing_luck_summary(with_headline, index_cagr=0.10).iloc[0]

    assert out["n_variants"] == 3
    assert out["cagr_range"] == pytest.approx(base["cagr_range"])
    assert out["sharpe_range"] == pytest.approx(base["sharpe_range"])
    assert out["cagr_max"] == pytest.approx(0.08)


def test_summary_scales_the_range_against_the_headline_gap_too():
    variants = pd.DataFrame(
        {
            "sma_length": [200] * 4,
            "label": ["daily", "weekly_01", "monthly_07", "semi_monthly"],
            "is_anchor_variant": [True, True, True, False],
            "cagr": [0.06, 0.05, 0.08, 0.084],
            "sharpe": [0.30, 0.25, 0.40, 0.36],
        }
    )
    row = timing_luck_summary(variants, index_cagr=0.10).iloc[0]

    assert row["cagr_headline"] == pytest.approx(0.084)
    assert row["gap_headline_to_index"] == pytest.approx(0.016)
    assert row["range_over_headline_gap"] == pytest.approx(0.03 / 0.016)
    # The daily-gap scaling is unchanged by the extra row.
    assert row["gap_daily_to_index"] == pytest.approx(0.04)
    assert row["range_over_daily_gap"] == pytest.approx(0.75)
    # Headline sits above all three anchors here, so it ranks 4th of 3 + itself.
    assert row["headline_rank_among_anchors"] == 4

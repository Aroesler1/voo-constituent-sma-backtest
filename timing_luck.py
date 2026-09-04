"""Signal-evaluation schedules and the rebalance-timing-luck sweep.

A moving-average rule is usually reported at one evaluation frequency and one
arbitrary anchor inside that frequency: "monthly" almost always means the last
or the first trading day of the month. Hoffstein, Sibears & Faber, "Rebalance
Timing Luck: The Difference between Hired and Fired" (*The Journal of Index
Investing* 10(1), 2019, 27-36; SSRN 3319045) name the resulting dispersion
*rebalance timing luck*: the standard deviation of returns across otherwise
identical portfolios that differ only in which day inside the period they
rebalance on. Their follow-up with Braun, "Rebalance Timing Luck: The (Dumb)
Luck of Smart Beta" (SSRN 3673910, 2020), measures it above 100 bps annualised
for long-only factor indices.

This module enumerates the 27 schedules the brief asks for (1 daily + 5 weekly
+ 21 monthly) and runs the SMA sweep against every one of them.

Anchor convention. Both the weekly and the monthly anchors are **trading-day
ordinals**, not calendar weekdays or calendar days: weekly variant *k* rebalances
on the k-th trading day of each week, monthly variant *n* on the n-th trading
day of each month. In a full five-day week the k-th trading day is the k-th
weekday, so the weekly leg matches the brief's "each of the five weekdays"
whenever no holiday intervenes; in a short week it still fires exactly once.
That matters because the point of the sweep is to isolate *which* day, holding
*how often* fixed. Calendar-weekday anchors would silently drop the Monday
variant's rebalance in every holiday week and confound anchor with frequency.

Months shorter than the requested ordinal (a 19-trading-day February against
variant 21) fall back to the month's last trading day, for the same reason.
``schedule_diagnostics`` reports how often that fallback fires.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

WEEKLY_ANCHORS = (1, 2, 3, 4, 5)
MONTHLY_ANCHORS = tuple(range(1, 22))


@dataclass(frozen=True)
class EvaluationSchedule:
    """One signal-evaluation schedule: a frequency plus an anchor inside it."""

    frequency: str  # 'daily' | 'weekly' | 'monthly'
    anchor: int | None = None

    def __post_init__(self) -> None:
        if self.frequency not in {"daily", "weekly", "monthly"}:
            raise ValueError("frequency must be 'daily', 'weekly' or 'monthly'.")
        if self.frequency == "daily":
            if self.anchor is not None:
                raise ValueError("The daily schedule takes no anchor.")
            return
        if self.anchor is None:
            raise ValueError(f"The {self.frequency} schedule requires an anchor.")
        limit = 5 if self.frequency == "weekly" else 21
        if not 1 <= int(self.anchor) <= limit:
            raise ValueError(f"{self.frequency} anchor must be in 1..{limit}.")

    @property
    def label(self) -> str:
        """Stable short name used as a column key and a plot tick."""
        if self.frequency == "daily":
            return "daily"
        return f"{self.frequency}_{int(self.anchor):02d}"


def enumerate_schedules() -> list[EvaluationSchedule]:
    """Return the 27 schedules: daily, 5 weekly anchors, 21 monthly anchors."""
    out = [EvaluationSchedule("daily")]
    out += [EvaluationSchedule("weekly", k) for k in WEEKLY_ANCHORS]
    out += [EvaluationSchedule("monthly", n) for n in MONTHLY_ANCHORS]
    return out


def _period_keys(index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    """Group trading days into weeks or months."""
    if frequency == "weekly":
        iso = index.isocalendar()
        return (iso["year"].to_numpy() * 100 + iso["week"].to_numpy()).astype(np.int64)
    return (index.year.to_numpy() * 100 + index.month.to_numpy()).astype(np.int64)


def build_evaluation_calendar(
    index: pd.DatetimeIndex,
    schedule: EvaluationSchedule,
) -> pd.DatetimeIndex:
    """Return the execution dates implied by an evaluation schedule.

    The backtest engine reads a signal as of the previous close and executes at
    the next open, so the calendar it consumes holds *execution* dates. Every
    date returned here is therefore the trading day after an evaluation day.

    Args:
        index: Trading calendar, ascending.
        schedule: The evaluation schedule.

    Returns:
        Execution dates, a subset of ``index`` excluding its first element.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values().unique()
    if len(idx) < 2:
        return pd.DatetimeIndex([])

    if schedule.frequency == "daily":
        return idx[1:]

    keys = _period_keys(idx, schedule.frequency)
    anchor = int(schedule.anchor)

    # Ordinal position of each trading day inside its week/month, 1-based.
    change = np.empty(len(keys), dtype=bool)
    change[0] = True
    change[1:] = keys[1:] != keys[:-1]
    group_start = np.maximum.accumulate(np.where(change, np.arange(len(keys)), 0))
    ordinal = np.arange(len(keys)) - group_start + 1

    period_length = pd.Series(ordinal).groupby(keys).transform("max").to_numpy()
    # Short periods fall back to their own last trading day so that every
    # anchor variant fires exactly once per period.
    target = np.minimum(anchor, period_length)
    eval_positions = np.flatnonzero(ordinal == target)

    # Execute on the trading day after the evaluation day; an evaluation on the
    # sample's final day has no execution day and is dropped.
    exec_positions = eval_positions + 1
    exec_positions = exec_positions[exec_positions < len(idx)]
    return pd.DatetimeIndex(idx[exec_positions])


def schedule_diagnostics(index: pd.DatetimeIndex, schedule: EvaluationSchedule) -> dict[str, float]:
    """Count rebalances and short-period fallbacks for one schedule."""
    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values().unique()
    calendar = build_evaluation_calendar(idx, schedule)
    if schedule.frequency == "daily":
        n_short = 0
        n_periods = len(idx) - 1
    else:
        keys = _period_keys(idx, schedule.frequency)
        period_lengths = pd.Series(keys).groupby(keys).size()
        n_periods = int(len(period_lengths))
        n_short = int((period_lengths < int(schedule.anchor)).sum())
    return {
        "label": schedule.label,
        "n_rebalances": int(len(calendar)),
        "n_periods": int(n_periods),
        "n_short_period_fallbacks": int(n_short),
    }


def timing_luck_summary(
    variants: pd.DataFrame,
    *,
    index_cagr: float,
    group_col: str = "sma_length",
) -> pd.DataFrame:
    """Summarise dispersion across evaluation days, per SMA length.

    Args:
        variants: One row per (length, schedule) with ``cagr``, ``sharpe`` and a
            ``label`` column; the daily variant must be labelled ``daily``.
        index_cagr: Buy-and-hold CAGR of the benchmark, used to scale the range.
        group_col: Column identifying the rule whose variants are compared.

    Returns:
        One row per rule with the range and standard deviation of CAGR and
        Sharpe, and the CAGR range as a fraction of the daily-rule-to-index gap.
    """
    required = {group_col, "label", "cagr", "sharpe"}
    missing = required - set(variants.columns)
    if missing:
        raise ValueError(f"variants is missing columns: {sorted(missing)}")

    rows: list[dict[str, float]] = []
    for key, grp in variants.groupby(group_col, sort=True):
        cagr = grp["cagr"].astype(float)
        sharpe = grp["sharpe"].astype(float)
        daily_rows = grp.loc[grp["label"] == "daily", "cagr"]
        daily_cagr = float(daily_rows.iloc[0]) if len(daily_rows) else np.nan

        # The gap the dispersion is measured against: how far the daily rule
        # sits below simply holding the index. A range that is a large fraction
        # of this gap means the reported shortfall is mostly calendar choice.
        gap = float(index_cagr) - daily_cagr
        cagr_range = float(cagr.max() - cagr.min())

        rows.append(
            {
                group_col: key,
                "n_variants": int(len(grp)),
                "cagr_daily": daily_cagr,
                "cagr_min": float(cagr.min()),
                "cagr_max": float(cagr.max()),
                "cagr_range": cagr_range,
                "cagr_std": float(cagr.std(ddof=1)),
                "sharpe_min": float(sharpe.min()),
                "sharpe_max": float(sharpe.max()),
                "sharpe_range": float(sharpe.max() - sharpe.min()),
                "sharpe_std": float(sharpe.std(ddof=1)),
                "gap_daily_to_index": gap,
                "range_over_gap": cagr_range / gap if np.isfinite(gap) and gap != 0.0 else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)

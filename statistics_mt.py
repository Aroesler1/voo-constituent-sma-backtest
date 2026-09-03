"""Multiple-testing-aware backtest statistics.

Probabilistic and Deflated Sharpe Ratio, Bailey & Lopez de Prado:
- "The Sharpe Ratio Efficient Frontier" (2012)
- "The Deflated Sharpe Ratio" (2014)

Every Sharpe in this module is PER-PERIOD (daily), not annualized.
Named statistics_mt to avoid shadowing the stdlib `statistics` module,
whose NormalDist we use.
"""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Optional, Sequence

import numpy as np
import pandas as pd

_NORMAL = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


def per_period_sharpe(returns: pd.Series) -> float:
    s = pd.to_numeric(returns, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0
    std = float(s.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(s.mean()) / std


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P[true per-period Sharpe > sr_benchmark], adjusted for skew/kurtosis."""
    s = pd.to_numeric(returns, errors="coerce").dropna()
    n = len(s)
    if n < 3:
        return 0.5
    sr = per_period_sharpe(s)
    skew = float(s.skew())
    kurt = float(s.kurt()) + 3.0  # pandas kurt() is excess
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(_NORMAL.cdf(z))


def expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """Expected max per-period Sharpe across n_trials zero-skill strategies."""
    n = max(int(n_trials), 1)
    if n == 1 or var_trial_sr <= 0:
        return 0.0
    sd = math.sqrt(var_trial_sr)
    z1 = _NORMAL.inv_cdf(1.0 - 1.0 / n)
    z2 = _NORMAL.inv_cdf(1.0 - 1.0 / (n * math.e))
    return float(sd * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def deflated_sharpe(
    returns: pd.Series,
    n_trials: int,
    trial_sharpes: Optional[Sequence[float]] = None,
) -> float:
    """DSR: PSR against the expected max of n_trials noise strategies."""
    s = pd.to_numeric(returns, errors="coerce").dropna()
    if trial_sharpes is not None and len(trial_sharpes) >= 2:
        var_trial = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
    else:
        var_trial = 1.0 / max(len(s) - 1, 1)
    sr_star = expected_max_sharpe(n_trials=n_trials, var_trial_sr=var_trial)
    return probabilistic_sharpe(s, sr_benchmark=sr_star)


# ---------------------------------------------------------------------------
# Family-wise error control across a strategy sweep
# ---------------------------------------------------------------------------


def _circular_block_indices(
    n_obs: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular block bootstrap indices.

    Blocks rather than individual observations, because strategy returns are
    serially dependent and an i.i.d. resample would understate the variance of
    the maximum statistic -- which is precisely the quantity the procedure
    needs to get right.
    """
    n_blocks = int(np.ceil(n_obs / block_size))
    starts = rng.integers(0, n_obs, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel() % n_obs
    return idx[:n_obs]


def romano_wolf_stepdown(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    n_boot: int = 1000,
    block_size: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Romano-Wolf (2005) stepwise multiple test on a family of strategies.

    The Deflated Sharpe Ratio asks whether ONE selected configuration beats the
    expected best of N noise strategies. This asks a different and complementary
    question: across the whole sweep, WHICH configurations have a mean excess
    return significantly above zero, while controlling the family-wise error
    rate at `alpha`?

    Controlling FWER matters because a sweep is a family. Testing twelve
    configurations at 5% each gives roughly a 46% chance of at least one false
    positive; the stepdown procedure holds the probability of ANY false
    rejection at 5% instead, and unlike a Bonferroni correction it gains power
    by using the observed dependence between strategies rather than assuming the
    worst case.

    Procedure: bootstrap the joint distribution of the centred t-statistics with
    a circular block bootstrap, take the (1-alpha) quantile of the MAXIMUM over
    the strategies not yet rejected, reject anything exceeding it, and repeat
    until no further rejections occur.

    Returns one row per strategy with its t-statistic, the critical value it
    faced, the step at which it was rejected (0 if never), and an adjusted
    p-value.
    """
    frame = returns.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    frame = frame.dropna(axis=1, how="all").fillna(0.0)
    n_obs, n_strat = frame.shape
    if n_obs < 30 or n_strat < 2:
        raise ValueError("need at least 30 observations and 2 strategies")

    if block_size is None:
        # Politis-White style rule of thumb; any O(n^(1/3)) choice is defensible
        block_size = max(2, int(round(n_obs ** (1.0 / 3.0))))

    values = frame.to_numpy(dtype=float)
    means = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1) / np.sqrt(n_obs)
    scale[scale <= 0] = np.nan
    t_stats = means / scale

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, n_strat))
    for b in range(n_boot):
        idx = _circular_block_indices(n_obs, block_size, rng)
        sample = values[idx]
        # centred on the observed means: the bootstrap approximates the NULL
        b_scale = sample.std(axis=0, ddof=1) / np.sqrt(n_obs)
        b_scale[b_scale <= 0] = np.nan
        boot[b] = (sample.mean(axis=0) - means) / b_scale

    columns = list(frame.columns)
    rejected_step = {c: 0 for c in columns}
    critical = {c: np.nan for c in columns}
    adjusted_p = {c: np.nan for c in columns}

    remaining = list(range(n_strat))
    step = 0
    while remaining:
        step += 1
        sub = boot[:, remaining]
        max_null = np.nanmax(sub, axis=1)
        crit = float(np.nanquantile(max_null, 1.0 - alpha))

        newly = [k for k in remaining if np.isfinite(t_stats[k]) and t_stats[k] > crit]
        for k in remaining:
            critical[columns[k]] = crit
            # adjusted p: mass of the max-null at or above this statistic
            adjusted_p[columns[k]] = float(np.nanmean(max_null >= t_stats[k]))
        if not newly:
            break
        for k in newly:
            rejected_step[columns[k]] = step
        remaining = [k for k in remaining if k not in newly]

    return pd.DataFrame({
        "strategy": columns,
        "mean": means,
        "t_stat": t_stats,
        "critical_value": [critical[c] for c in columns],
        "adjusted_p": [adjusted_p[c] for c in columns],
        "rejected_at_step": [rejected_step[c] for c in columns],
        "significant": [rejected_step[c] > 0 for c in columns],
    }).sort_values("t_stat", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (CSCV)
# ---------------------------------------------------------------------------


def probability_of_backtest_overfitting(
    returns: pd.DataFrame,
    n_splits: int = 16,
) -> dict:
    """PBO via combinatorially symmetric cross-validation.

    Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of Backtest
    Overfitting", Journal of Computational Finance 20(4), 2017.

    The Deflated Sharpe Ratio asks whether the SELECTED configuration's Sharpe
    survives the fact that N were tried. PBO asks a sharper question about the
    selection procedure itself: if you pick the best configuration in sample,
    how often does it land in the bottom half out of sample? A PBO near 0.5 says
    the in-sample ranking carries no information about out-of-sample ranking,
    which is what overfitting looks like from the outside.

    CSCV avoids the arbitrariness of a single train/test cut. The sample is
    split into `n_splits` contiguous blocks; every way of choosing half of them
    as the training set is enumerated, with the complement as the test set. That
    is symmetric (each block appears in training exactly as often as in testing)
    and it uses the whole sample rather than privileging one split point.

    For each combination the in-sample best configuration n* is found, its
    out-of-sample rank r among the N configurations is taken (1 = worst), and
    the relative rank omega = r / (N + 1) is mapped to the logit
    lambda = log(omega / (1 - omega)). PBO is the fraction of combinations with
    lambda <= 0.

    Parameters
    ----------
    returns : DataFrame
        One column per configuration, one row per period. Columns must be the
        SAME strategy family evaluated on the SAME dates, which is what makes
        the rank comparison meaningful.
    n_splits : int
        Number of contiguous blocks; must be even. 16 gives C(16,8) = 12,870
        combinations, the value used in the original paper.

    Returns
    -------
    dict with `pbo`, `n_combinations`, `n_configs`, `median_oos_rank`,
    `logits` and `oos_ranks`.

    A caveat that matters when N is small: the out-of-sample rank can only take
    N distinct values, so PBO is quantized. With five configurations omega is
    one of {1/6, ..., 5/6} and lambda <= 0 means "ranked third or worse of
    five", the median rank included. PBO is coarse here and should be read as
    an indicator, not a precise probability.
    """
    frame = returns.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all").dropna(axis=1, how="all").fillna(0.0)
    n_obs, n_cfg = frame.shape
    if n_cfg < 2:
        raise ValueError("need at least 2 configurations")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even so the halves are the same size")
    if n_obs < 2 * n_splits:
        raise ValueError(f"need at least {2 * n_splits} observations for {n_splits} splits")

    rows_per_block = n_obs // n_splits
    usable = rows_per_block * n_splits
    # Trailing observations that do not fill a whole block are dropped rather
    # than folded into the last one, which would make the blocks unequal and
    # break the symmetry the procedure is named for.
    values = frame.to_numpy(dtype=float)[:usable]
    blocks = values.reshape(n_splits, rows_per_block, n_cfg)

    # Block-level sufficient statistics: any union of blocks has an exact mean
    # and variance from these, so no combination needs to materialise its data.
    counts = np.full(n_splits, rows_per_block, dtype=float)
    sums = blocks.sum(axis=1)
    sumsq = (blocks ** 2).sum(axis=1)

    def _sharpe(selected: list[int]) -> np.ndarray:
        n = counts[selected].sum()
        total = sums[selected].sum(axis=0)
        total_sq = sumsq[selected].sum(axis=0)
        mean = total / n
        var = (total_sq - n * mean ** 2) / (n - 1.0)
        std = np.sqrt(np.maximum(var, 0.0))
        return np.divide(mean, std, out=np.zeros_like(mean), where=std > 0)

    half = n_splits // 2
    all_blocks = set(range(n_splits))
    logits: list[float] = []
    oos_ranks: list[int] = []

    for train in itertools.combinations(range(n_splits), half):
        train_list = list(train)
        test_list = sorted(all_blocks - set(train))

        best = int(np.argmax(_sharpe(train_list)))
        sr_oos = _sharpe(test_list)
        # ranks 1..N with 1 = worst; ties broken by position, which is
        # immaterial at N=5 and avoids a fractional rank the logit cannot take
        rank = int(np.argsort(np.argsort(sr_oos))[best]) + 1

        omega = rank / (n_cfg + 1.0)
        logits.append(float(np.log(omega / (1.0 - omega))))
        oos_ranks.append(rank)

    logit_arr = np.asarray(logits, dtype=float)
    return {
        "pbo": float(np.mean(logit_arr <= 0.0)),
        "n_combinations": len(logits),
        "n_configs": n_cfg,
        "n_splits": n_splits,
        "obs_used": usable,
        "median_oos_rank": float(np.median(oos_ranks)),
        "logits": logit_arr,
        "oos_ranks": np.asarray(oos_ranks, dtype=int),
    }

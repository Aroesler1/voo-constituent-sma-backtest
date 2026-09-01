"""Multiple-testing-aware backtest statistics.

Probabilistic and Deflated Sharpe Ratio, Bailey & Lopez de Prado:
- "The Sharpe Ratio Efficient Frontier" (2012)
- "The Deflated Sharpe Ratio" (2014)

Every Sharpe in this module is PER-PERIOD (daily), not annualized.
Named statistics_mt to avoid shadowing the stdlib `statistics` module,
whose NormalDist we use.
"""

from __future__ import annotations

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

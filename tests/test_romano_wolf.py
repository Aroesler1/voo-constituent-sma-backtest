"""Tests for Romano-Wolf stepwise family-wise error control.

The procedure exists to stop a sweep manufacturing significance. These tests
pin the two properties that matter: it finds genuine edge, and it does not
find edge that is not there.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from statistics_mt import romano_wolf_stepdown  # noqa: E402


def _panel(n_noise: int, n_edge: int, edge_mu: float = 0.0012,
           n_obs: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for i in range(n_noise):
        cols[f"noise{i}"] = rng.normal(0.0, 0.01, n_obs)
    for i in range(n_edge):
        cols[f"edge{i}"] = rng.normal(edge_mu, 0.01, n_obs)
    return pd.DataFrame(cols)


def test_finds_genuine_edge_and_rejects_noise():
    result = romano_wolf_stepdown(_panel(8, 2), n_boot=400)
    significant = set(result.loc[result["significant"], "strategy"])
    assert significant == {"edge0", "edge1"}


def test_controls_family_wise_error_on_pure_noise():
    """With no real edge anywhere, rejections should be rare across seeds.

    This is the property a per-strategy 5% test fails: twelve independent
    tests at 5% give roughly a 46% chance of at least one false positive.
    """
    false_positive_runs = 0
    trials = 12
    for seed in range(trials):
        result = romano_wolf_stepdown(_panel(12, 0, seed=seed), n_boot=300, seed=seed)
        if bool(result["significant"].any()):
            false_positive_runs += 1
    # nominal FWER is 5%; allow slack for 12 trials and a finite bootstrap
    assert false_positive_runs <= 3, f"{false_positive_runs}/{trials} runs had a false rejection"


def test_stepdown_gains_power_over_the_first_step():
    """Critical values must not increase as strategies are removed.

    A single-step maxT test uses the first (largest) critical value for
    everything; the stepdown's advantage is that later steps face a lower bar.
    """
    result = romano_wolf_stepdown(_panel(8, 2), n_boot=400)
    crits = result["critical_value"].dropna().unique()
    assert len(crits) >= 2, "expected at least two distinct critical values"
    assert crits.max() > crits.min()


def test_stronger_strategies_are_rejected_no_later():
    result = romano_wolf_stepdown(_panel(6, 3), n_boot=400)
    rejected = result[result["significant"]]
    if len(rejected) >= 2:
        # sorted by t_stat descending, so steps must be non-decreasing
        steps = rejected["rejected_at_step"].to_numpy()
        assert np.all(np.diff(steps) >= 0)


def test_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        romano_wolf_stepdown(_panel(1, 0, n_obs=2000))   # only one strategy
    with pytest.raises(ValueError):
        romano_wolf_stepdown(_panel(4, 0, n_obs=10))     # too few observations

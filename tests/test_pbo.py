"""Tests for the Probability of Backtest Overfitting (CSCV).

Bailey, Borwein, Lopez de Prado & Zhu (2017). What is pinned here is that PBO
separates a sweep in which one configuration is genuinely better from one in
which the configurations are indistinguishable, because that separation is the
only thing the statistic is for.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from statistics_mt import probability_of_backtest_overfitting  # noqa: E402


def _panel(n_obs=2000, n_cfg=5, seed=0, edge=None):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, (n_obs, n_cfg))
    frame = pd.DataFrame(data, columns=[f"sma_{150 + 25 * i}" for i in range(n_cfg)],
                         index=pd.date_range("2000-01-03", periods=n_obs, freq="B"))
    if edge is not None:
        frame.iloc[:, edge] += 0.004      # a genuinely, persistently better config
    return frame


def test_pbo_is_zero_when_one_configuration_is_genuinely_best():
    """If the in-sample winner is also the out-of-sample winner on every split,
    the selection procedure is not overfitting and PBO must say so."""
    result = probability_of_backtest_overfitting(_panel(edge=2), n_splits=16)
    assert result["pbo"] == pytest.approx(0.0, abs=1e-9)
    assert result["median_oos_rank"] == 5.0
    assert result["n_combinations"] == 12870


def _tied_panel(n_obs=2000, n_cfg=5, seed=0):
    """Configurations that are noisy but exactly tied over the full sample.

    Each column is de-meaned, so no configuration has any real edge and any
    in-sample lead is pure luck. This isolates the mechanism CSCV is built on:
    the training and test blocks are complementary, so a configuration that got
    lucky in training must give that luck back in testing. Simply drawing
    independent noise does not test this -- one column will win the full sample
    by chance and then win both halves, which is why a raw-noise panel's PBO
    swings with the seed (0.88 at seed 0, 0.17 at seed 1).
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, (n_obs, n_cfg))
    data = data - data.mean(axis=0, keepdims=True)
    return pd.DataFrame(data, columns=[f"sma_{150 + 25 * i}" for i in range(n_cfg)],
                        index=pd.date_range("2000-01-03", periods=n_obs, freq="B"))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pbo_is_high_when_configurations_are_exactly_tied(seed):
    """No edge anywhere: the in-sample winner should land in the bottom half
    out of sample almost always, whatever the draw."""
    result = probability_of_backtest_overfitting(_tied_panel(seed=seed), n_splits=16)
    assert result["pbo"] > 0.9
    assert result["median_oos_rank"] <= 2.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pbo_separates_genuine_edge_from_a_tied_sweep(seed):
    skill = probability_of_backtest_overfitting(_panel(seed=seed, edge=0), n_splits=16)
    tied = probability_of_backtest_overfitting(_tied_panel(seed=seed), n_splits=16)
    assert skill["pbo"] < 0.1
    assert tied["pbo"] > 0.9


def test_cscv_enumerates_every_symmetric_split():
    from math import comb
    for n_splits in (8, 10, 16):
        result = probability_of_backtest_overfitting(_panel(n_obs=600), n_splits=n_splits)
        assert result["n_combinations"] == comb(n_splits, n_splits // 2)


def test_blocks_are_equal_sized_so_the_split_is_symmetric():
    """Observations that do not fill a whole block are dropped, not folded into
    the last one; unequal blocks would break the symmetry CSCV is named for."""
    result = probability_of_backtest_overfitting(_panel(n_obs=1005), n_splits=16)
    assert result["obs_used"] == (1005 // 16) * 16
    assert result["obs_used"] % 16 == 0


def test_odd_n_splits_is_rejected():
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(_panel(), n_splits=15)


def test_too_few_observations_is_rejected_rather_than_silently_degraded():
    with pytest.raises(ValueError, match="observations"):
        probability_of_backtest_overfitting(_panel(n_obs=20), n_splits=16)


def test_single_configuration_is_rejected():
    with pytest.raises(ValueError, match="2 configurations"):
        probability_of_backtest_overfitting(_panel(n_cfg=1), n_splits=16)


def test_logits_agree_with_the_reported_pbo():
    """PBO is defined as the share of splits with lambda <= 0; recompute it from
    the returned logits so the headline number cannot drift from its inputs."""
    result = probability_of_backtest_overfitting(_panel(seed=3), n_splits=12)
    assert result["pbo"] == pytest.approx(float(np.mean(result["logits"] <= 0.0)))
    assert len(result["logits"]) == result["n_combinations"]
    assert set(np.unique(result["oos_ranks"])) <= set(range(1, result["n_configs"] + 1))

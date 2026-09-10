"""Frequency changes cannot masquerade as calendar anchor uncertainty."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from timing_luck import frequency_isolated_summary


def _variants():
    return pd.DataFrame({
        "sma_length": [200] * 6,
        "frequency": ["daily", "weekly", "weekly", "monthly", "monthly", "semi_monthly"],
        "label": ["daily", "weekly_01", "weekly_02", "monthly_01", "monthly_02", "semi_monthly"],
        "cagr": [0, 0.1, 0.1, 0.2, 0.2, 99], "sharpe": [0, 1, 1, 2, 2, 99],
    })


def test_frequency_only_effect_has_no_anchor_luck():
    frequencies, split = frequency_isolated_summary(_variants())
    assert frequencies.cagr_range.eq(0).all()
    assert split.n_anchors.item() == 5
    assert split.between_frequency_share.item() == pytest.approx(1)
    assert split.pooled_cagr_range.item() == pytest.approx(0.2)
    assert np.isnan(frequencies.loc[frequencies.frequency.eq("daily"), "cagr_std"].item())


def test_variance_identity_and_unequal_group_weights():
    v = _variants()
    v.loc[1:4, "cagr"] = [-1, 1, -2, 2]
    frequencies, split = frequency_isolated_summary(v)
    s = split.iloc[0]
    assert s.cagr_total_ss == pytest.approx(10)
    assert s.cagr_within_frequency_ss == pytest.approx(10)
    assert s.cagr_between_frequency_ss == pytest.approx(0)
    assert frequencies.set_index("frequency").loc["monthly", "cagr_range"] == 4


def test_constant_results_have_undefined_variance_share():
    v = _variants()
    v.cagr = 0.1
    _, split = frequency_isolated_summary(v)
    assert split.between_frequency_share.isna().all()


@pytest.mark.parametrize("defect", ["duplicate", "missing", "infinite", "unknown"])
def test_bad_input_fails_loudly(defect):
    v = _variants()
    if defect == "duplicate":
        v = pd.concat([v, v.iloc[[0]]])
    elif defect == "missing":
        v = v.drop(columns="frequency")
    elif defect == "infinite":
        v.loc[0, "cagr"] = np.inf
    else:
        v.loc[0, "frequency"] = "quarterly"
    with pytest.raises(ValueError):
        frequency_isolated_summary(v)


def test_committed_report_reproduction():
    from run_schedule_audit import main
    assert main(["--check"]) == 0

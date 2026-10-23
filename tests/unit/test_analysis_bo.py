"""The analysis rules on hand-made tables: shared f*, paired gains, MDE, selection, verdict."""

import numpy as np
import pandas as pd
import pytest

from anytime_acq.analysis.bo import (
    arm_table,
    final,
    mde_gain,
    paired_gain,
    regrets,
    select_portfolio,
    switching_verdict,
    tasks_for_mde,
)


def _tables():
    # two members (d=2), two tasks; member 1's arm finds a lower value than member 0's f* estimate on seed 11
    best = pd.DataFrame(
        {
            "member": [0, 0, 0, 0, 1, 1, 1, 1],
            "seed": [10, 10, 11, 11, 10, 10, 11, 11],
            "arm": ["a"] * 4 + ["b"] * 4,
            "t": [0, 1] * 4,
            "best_f": [0.5, 0.2, 0.6, 0.3, 0.5, 0.1, 0.6, 0.05],
        }
    )
    fstar = pd.DataFrame({"member": [0, 0, 1, 1], "seed": [10, 11, 10, 11], "fstar": [0.1, 0.1, 0.1, 0.1]})
    return best, fstar, pd.Series({0: 2, 1: 2})


def test_regrets_use_one_fstar_per_task() -> None:
    reg = regrets(*_tables(), eps=0.0)
    fin = final(reg).set_index(["arm", "seed"]).regret
    assert fin[("a", 10)] == pytest.approx(0.1) and fin[("b", 10)] == pytest.approx(0.0)
    assert fin[("a", 11)] == pytest.approx(0.25) and fin[("b", 11)] == pytest.approx(0.0)  # f* lowered to 0.05


def test_paired_gain_is_one_minus_the_geometric_mean_ratio() -> None:
    fin = pd.DataFrame({"d": 2, "seed": [0, 0, 1, 1], "arm": ["a", "b"] * 2, "log_regret": [-2.0, -1.0, -3.0, -3.0]})
    g = paired_gain(fin, 2, "a", "b", n_boot=200)
    assert g["gain"] == pytest.approx(1 - 10**-0.5)  # mean log ratio -0.5
    assert g["frac_a_better"] == 0.5 and g["frac_tied"] == 0.5


def test_mde_and_sample_size_are_inverse() -> None:
    n = tasks_for_mde(0.5, 0.05)
    assert mde_gain(0.5, n) <= 0.05 < mde_gain(0.5, n - 1)


def test_selection_rule() -> None:
    arms = ["ei", "ei0.01", "pi0.05", "pi0.1", "ucb0.9", "ucb0.99", "mean", "mes", "es", "random"]
    vals = [-2.0, -2.1, -1.5, -1.6, -2.5, -2.2, -1.0, -2.3, -1.9, -9.0]  # random best: must be excluded
    t = pd.DataFrame({"d": 4, "arm": arms, "mean_log_regret": vals})
    sel = select_portfolio(t, 4)
    assert sel["base"] == "ucb0.9"
    assert sel["family_best"] == {"ei": "ei0.01", "pi": "pi0.1", "ucb": "ucb0.9"}
    assert sel["portfolio"] == ["ucb0.9", "ei0.01", "pi0.1", "mes", "es", "mean"]


def test_arm_table_sorted_by_mean() -> None:
    fin = pd.DataFrame({"d": 2, "arm": ["x", "y"] * 3, "seed": np.repeat([0, 1, 2], 2), "log_regret": [1, 0] * 3})
    assert list(arm_table(fin).arm) == ["y", "x"]


def _g(gain, lo, hi):
    return {"gain": gain, "gain_lo": lo, "gain_hi": hi}


def test_switching_verdict() -> None:
    ok_null = [_g(0.0, -0.02, 0.02)] * 3
    assert switching_verdict([_g(0.08, 0.02, 0.14)] * 2 + [_g(0, -1, 1)], [_g(0.2, 0.1, 0.3)] * 3, ok_null) == "GO"
    assert switching_verdict([_g(0.01, -0.02, 0.04)] * 3, [_g(0.03, 0.0, 0.06)] * 3, ok_null) == "NO-GO"
    # a small point estimate with a wide CI rules nothing out
    assert switching_verdict([_g(0.01, -0.1, 0.1)] * 3, [_g(0.03, -0.1, 0.15)] * 3, ok_null) == "INCONCLUSIVE"
    assert switching_verdict([_g(0.04, 0.0, 0.08)] * 3, [_g(0.2, 0.1, 0.3)] * 3, ok_null) == "INCONCLUSIVE"
    bad_null = [_g(0.05, 0.01, 0.09), *ok_null[:2]]
    assert switching_verdict([_g(0.08, 0.02, 0.14)] * 3, [_g(0.2, 0.1, 0.3)] * 3, bad_null) == "INVALID"


def test_headroom_verdict() -> None:
    from anytime_acq.analysis.bo import headroom_verdict

    def g(gain: float, lo: float, hi: float) -> dict[str, object]:
        return {"gain": gain, "gain_lo": lo, "gain_hi": hi}

    null = [g(0.0, -0.1, 0.1)] * 2
    assert headroom_verdict([g(0.2, 0.05, 0.3), g(0.01, -0.1, 0.1)], null) == "GO"
    assert headroom_verdict([g(0.2, 0.05, 0.3), g(-0.01, -0.1, 0.1)], null) == "INCONCLUSIVE"
    assert headroom_verdict([g(0.02, -0.05, 0.08), g(0.0, -0.05, 0.09)], null) == "NO-GO"
    assert headroom_verdict([g(0.2, 0.05, 0.3)] * 2, [g(0.1, 0.01, 0.2), g(0, -1, 1)]) == "INVALID"

"""Analysis of the api pipeline (amortized policy improvement, round 1): paired effects on normalized
regret, the cross-fitted advantage map, and the pre-registered rule of notebook 06.

Normalized regret R is the final regret in % of the task's gap f_med - f* (0003 §1). Effects are paired
on tasks: delta = mean_task(R_a - R_b) (negative = a is better), with a percentile bootstrap CI over
tasks, and relative to the mean R of b.
"""

import numpy as np
import pandas as pd

from .stats import bootstrap_ci


def paired_delta(
    fin: pd.DataFrame, a: str, b: str, col: str = "norm_regret", n_boot: int = 10_000
) -> dict[str, object]:
    """fin: one row per (task, arm) with `col`. -> mean paired difference a - b, CI, relative effect."""
    g = fin.pivot_table(index="task", columns="arm", values=col)
    diff = (g[a] - g[b]).dropna().to_numpy()
    lo, hi = bootstrap_ci(diff, n_boot=n_boot)
    ref = float(np.mean(g[b].to_numpy()))
    return {
        "a": a,
        "b": b,
        "n": len(diff),
        "delta": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "ref_mean": ref,
        "rel": float(diff.mean() / ref),
        "rel_lo": float(lo / ref),
        "rel_hi": float(hi / ref),
        "frac_a_better": float((diff < 0).mean()),
        "frac_tied": float((diff == 0).mean()),
    }


def advantage_map(labels: pd.DataFrame, n_cont: int) -> pd.DataFrame:
    """Per labeled state: the cross-fitted advantage of the best labeled candidate over x0 (0003 §8.3,
    here *clairvoyant*: continuations on the true task, so a diagnostic, never a result). The best
    candidate is selected on one half of the continuation replicates and scored on the other, both ways,
    and averaged; in-sample `best_in_sample` (selected and scored on all) shows the optimizer's curse.
    labels: the `labels` table (columns state, j, valid, R0..R{n_cont-1})."""
    half = n_cont // 2
    ks = [f"R{k}" for k in range(n_cont)]
    rows = []
    for s, g in labels.groupby("state"):
        g = g.sort_values("j")
        R = g[ks].to_numpy()  # [nl, M]
        adv = R[:1] - R  # positive = candidate better than x0
        ok = g.valid.to_numpy()
        adv_ok = np.where(ok[:, None], adv, -np.inf)
        cf = []
        for sel, sc in ((slice(0, half), slice(half, n_cont)), (slice(half, n_cont), slice(0, half))):
            j = int(adv_ok[:, sel].mean(1).argmax())
            cf.append(adv[j, sc].mean())
        rows.append(
            {
                "state": s,
                "t": int(g.t.iloc[0]),
                "behaviour": g.behaviour.iloc[0],
                "best_crossfit": float(np.mean(cf)),
                "best_in_sample": float(adv_ok.mean(1).max()),
            }
        )
    return pd.DataFrame(rows)


def api_verdict(main: dict[str, object], null: dict[str, object], worthwhile: float = 0.05) -> str:
    """The pre-registered rule of notebook 06 (paired_delta outputs on the test split, b = EI).

    INVALID if the null head (labels permuted) is better than EI with its CI below 0: a gain without
    information means a bug or leakage. GO if pi_1 at c* lowers mean normalized regret by at least
    `worthwhile` (relative) with the CI below 0. NO-GO if the CI rules out a relative gain of
    `worthwhile` (rel_lo > -worthwhile). Otherwise INCONCLUSIVE."""
    if float(null["hi"]) < 0:  # type: ignore[arg-type]
        return "INVALID"
    if float(main["rel"]) <= -worthwhile and float(main["hi"]) < 0:  # type: ignore[arg-type]
        return "GO"
    if float(main["rel_lo"]) > -worthwhile:  # type: ignore[arg-type]
        return "NO-GO"
    return "INCONCLUSIVE"


def gate_verdict(main: dict[str, object], worthwhile: float = 0.05) -> str:
    """The pre-registered rule of notebook 07 (api_gate), on paired_delta(test, "mb", "ei"): GO if the
    model-based rollout policy lowers mean normalized regret by >= `worthwhile` (relative) with the CI of the
    difference below 0; NO-GO if the CI rules out that gain; otherwise INCONCLUSIVE."""
    if float(main["rel"]) <= -worthwhile and float(main["hi"]) < 0:  # type: ignore[arg-type]
        return "GO"
    if float(main["rel_lo"]) > -worthwhile:  # type: ignore[arg-type]
        return "NO-GO"
    return "INCONCLUSIVE"

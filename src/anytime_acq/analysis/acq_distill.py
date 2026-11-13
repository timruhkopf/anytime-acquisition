"""Summary and pre-registered rule of the distillation of the arms into the acquisition branch (notebook 16)."""

import numpy as np
import pandas as pd

from .stats import bootstrap_ci

GOOD_PICK = 0.01  # the student's pick is "good" if the teacher ranks at most 1% of the candidates above it


def summarize(states: pd.DataFrame, closed: pd.DataFrame, eps: float, n_boot: int) -> pd.DataFrame:
    """Per arm: agreement on the deployment candidate set (`cand0`), on unseen candidates and the dense set, and
    the closed-loop difference in log10 terminal regret (student - teacher; positive = the student is worse)."""
    rows = []
    for arm, g in states.groupby("arm"):
        r: dict[str, float | str] = {"arm": str(arm)}
        for qs, h in g.groupby("query_set"):
            rp = h.rank_pct.to_numpy()
            r[f"{qs}:top1"] = float(h.top1.to_numpy().mean())
            r[f"{qs}:good_pick"] = float((rp <= GOOD_PICK).mean())
            r[f"{qs}:rank_pct_p90"] = float(np.quantile(rp, 0.9))
            r[f"{qs}:spearman"] = float(h.spearman.to_numpy().mean())
            r[f"{qs}:spearman_top"] = float(h.spearman_top.to_numpy().mean())
        c = closed[closed.arm == arm]
        x = np.log10(c.regret_student.to_numpy() + eps) - np.log10(c.regret_teacher.to_numpy() + eps)
        lo, hi = bootstrap_ci(x, n_boot=n_boot)
        r |= {"cl:diff": float(x.mean()), "cl:lo": lo, "cl:hi": hi, "cl:same_run": float(c.same_run.to_numpy().mean())}
        r["cl:same_steps"] = float(c.same_steps.to_numpy().mean())
        rows.append(r)
    return pd.DataFrame(rows)


def distill_verdict(summary: pd.DataFrame, good: float = 0.9, bad: float = 0.5, cl_tol: float = 0.1) -> str:
    """GO: for every arm the student's pick on the deployment candidates is good in >= 90% of states and its
    closed-loop log10 regret is within 0.1 of the teacher's. NO-GO: for some arm good picks in < 50% of states."""
    g, d = summary["cand0:good_pick"].to_numpy(), np.abs(summary["cl:diff"].to_numpy())
    if bool((g >= good).all()) and bool((d <= cl_tol).all()):
        return "GO"
    if bool((g < bad).any()):
        return "NO-GO"
    return "INCONCLUSIVE"

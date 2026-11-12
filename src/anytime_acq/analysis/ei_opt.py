"""Aggregates and the pre-registered rule of notebook 11 (ei_opt; tasks/ei_opt.py)."""

import itertools

import numpy as np
import pandas as pd

DET_TOL = 1e-6  # replicate sd of the final regret (PFN bins) below which a task counts as deterministic
FAIL_BINS = 10.0  # a run "fails" if its final regret exceeds this many PFN bins


def pick_dispersion(st: pd.DataFrame) -> pd.DataFrame:
    """Per (state, variant): the mean pairwise distance between replicate picks, and the sd over replicates of
    f at the pick (PFN bins)."""
    xc = [c for c in st.columns if c.startswith("x") and c[1:].isdigit()]
    rows = []
    for key, g in st.groupby(["state", "variant"]):
        s, v = key  # type: ignore[misc]
        P = g[xc].to_numpy()
        dists = [float(np.linalg.norm(P[i] - P[j])) for i, j in itertools.combinations(range(len(P)), 2)]
        rows.append(
            {
                "state": s,
                "variant": v,
                "t": int(g.t.iloc[0]),
                "gap_bins": float(g.gap_bins.iloc[0]),
                "pick_dist": float(np.mean(dists)) if dists else 0.0,
                "f_sd_bins": float(g.f_rel_bins.std(ddof=0)),
                "ei_ratio": float(g.ei_ratio.mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize(st: pd.DataFrame, fin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (spread per task x variant, summary per variant)."""
    spread = (
        fin.groupby(["task", "variant"])
        .agg(sd_norm=("norm_regret", lambda v: v.std(ddof=0)), sd_bins=("regret_bins", lambda v: v.std(ddof=0)))
        .reset_index()
    )
    disp = pick_dispersion(st)
    rows = []
    for v, g in fin.groupby("variant"):
        sp = spread[spread.variant == v]
        dv = disp[disp.variant == v]
        sv = st[st.variant == v]
        per_task = g.groupby("task").norm_regret.mean()
        rows.append(
            {
                "variant": v,
                "ei_ratio_median": float(sv.ei_ratio.median()),
                "ei_ratio_q10": float(sv.ei_ratio.quantile(0.1)),
                "share_ei_ratio_below_0.99": float((sv.ei_ratio < 0.99).mean()),
                "pick_dist_mean": float(dv.pick_dist.mean()),
                "share_states_pick_dist_above_1e-3": float((dv.pick_dist > 1e-3).mean()),
                "f_sd_bins_median": float(dv.f_sd_bins.median()),
                "f_sd_bins_mean": float(dv.f_sd_bins.mean()),
                "norm_regret_mean": float(g.norm_regret.mean()),
                "norm_regret_q90": float(per_task.quantile(0.9)),
                "norm_regret_q95": float(per_task.quantile(0.95)),
                "regret_bins_mean": float(g.regret_bins.mean()),
                "fail_rate": float((g.regret_bins > FAIL_BINS).mean()),
                "rep_sd_norm_median": float(sp.sd_norm.median()),
                "rep_sd_norm_mean": float(sp.sd_norm.mean()),
                "share_tasks_deterministic": float((sp.sd_bins < DET_TOL).mean()),
            }
        )
    return spread, pd.DataFrame(rows)


def ei_opt_verdict(summary: pd.DataFrame, paired: pd.DataFrame, worthwhile: float = 0.05) -> str:
    """Pre-registered (notebook 11): adopt `fixed+refine` as the base policy pi_0 (GO) if it is deterministic
    (>= 99% of tasks with replicate sd of the final regret < DET_TOL) and not worse than the current sampled EI
    by `worthwhile` (relative) on mean normalized regret (upper CI bound of the relative paired difference
    < +worthwhile). NO-GO if it is not deterministic or clearly worse (lower bound > +worthwhile). Else
    INCONCLUSIVE."""
    det = float(summary.loc[summary.variant == "fixed+refine", "share_tasks_deterministic"].iloc[0])
    r = paired.loc[(paired.a == "fixed+refine") & (paired.metric == "norm_regret")].iloc[0]
    if det < 0.99 or float(r.rel_lo) > worthwhile:
        return "NO-GO"
    if float(r.rel_hi) < worthwhile:
        return "GO"
    return "INCONCLUSIVE"

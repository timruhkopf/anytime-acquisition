"""Cross-fitted (luck-corrected) option headroom and the pre-registered rule of notebook 12 (option_headroom)."""

import numpy as np
import pandas as pd

from .stats import bootstrap_ci

GOOD_BINS = (1.0, 10.0)


def family(arm: str) -> str:
    return "nulls" if arm.startswith("ei@") else "arms"


def crossfit_per_state(M: pd.DataFrame, k: int, fam: str) -> pd.DataFrame:
    """Per state: the option chosen on one seed replicate (best advantage, or no deviation if none is > 0),
    scored on the other, averaged over both directions (`H`); the in-sample best (`oracle`); whether it deviated."""
    sub = M[(M.k == k) & (M.option != "base") & (M.arm.map(family) == fam)]
    P = sub.pivot_table(index=["state", "rep"], columns="option", values="adv").unstack("rep")
    reps = sorted(sub.rep.unique())
    assert len(reps) == 2, "cross-fitting needs exactly two replicate seeds"
    a, b = (P.xs(r, axis=1, level="rep") for r in reps)
    out = []
    for sel, sc in ((a, b), (b, a)):
        best = sel.max(axis=1)
        j = sel.idxmax(axis=1)
        gain = np.where(best > 0, [sc.loc[i, j[i]] for i in sel.index], 0.0)
        out.append(pd.Series(gain, index=sel.index))
    H = (out[0] + out[1]) / 2
    oracle = (a.max(axis=1).clip(lower=0) + b.max(axis=1).clip(lower=0)) / 2
    dev = ((a.max(axis=1) > 0).astype(float) + (b.max(axis=1) > 0).astype(float)) / 2
    return pd.DataFrame({"H": H, "oracle": oracle, "deviates": dev})


def crossfit_table(M: pd.DataFrame, ks: list[int], n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (headroom per k and family, with the arms - nulls contrast; per-option summary)."""
    rows = []
    for k in ks:
        per = {f: crossfit_per_state(M, k, f) for f in ("arms", "nulls")}
        for f, d in per.items():
            lo, hi = bootstrap_ci(d.H.to_numpy(), n_boot=n_boot)
            olo, ohi = bootstrap_ci(d.oracle.to_numpy(), n_boot=n_boot)
            rows.append(
                {
                    "k": k,
                    "family": f,
                    "H": float(d.H.mean()),
                    "lo": lo,
                    "hi": hi,
                    "oracle": float(d.oracle.mean()),
                    "oracle_lo": olo,
                    "oracle_hi": ohi,
                    "deviates": float(d.deviates.mean()),
                    "n_states": len(d),
                }
            )
        diff = (per["arms"].H - per["nulls"].H).to_numpy()
        lo, hi = bootstrap_ci(diff, n_boot=n_boot)
        rows.append({"k": k, "family": "arms-nulls", "H": float(diff.mean()), "lo": lo, "hi": hi})
    head = pd.DataFrame(rows)
    base = M[M.option == "base"].set_index(["state", "rep"])
    opts = []
    for name, g in M[M.option != "base"].groupby("option"):
        lo, hi = bootstrap_ci(g.adv.to_numpy(), n_boot=n_boot)
        bg = base.loc[list(zip(g.state, g.rep, strict=True)), "basin_gap_bins"].to_numpy()
        row = {
            "option": name,
            "arm": g.arm.iloc[0],
            "k": int(g.k.iloc[0]),
            "adv_mean": float(g.adv.mean()),
            "lo": lo,
            "hi": hi,
            "adv_median": float(g.adv.median()),
            "share_pos": float((g.adv > 0).mean()),
            "share_neg": float((g.adv < 0).mean()),
            "disagree": float(g.disagree.mean()),
            "pi_xstar": float(g.pi_xstar.mean()),
            "dist_argmin_xstar": float(g.dist_argmin_xstar.mean()),
        }
        for dl in GOOD_BINS:
            og, bgood = g.basin_gap_bins.to_numpy() <= dl, bg <= dl
            row[f"basin_gain_{dl:g}"] = float((og & ~bgood).mean() - (bgood & ~og).mean())
        opts.append(row)
    return head, pd.DataFrame(opts)


def option_verdict(head: pd.DataFrame, k: int, worthwhile: float) -> str:
    """Pre-registered (notebook 12), on the arms' cross-fitted headroom at k: GO if its CI is above 0 and
    H >= `worthwhile` (log10 regret); NO-GO if the CI's upper bound is below `worthwhile`; else INCONCLUSIVE."""
    r = head.loc[(head.family == "arms") & (head.k == k)].iloc[0]
    if float(r.lo) > 0 and float(r.H) >= worthwhile:
        return "GO"
    if float(r.hi) < worthwhile:
        return "NO-GO"
    return "INCONCLUSIVE"

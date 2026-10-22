"""Analysis of the bo pipeline's tables: regrets on a shared f*, paired ratios, the pre-registered rules.

All comparisons are paired on tasks. The effect size is the ratio of geometric-mean final regrets,
10 ** mean_task(log10(r_a + eps) - log10(r_b + eps)), reported as a gain 1 - ratio (positive = a is
better). CIs are percentile bootstraps over tasks.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .stats import bootstrap_ci

EPS = 1e-4
FAMILIES = ("ei", "pi", "ucb")
FIXED_ARMS = ("mes", "es", "mean")
RANDOM = "random"


def regrets(best: pd.DataFrame, fstar: pd.DataFrame, d_of_member: pd.Series, eps: float = EPS) -> pd.DataFrame:
    """best/fstar tables of a snapshot (with `member`) -> best + d, regret, log_regret.

    f* per (d, seed) is the lowest of every member's estimate and every evaluated value, so every method
    on a task is measured against the same f* (standard S6)."""
    b = best.assign(d=best.member.map(d_of_member))
    f = fstar.assign(d=fstar.member.map(d_of_member)).groupby(["d", "seed"]).fstar.min()
    f = np.minimum(f, b.groupby(["d", "seed"]).best_f.min())
    b = b.join(f.rename("fstar_shared"), on=["d", "seed"])
    b["regret"] = (b.best_f - b.fstar_shared).clip(lower=0)
    b["log_regret"] = np.log10(b.regret + eps)
    return b


def final(reg: pd.DataFrame) -> pd.DataFrame:
    """Rows at the last step: one per (d, arm, seed)."""
    last = reg.groupby("d").t.transform("max")
    return reg.loc[reg.t == last, ["d", "arm", "seed", "regret", "log_regret"]]


def arm_table(fin: pd.DataFrame) -> pd.DataFrame:
    """Mean final log10 regret per (d, arm), with bootstrap CI over tasks."""
    rows = []
    for (d, arm), g in fin.groupby(["d", "arm"]):
        lo, hi = bootstrap_ci(g.log_regret.to_numpy())
        rows.append({"d": d, "arm": arm, "n": len(g), "mean_log_regret": g.log_regret.mean(), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).sort_values(["d", "mean_log_regret"]).reset_index(drop=True)


def paired_gain(fin: pd.DataFrame, d: int, a: str, b: str, n_boot: int = 10_000) -> dict[str, object]:
    """Gain of arm a over arm b at d: 1 - geometric-mean regret ratio, with a bootstrap CI over tasks."""
    g = fin.loc[fin.d == d].pivot_table(index="seed", columns="arm", values="log_regret")
    diff = (g[a] - g[b]).dropna().to_numpy()
    lo, hi = bootstrap_ci(diff, n_boot=n_boot)
    return {
        "d": d,
        "a": a,
        "b": b,
        "n": len(diff),
        "gain": float(1 - 10 ** diff.mean()),
        "gain_lo": float(1 - 10**hi),  # a larger log-ratio is a smaller gain
        "gain_hi": float(1 - 10**lo),
        "sd_log_ratio": float(diff.std(ddof=1)),
        "frac_a_better": float((diff < 0).mean()),
        "frac_tied": float((diff == 0).mean()),
    }


def mde_gain(sd_log_ratio: float, n: int, z: float = 1.96, power_z: float = 0.84) -> float:
    """Smallest gain detectable with 80% power (two-sided 5%), from the sd of the per-task log10 ratio."""
    return float(1 - 10 ** (-(z + power_z) * sd_log_ratio / np.sqrt(n)))


def tasks_for_mde(sd_log_ratio: float, target_gain: float, z: float = 1.96, power_z: float = 0.84) -> int:
    """Tasks needed so that mde_gain(sd, n) <= target_gain."""
    delta = -np.log10(1 - target_gain)
    return int(np.ceil(((z + power_z) * sd_log_ratio / delta) ** 2))


def select_portfolio(table: pd.DataFrame, d: int) -> dict[str, object]:
    """The pre-registered selection rule (notebook 01): B* = lowest mean final log regret (random search
    excluded); per family the best hyperparameter; portfolio = [B*, then the rest of
    {ei*, pi*, ucb*, mes, es, mean}]."""
    t = table.loc[(table.d == d) & (table.arm != RANDOM)].set_index("arm").mean_log_regret
    base = str(t.idxmin())
    best_of = {fam: str(t[[a for a in t.index if _family(a) == fam]].idxmin()) for fam in FAMILIES}
    pool = [best_of[f] for f in FAMILIES] + [a for a in FIXED_ARMS if a in t.index]
    return {"base": base, "family_best": best_of, "portfolio": [base] + [a for a in pool if a != base]}


def _family(arm: str) -> str:
    return arm.rstrip("0123456789.")


def switching_verdict(
    belief: Sequence[dict[str, object]],
    privileged: Sequence[dict[str, object]],
    null: Sequence[dict[str, object]],
    worthwhile: float = 0.05,
    ruled_out: float = 0.10,
) -> str:
    """The pre-registered decision rule (notebook 02); each argument holds one paired_gain per d.

    INVALID if a null CI excludes 0. GO if the belief controller's gain over B* is >= `worthwhile` with
    its CI above 0 at >= 2 of the d. NO-GO if the privileged controller's (the reference with task
    knowledge) CI upper bound is below `ruled_out` at >= 2 of the d: even with task knowledge no gain of
    that size is plausible. Otherwise INCONCLUSIVE (e.g. too imprecise to rule anything out)."""
    if any(n["gain_lo"] > 0 or n["gain_hi"] < 0 for n in null):
        return "INVALID"
    if sum(b["gain"] >= worthwhile and b["gain_lo"] > 0 for b in belief) >= 2:
        return "GO"
    if sum(p["gain_hi"] < ruled_out for p in privileged) >= 2:
        return "NO-GO"
    return "INCONCLUSIVE"


def headroom_verdict(
    privileged: Sequence[dict[str, object]], null: Sequence[dict[str, object]], worthwhile: float = 0.10
) -> str:
    """The pre-registered rule of notebook 04 (bbob_switching), per surrogate; one paired_gain per d.

    INVALID if a null CI excludes 0. GO (luck-corrected headroom) if the privileged controller gains
    >= `worthwhile` over B* with its CI above 0 at >= 1 d and a positive point estimate at every other d.
    NO-GO if its CI upper bound is below `worthwhile` at every d. Otherwise INCONCLUSIVE."""
    if any(n["gain_lo"] > 0 or n["gain_hi"] < 0 for n in null):
        return "INVALID"
    strong = [p["gain"] >= worthwhile and p["gain_lo"] > 0 for p in privileged]
    if any(strong) and all(s or p["gain"] > 0 for s, p in zip(strong, privileged, strict=True)):
        return "GO"
    if all(p["gain_hi"] < worthwhile for p in privileged):
        return "NO-GO"
    return "INCONCLUSIVE"

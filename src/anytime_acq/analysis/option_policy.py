"""Value of a one-deviation option policy from exact option labels, its comparators, and the pre-registered rule
of notebook 13 (option_policy).

Arrays are indexed [task, decision step, option]. A decision is (j, o): deviate at decision step j to option o,
or j = -1: never deviate (the base policy). Its value on a seed is the exact advantage A[task, j, o] over the
base (log10 terminal regret, positive = better), or 0.
"""

import numpy as np
import pandas as pd

from .stats import bootstrap_ci


def first_trigger(Q: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """The one-deviation policy: at the first decision step whose best predicted advantage exceeds c, commit to
    that option. Q [N, S, O] -> (j [N] or -1, o [N])."""
    m, a = Q.max(2), Q.argmax(2)
    trig = m > c
    j = np.where(trig.any(1), trig.argmax(1), -1)
    return j, a[np.arange(len(Q)), np.maximum(j, 0)]


def decision_value(A: np.ndarray, j: np.ndarray, o: np.ndarray) -> np.ndarray:
    return np.where(j >= 0, A[np.arange(len(A)), np.maximum(j, 0), o], 0.0)


def threshold_grid(Q: np.ndarray, n: int = 40) -> np.ndarray:
    m = Q.max(2).ravel()
    return np.unique(np.concatenate([[0.0], np.quantile(m, np.linspace(0.0, 1.0, n)), [np.inf]]))


def choose_threshold(Q: np.ndarray, A: np.ndarray) -> float:
    """The threshold with the best mean value on (Q, A); c = inf (never deviate, value 0) is in the grid, and
    ties go to the larger c (the more conservative policy)."""
    best_c, best_v = np.inf, 0.0
    for c in threshold_grid(Q)[::-1]:
        v = float(decision_value(A, *first_trigger(Q, c)).mean())
        if v > best_v + 1e-12:
            best_c, best_v = float(c), v
    return best_c


def static_choice(A: np.ndarray) -> tuple[int, int]:
    """The best single decision for all tasks (a context-free schedule: switch at step j to option o), or
    (-1, 0) if none has a positive mean."""
    m = A.mean(0)
    j, o = np.unravel_index(int(m.argmax()), m.shape)
    return (int(j), int(o)) if float(m.max()) > 0 else (-1, 0)


def shuffled_value(A: np.ndarray, j: np.ndarray, o: np.ndarray) -> np.ndarray:
    """[N]: the expected value on each task of a decision drawn from the decisions of all N tasks: the same
    deviation rate, the same steps and options, but unrelated to the task's own context."""
    N, S, n_opt = A.shape
    w = np.zeros((S, n_opt))
    np.add.at(w, (j[j >= 0], o[j >= 0]), 1.0 / N)
    return (A * w[None]).sum((1, 2))


def oracle_crossfit(A0: np.ndarray, A1: np.ndarray) -> np.ndarray:
    """[N]: the privileged headroom of the one-deviation class: per task, the best decision on one seed (none if
    no advantage is > 0), scored on the other, averaged over both directions."""
    N = len(A0)
    out = np.zeros(N)
    for sel, sc in ((A0, A1), (A1, A0)):
        flat = sel.reshape(N, -1)
        b = flat.argmax(1)
        out += np.where(flat.max(1) > 0, sc.reshape(N, -1)[np.arange(N), b], 0.0) / 2
    return out


def summarize(per_task: pd.DataFrame, variants: list[str], n_boot: int) -> pd.DataFrame:
    """Per variant and seed (0, 1): the mean gain over the base, over the static schedule of the variant's
    portfolio and over its shuffled null, each paired over tasks with a bootstrap CI; plus the comparators."""
    rows = []

    def add(variant: str, seed: int, contrast: str, x: np.ndarray, **extra: float) -> None:
        lo, hi = bootstrap_ci(x, n_boot=n_boot)
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "contrast": contrast,
                "mean": float(x.mean()),
                "lo": lo,
                "hi": hi,
                "n_tasks": len(x),
                **extra,
            }
        )

    for v in variants:
        pf = v.split("/")[0]
        dev = float((per_task[f"{v}:j"] >= 0).mean())
        for s in (0, 1):
            g, st, sh = (per_task[f"{v}:{k}{s}"].to_numpy() for k in ("v", "static", "shuffled"))
            add(v, s, "vs_base", g, deviates=dev)
            add(v, s, "vs_static", g - st, deviates=dev)
            add(v, s, "vs_shuffled", g - sh, deviates=dev)
            if v.endswith("/belief/mlp"):  # the comparators, once per portfolio
                add(f"{pf}/static", s, "vs_base", st)
        if v.endswith("/belief/mlp"):
            add(f"{pf}/oracle", -1, "vs_base", per_task[f"{pf}:oracle"].to_numpy())
    return pd.DataFrame(rows)


def policy_verdict(summary: pd.DataFrame, primary: str, worthwhile: float) -> str:
    """Pre-registered (notebook 13), on the primary variant under seed 0. GO: the policy beats the base (CI > 0)
    and its gain over the best static schedule (the context-dependent part) has CI > 0 and is >= `worthwhile`.
    NO-GO: the upper CI bound of the gain over the static schedule is below `worthwhile`. Else INCONCLUSIVE."""
    r = summary[(summary.variant == primary) & (summary.seed == 0)].set_index("contrast")
    base, ctx = r.loc["vs_base"], r.loc["vs_static"]
    if float(base.lo) > 0 and float(ctx.lo) > 0 and float(ctx["mean"]) >= worthwhile:
        return "GO"
    if float(ctx.hi) < worthwhile:
        return "NO-GO"
    return "INCONCLUSIVE"

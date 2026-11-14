"""Gating one deviation by a score (notebook 15, ei_failure): deviate at step j to option o iff score_j >= c.

Arrays: score [N, S] (per task and decision step), advantages A [N, S, O] (log10 regret, positive = better).
"""

import numpy as np
from scipy.stats import rankdata

Gate = tuple[int, int, float]  # (step index j, option o, threshold c); j = -1: never deviate
NEVER: Gate = (-1, 0, float("inf"))


def gate_choice(score: np.ndarray, A: np.ndarray, n_grid: int = 20) -> Gate:
    """The gate with the best mean value on (score, A). Never deviating (value 0) wins ties, then the ungated
    schedule (c = -inf: every task deviates), then lower thresholds."""
    best, best_v = NEVER, 0.0
    for j in range(A.shape[1]):
        cs = [-np.inf, *np.unique(np.quantile(score[:, j], np.linspace(0.05, 0.95, n_grid - 1)))]
        for o in range(A.shape[2]):
            for c in cs:
                v = float(np.where(score[:, j] >= c, A[:, j, o], 0.0).mean())
                if v > best_v + 1e-12:
                    best, best_v = (j, o, float(c)), v
    return best


def gate_value(score: np.ndarray, A: np.ndarray, gate: Gate) -> tuple[np.ndarray, np.ndarray]:
    """([N] value, [N] bool deviates) of `gate` on (score, A)."""
    j, o, c = gate
    if j < 0:
        return np.zeros(len(A)), np.zeros(len(A), dtype=bool)
    m = score[:, j] >= c
    return np.where(m, A[:, j, o], 0.0), m


def spearman_ci(
    a: np.ndarray, y: np.ndarray, b: np.ndarray | None = None, n_boot: int = 300, seed: int = 0
) -> tuple[float, float, float]:
    """(estimate, lo, hi) of Spearman(a, y), or of Spearman(a, y) - Spearman(b, y) (paired), bootstrap over rows."""

    def rho(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.corrcoef(rankdata(u), rankdata(v))[0, 1])

    def stat(i: np.ndarray) -> float:
        return rho(a[i], y[i]) - (rho(b[i], y[i]) if b is not None else 0.0)

    n = len(y)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    boots = np.array([stat(i) for i in idx])
    return stat(np.arange(n)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

"""Predictive quality of the PFN on BO contexts (research_log/decisions/0007 §4, tests 1-2).

The PFN's bar-distribution predictive is scored against the noise-free truth f(x) at held-out points, for the
BO context of a labeled state and for a random-design context with the same task and the same n. For calibration
claims, every conditioning variable must be a function of the context only: an adaptive design (chosen from the
observed data) is ignorable, so under the exact posterior the PIT is uniform and event probabilities are
calibrated for BO and random designs alike. Conditioning on f* (the gap) or on f(x) breaks that guarantee;
those strata are descriptive only.
"""

import math

import numpy as np
import torch
from torch import Tensor

from ..experimental.api import _reflect


def local_points(x_inc: Tensor, z: Tensor, sigmas: tuple[float, ...]) -> Tensor:
    """[L, K, d]: Gaussian perturbations of the incumbent x_inc [L, d], z [L, K, d] standard normal, sigmas cycled
    over the K points; reflected into [0, 1]^d."""
    K = z.shape[1]
    sig = torch.tensor(sigmas, device=z.device).repeat(math.ceil(K / len(sigmas)))[:K]
    return _reflect(x_inc[:, None] + sig[None, :, None] * z)


def score_points(bd, logits: Tensor, y: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    """Log density at y (the bar density of y's bin) and the randomized PIT: the CDF up to y's bin plus u times
    the bin's mass (within-bin randomization, so an exact predictive gives an exactly uniform PIT).
    logits [..., nb], y and u [...]."""
    idx = bd.map_to_bucket_idx(y)
    logp = torch.log_softmax(logits, -1)
    log_density = (logp - torch.log(bd.bucket_widths)).gather(-1, idx[..., None]).squeeze(-1)
    p = logp.exp()
    below = (p.cumsum(-1) - p).gather(-1, idx[..., None]).squeeze(-1)
    mass = p.gather(-1, idx[..., None]).squeeze(-1)
    return log_density, below + u * mass


def event_prob(bd, logits: Tensor, thr: Tensor) -> Tensor:
    """P(Y < thr) under the bar density; logits [L, P, nb], thr [L, P]."""
    return bd.pi(logits, thr)


def ece(p: np.ndarray, o: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins of the predicted probability."""
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    err = 0.0
    for part in np.array_split(order, n_bins):
        if len(part):
            err += len(part) * abs(p[part].mean() - o[part].mean())
    return err / len(p)


def ks_uniform(pit: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of the PIT values to U(0, 1)."""
    if len(pit) == 0:
        return float("nan")
    s = np.sort(pit)
    n = len(s)
    i = np.arange(1, n + 1)
    return float(max((i / n - s).max(), (s - (i - 1) / n).max()))

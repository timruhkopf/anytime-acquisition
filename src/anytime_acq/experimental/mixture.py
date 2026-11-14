"""An in-context mixture of acquisition functions (tasks/mixture_policy.py; notebook 17).

At every step one PFN pass scores the seed-s candidate set under every arm. Each arm's scores are normalized
within the state: (score - best) / (best - median) over the valid candidates, clamped at -4, so every arm's best
candidate is at 0 and its median at -1, whatever the arm's units. The policy picks the argmax of
sum_k w_k * normalized score_k (ties to the lower posterior mean, as the arms do).
- w = one arm: that arm's pick. Mixed weights can pick a candidate that no arm ranks first.
- w = softmax(theta @ phi): phi = [1, tau, tau^2, standardized context features], tau = t / T (budget used).
  A mask on phi's entries makes a policy budget-only (constant and tau terms) or context-dependent.

Context features (`FEATURES`) come from the same pass: readings of the PFN's belief, where the arms' picks sit,
and the improvement history. Nothing privileged.
"""

from collections.abc import Callable

import torch
from pfn_repro.acquisitions.standard import arm_scores, fstar_quantiles, parse_arm
from torch import Tensor

from ..bo.engine import Engine
from .api import _argmax, pfn_pass
from .options import _bd, seeded_candidates

N_TIME = 3  # phi[:3] = 1, tau, tau^2
FEATURES = (
    "log_max_ei",
    "max_pi",
    "sd_local",
    "sd_global",
    "log_expected_fstar_gap",
    "y_inc",
    "ei_pick_dist_inc",
    "ucb_pick_dist_inc",
    "ei_pick_is_local",
    "since_improve",
    "log_gap_best_second",
)


def _log(x: Tensor, floor: float) -> Tensor:
    return torch.log10(x.clamp(min=0) + floor)


@torch.no_grad()
def mixture_step(engine: Engine, arms: list[str], X: Tensor, Y: Tensor, s: int, T: int) -> dict[str, Tensor]:
    """One state per lane -> C [L, P, d], Z [L, P, K] normalized arm scores, valid [L, P], mu [L, P],
    feats [L, len(FEATURES)] (raw, not standardized)."""
    bd = _bd(engine)
    ns = engine.candidates.n_sobol
    L, n, _ = X.shape
    r = torch.arange(L, device=X.device)
    C = seeded_candidates(engine, X, Y, s)
    lg, _ = pfn_pass(engine, X, Y, C, with_z=False)
    inc = Y.min(1).values
    xi = X[r, Y.argmin(1)]
    mu = bd.mean(lg).float()
    sd = bd.variance(lg).clamp(min=0).sqrt().float()
    valid = torch.cdist(C, X).min(-1).values > 1e-7
    fs = fstar_quantiles(bd, lg, inc)
    S = torch.stack(
        [
            arm_scores(a, bd, lg, inc, engine.ucb_quantile, mes_fs=fs if parse_arm(a)[0] == "mes" else None).float()
            for a in arms
        ],
        -1,
    )
    v = valid[..., None]
    best = S.masked_fill(~v, float("-inf")).amax(1, keepdim=True)
    med = S.masked_fill(~v, float("nan")).nanmedian(1, keepdim=True).values
    Z = ((S - best) / (best - med).clamp(min=1e-30)).clamp(min=-4.0)
    ei = bd.ei(lg, inc[:, None]).float().masked_fill(~valid, 0.0)
    pi = bd.pi(lg, inc[:, None]).float().masked_fill(~valid, 0.0)
    del lg

    def pick(name: str) -> Tensor:
        k = arms.index(name) if name in arms else None
        sc = Z[..., k] if k is not None else ei
        return C[r, _argmax(sc, mu, valid)]

    ei_j = _argmax(ei, mu, valid)
    ucb = next((a for a in arms if parse_arm(a)[0] == "ucb"), None)
    ys = Y.sort(1).values
    f = {
        "log_max_ei": _log(ei.max(1).values, 1e-12),
        "max_pi": pi.max(1).values,
        "sd_local": sd[:, ns:].mean(1),
        "sd_global": sd[:, :ns].mean(1),
        "log_expected_fstar_gap": _log(inc - fs.mean(1), 1e-6),
        "y_inc": inc,
        "ei_pick_dist_inc": (C[r, ei_j] - xi).norm(dim=-1),
        "ucb_pick_dist_inc": (pick(ucb) - xi).norm(dim=-1) if ucb else torch.zeros_like(inc),
        "ei_pick_is_local": (ei_j >= ns).float(),
        "since_improve": (n - 1 - Y.argmin(1)).float() / T,
        "log_gap_best_second": _log(ys[:, 1] - ys[:, 0], 1e-6),
    }
    return {"C": C, "Z": Z, "valid": valid, "mu": mu, "feats": torch.stack([f[k] for k in FEATURES], 1)}


def phi(feats: Tensor, t: int, T: int, mean: Tensor, sd: Tensor) -> Tensor:
    """[L, 3 + F]: 1, tau, tau^2, standardized features (clamped at +-5)."""
    tau = torch.full((len(feats), 1), t / T, device=feats.device)
    z = ((feats - mean) / sd).clamp(-5, 5)
    return torch.cat([torch.ones_like(tau), tau, tau**2, z], 1)


def one_hot_theta(k: int, n_arms: int, bias: float = 50.0) -> Tensor:
    """theta [K, 3 + F] of the static arm k (all weight on it at every state)."""
    th = torch.zeros(n_arms, N_TIME + len(FEATURES))
    th[k, 0] = bias
    return th


Policy = Callable[[Tensor, Tensor, Tensor, Tensor, int], Tensor]


def mixture_policy(
    engine: Engine,
    arms: list[str],
    theta: Tensor,
    mask: Tensor,
    pol: Tensor,
    T: int,
    s: int,
    stats: tuple[Tensor, Tensor],
    record: list[Tensor] | None = None,
) -> Policy:
    """A `Policy` for `experimental/api.py::rollout`. theta [M, K, 3 + F], mask [M, 3 + F] (1 = the policy may use
    this entry of phi), pol [L]: lane l follows policy pol[l]. `record` collects the weights per step [L, K]."""
    mean, sd = stats

    def policy(X: Tensor, Y: Tensor, rows: Tensor, reps: Tensor, t: int) -> Tensor:
        out, ws = [], []
        for a in range(0, len(X), engine.chunk):
            sl = slice(a, a + engine.chunk)
            st = mixture_step(engine, arms, X[sl], Y[sl], s, T)
            p = phi(st["feats"], t, T, mean, sd) * mask[pol[sl]]
            w = torch.softmax(torch.einsum("lkf,lf->lk", theta[pol[sl]], p), -1)
            comb = (st["Z"] * w[:, None]).sum(-1)
            j = _argmax(comb, st["mu"], st["valid"])
            out.append(st["C"][torch.arange(len(j), device=X.device), j])
            ws.append(w)
        if record is not None:
            record.append(torch.cat(ws).cpu())
        return torch.cat(out)

    return policy

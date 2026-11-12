"""Options "follow arm a for k steps, then pi_0", deterministic arms, and privileged basin labels
(research_log/decisions/0008; tasks/option_headroom.py).

- Every arm maximizes its acquisition (`pfn_repro`'s `arm_scores`) over a candidate set that depends only on the
  data and a replicate seed s (`fixed_candidates`): the scrambled Sobol set of seed s and fixed perturbation
  directions of seed s around the incumbent. For a fixed s, every policy is a deterministic function of the data,
  so on noise-free tasks one continuation is an exact outcome. The seed s is the luck replicate: a different but
  equally uninformed candidate set.
- A null arm ``ei@j`` maximizes EI over a fixed random half of that candidate set (mask seeded by (s, j, step)):
  a deviation from EI of comparable kind that carries no information.
- Basins (privileged): local descent on the true f (projected Adam) from a point; the descended value says which
  basin's minimum the point drains to. "Good basin" = descended value within delta of f*.
"""

from dataclasses import dataclass
from typing import cast

import torch
from pfn_repro.acquisitions.standard import arm_scores, fstar_quantiles, parse_arm
from pfn_repro.objectives.bar_distribution import BarDistribution
from torch import Tensor

from ..bo.engine import Engine
from ..bo.tasks import Tasks
from .api import _argmax, pfn_pass

SOBOL_SEED_BASE = 23
PERTURB_SEED_BASE = 29


def _bd(engine: Engine) -> BarDistribution:
    assert engine.pfn is not None
    return cast(BarDistribution, engine.pfn.bar_dist)


def seeded_candidates(engine: Engine, X: Tensor, Y: Tensor, s: int) -> Tensor:
    """[L, P, d]: the engine's candidate sizes, a function of the data and the replicate seed s only."""
    L, _, d = X.shape
    dev = X.device
    c = engine.candidates
    sob = torch.quasirandom.SobolEngine(d, scramble=True, seed=SOBOL_SEED_BASE + 1000 * s).draw(c.n_sobol).to(dev)
    n_loc = c.n_local * len(c.local_sigmas)
    z = torch.randn(n_loc, d, generator=torch.Generator().manual_seed(PERTURB_SEED_BASE + 1000 * s)).to(dev)
    sig = torch.tensor(c.local_sigmas, device=dev).repeat_interleave(c.n_local)[:, None]
    inc = X[torch.arange(L, device=dev), Y.argmin(1)][:, None]
    return torch.cat([sob[None].expand(L, -1, -1), (inc + sig * z[None]).clamp(0, 1)], 1)


def _null_mask(P: int, s: int, j: int, step: int, device: torch.device) -> Tensor:
    g = torch.Generator().manual_seed(7_919 * (s + 1) + 104_729 * (j + 1) + step)
    return (torch.rand(P, generator=g) < 0.5).to(device)


@torch.no_grad()
def arm_pick(engine: Engine, arm: str, X: Tensor, Y: Tensor, s: int, step: int) -> Tensor:
    """x [L, d]: the argmax of `arm` (an `arm_scores` name, or ``ei@j`` for the null) over the seed-s candidates.
    Ties go to the lower posterior mean, as in the engine."""
    out = []
    bd = _bd(engine)
    name, _, j = arm.partition("@")
    for a in range(0, len(X), engine.chunk):
        Xa, Ya = X[a : a + engine.chunk], Y[a : a + engine.chunk]
        C = seeded_candidates(engine, Xa, Ya, s)
        lg, _ = pfn_pass(engine, Xa, Ya, C, with_z=False)
        inc = Ya.min(1).values
        fs = fstar_quantiles(bd, lg, inc) if parse_arm(name)[0] == "mes" else None
        sc = arm_scores(name, bd, lg, inc, engine.ucb_quantile, mes_fs=fs)
        valid = torch.cdist(C, Xa).min(-1).values > 1e-7
        if j:
            valid = valid & _null_mask(C.shape[1], s, int(j), step, X.device)[None]
        r = torch.arange(len(Xa), device=X.device)
        out.append(C[r, _argmax(sc, bd.mean(lg), valid)])
        del lg
    return torch.cat(out)


def option_policy(engine: Engine, arms: list[str], until: Tensor, s: int, base: str = "ei"):
    """A `Policy` for `experimental/api.py::rollout`: lane l follows arms[l] at steps < until[l], then `base`.
    One PFN pass per distinct arm and step."""

    def pol(X: Tensor, Y: Tensor, rows: Tensor, reps: Tensor, t: int) -> Tensor:
        cur = [arms[i] if t < int(until[i]) else base for i in range(len(arms))]
        x = torch.empty(X.shape[0], X.shape[-1], device=X.device)
        for a in dict.fromkeys(cur):
            li = torch.tensor([i for i, b in enumerate(cur) if b == a], device=X.device)
            x[li] = arm_pick(engine, a, X[li], Y[li], s, t)
        return x

    return pol


# --------------------------------------------------------------------------------------------------
# privileged: basins by local descent on the true f
# --------------------------------------------------------------------------------------------------


@dataclass
class DescentSpec:
    steps: int = 300
    lr: float = 0.003
    chunk: int = 65_536  # points per pass


def _pack(rows: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
    uniq, inv = rows.unique(return_inverse=True)
    counts = torch.bincount(inv, minlength=len(uniq))
    m = int(counts.max())
    order = inv.argsort(stable=True)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.empty_like(inv)
    slot[order] = torch.arange(len(inv), device=rows.device) - starts[inv[order]]
    return uniq, inv, slot, m


@torch.enable_grad()
def _descend_chunk(tasks: Tasks, x: Tensor, rows: Tensor, spec: DescentSpec) -> Tensor:
    uniq, inv, slot, m = _pack(rows)
    P = x.new_full((len(uniq), m, x.shape[-1]), 0.5)
    P[inv, slot] = x
    P = P.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([P], lr=spec.lr)
    best = tasks.f(P.detach(), uniq).detach()
    for _ in range(spec.steps):
        opt.zero_grad()
        f = tasks.f(P, uniq)
        f.sum().backward()
        opt.step()
        with torch.no_grad():
            P.clamp_(0, 1)
            best = torch.minimum(best, tasks.f(P, uniq))
    return best[inv, slot]


def descend(tasks: Tasks, x: Tensor, rows: Tensor, spec: DescentSpec) -> Tensor:
    """[L]: the lowest true f reached by projected Adam descent from x [L, d] on task rows[l] (the value of the
    basin's minimum, as far as local descent finds it). Duplicate points are descended once."""
    key = torch.cat([rows[:, None].to(x.dtype), x], 1)
    uq, back = key.unique(dim=0, return_inverse=True)
    out = torch.empty(len(uq), device=x.device)
    for a in range(0, len(uq), spec.chunk):
        u = uq[a : a + spec.chunk]
        out[a : a + spec.chunk] = _descend_chunk(tasks, u[:, 1:], u[:, 0].long(), spec)
    return out[back]


# --------------------------------------------------------------------------------------------------
# privileged: how well does a belief locate the true optimum?
# --------------------------------------------------------------------------------------------------


@torch.no_grad()
def belief_at_optimum(engine: Engine, X: Tensor, Y: Tensor, xstar: Tensor, thr: Tensor, s: int) -> dict[str, Tensor]:
    """Per lane: P(y(x*) < thr) under the PFN given (X, Y) (thr [L], a threshold common to compared lanes), and
    the distance from the belief's predicted-minimum location (argmin of the posterior mean over the seed-s
    candidates plus the context) to x*."""
    bd = _bd(engine)
    out_p, out_d = [], []
    for a in range(0, len(X), engine.chunk):
        Xa, Ya, xs = X[a : a + engine.chunk], Y[a : a + engine.chunk], xstar[a : a + engine.chunk]
        C = torch.cat([seeded_candidates(engine, Xa, Ya, s), Xa, xs[:, None]], 1)
        lg, _ = pfn_pass(engine, Xa, Ya, C, with_z=False)
        out_p.append(bd.pi(lg[:, -1:], thr[a : a + engine.chunk, None])[:, 0])
        mu = bd.mean(lg[:, :-1])
        r = torch.arange(len(Xa), device=X.device)
        out_d.append((C[r, mu.argmin(1)] - xs).norm(dim=-1))
        del lg
    return {"pi_xstar": torch.cat(out_p), "dist_argmin_xstar": torch.cat(out_d)}

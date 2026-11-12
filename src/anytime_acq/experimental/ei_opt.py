"""How well is EI optimized? A deterministic, gradient-refined EI as the base policy (issue #13, decision 0007).

pi_0 so far (`experimental/api.py::base_picks`) is the argmax of the PFN's EI over a fresh random candidate set per
(task, step, replicate): a Cranley-Patterson-shifted Sobol set plus Gaussian perturbations of the incumbent. It is
therefore stochastic, and on noise-free tasks that randomness is the only noise in the round-1 labels. Variants:

- ``sampled``          the current pi_0 (the engine's candidate set at (step, replicate)).
- ``fixed``            a candidate set that depends only on the data: the unshifted scrambled Sobol set plus fixed
                       perturbations of the incumbent (one fixed draw of directions, shared by all lanes and steps).
- ``<base>+refine``    the top `k` candidates of <base> by EI as starts, then Adam on x through the PFN
                       (`logits_from_context_kv` is differentiable in x), best iterate per start; the final pick is
                       the best of the refined starts and the base argmax, so refinement never lowers EI.

EI is the PFN bar distribution's closed form (`pfn_repro`'s `BarDistribution.ei`), against the incumbent.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from pfn_repro.objectives.bar_distribution import BarDistribution
from torch import Tensor

from ..bo.engine import Engine
from .api import _argmax, pfn_pass

VARIANTS = ("sampled", "sampled+refine", "fixed", "fixed+refine")
FIXED_SOBOL_SEED = 23
FIXED_PERTURB_SEED = 29


@dataclass
class RefineSpec:
    k: int = 8  # starts per lane
    steps: int = 100
    lr: float = 0.01  # Adam on x in [0, 1]^d, cosine-annealed to lr * 1e-2
    min_dist: float = 1e-6  # a pick must not coincide with an evaluated point (as the engine's validity rule)


def fixed_candidates(engine: Engine, X: Tensor, Y: Tensor) -> Tensor:
    """[L, P, d]: the engine's candidate *sizes*, but a function of the data only: the unshifted scrambled Sobol
    set and fixed perturbation directions around the incumbent (no step, replicate or task dependence)."""
    L, _, d = X.shape
    dev = X.device
    c = engine.candidates
    sob = torch.quasirandom.SobolEngine(d, scramble=True, seed=FIXED_SOBOL_SEED).draw(c.n_sobol).to(dev)
    n_loc = c.n_local * len(c.local_sigmas)
    z = torch.randn(n_loc, d, generator=torch.Generator().manual_seed(FIXED_PERTURB_SEED)).to(dev)
    sig = torch.tensor(c.local_sigmas, device=dev).repeat_interleave(c.n_local)[:, None]
    inc = X[torch.arange(L, device=dev), Y.argmin(1)][:, None]
    return torch.cat([sob[None].expand(L, -1, -1), (inc + sig * z[None]).clamp(0, 1)], 1)


def _bd(engine: Engine) -> BarDistribution:
    assert engine.pfn is not None
    return cast(BarDistribution, engine.pfn.bar_dist)


def _ei(engine: Engine, lg: Tensor, inc: Tensor) -> Tensor:
    return _bd(engine).ei(lg, inc[:, None])


@torch.enable_grad()
def refine(engine: Engine, X: Tensor, Y: Tensor, starts: Tensor, spec: RefineSpec) -> tuple[Tensor, Tensor]:
    """starts [L, K, d] -> (best iterate per start [L, K, d], its EI [L, K]). Maximizes log EI with Adam on x,
    clamped to [0, 1]; iterates within `min_dist` of an evaluated point do not count."""
    pfn = engine.pfn
    assert pfn is not None
    nf = torch.full((len(X),), X.shape[-1], device=X.device, dtype=torch.long)
    with torch.no_grad():
        kv = pfn.context_kv(X, Y, n_features=nf)
    inc = Y.min(1).values
    x = starts.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=spec.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(spec.steps, 1), eta_min=spec.lr * 1e-2)
    best_x = starts.clone()
    best_ei = torch.full(starts.shape[:2], -1.0, device=X.device)
    for it in range(spec.steps + 1):
        lg = pfn.logits_from_context_kv(kv, x, n_features=nf).float()
        ei = _ei(engine, lg, inc)
        with torch.no_grad():
            ok = torch.cdist(x, X).min(-1).values > spec.min_dist
            better = ok & (ei > best_ei)
            best_ei = torch.where(better, ei, best_ei)
            best_x = torch.where(better[..., None], x, best_x)
        if it == spec.steps:
            break
        opt.zero_grad()
        (-torch.log(ei.clamp_min(1e-30)).sum()).backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    return best_x.detach(), best_ei.detach()


def pick(
    engine: Engine,
    variant: str,
    X: Tensor,
    Y: Tensor,
    rows: Tensor,
    reps: Tensor,
    step: int,
    n_tasks: int,
    spec: RefineSpec,
) -> tuple[Tensor, Tensor]:
    """-> (x [L, d], its EI [L]) of `variant` (VARIANTS), in chunks of `engine.chunk` lanes."""
    base, _, ref = variant.partition("+")
    assert base in ("sampled", "fixed") and ref in ("", "refine"), variant
    xs, eis = [], []
    for s in range(0, len(X), engine.chunk):
        Xa, Ya = X[s : s + engine.chunk], Y[s : s + engine.chunk]
        with torch.no_grad():
            if base == "sampled":
                C = engine.candidate_set(Xa, Ya, rows[s : s + engine.chunk], reps[s : s + engine.chunk], step, n_tasks)
            else:
                C = fixed_candidates(engine, Xa, Ya)
            lg, _ = pfn_pass(engine, Xa, Ya, C, with_z=False)
            assert engine.pfn is not None
            inc = Ya.min(1).values
            ei = _ei(engine, lg, inc)
            valid = torch.cdist(C, Xa).min(-1).values > 1e-7
            j = _argmax(ei, _bd(engine).mean(lg), valid)
            r = torch.arange(len(Xa), device=X.device)
            x0, e0 = C[r, j], ei[r, j]
            del lg
        if ref:
            k = min(spec.k, C.shape[1])
            top = ei.masked_fill(~valid, -1.0).topk(k, dim=1).indices
            xr, er = refine(engine, Xa, Ya, C.gather(1, top[..., None].expand(-1, -1, C.shape[-1])), spec)
            jb = er.argmax(1)
            xb, eb = xr[r, jb], er[r, jb]
            use = eb > e0
            x0, e0 = torch.where(use[:, None], xb, x0), torch.where(use, eb, e0)
        xs.append(x0)
        eis.append(e0)
    return torch.cat(xs), torch.cat(eis)


def ei_policy(
    engine: Engine, variant: str, n_tasks: int, spec: RefineSpec
) -> Callable[[Tensor, Tensor, Tensor, Tensor, int], Tensor]:
    """A `Policy` for `experimental/api.py::rollout`."""
    return lambda X, Y, rows, reps, t: pick(engine, variant, X, Y, rows, reps, t, n_tasks, spec)[0]

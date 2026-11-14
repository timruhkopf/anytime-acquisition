"""The model-based gate (research_log/decisions/0003 §8.3): a one-step improvement over EI whose lookahead
uses only the PFN, never the true task.

Belief continuation: from a state (D, t), query candidate c, draw its y from the PFN's posterior predictive,
then follow pi_0 (EI, experimental/api.py) to T, drawing every y from the PFN's predictive given the
fantasized data so far (the posterior predictive process; on the noise-free prior y = f). The value of a
continuation is its best y (lower is better); f* doesn't enter, since it is the same for every candidate.
Continuations k = 0..M-1 of one state share their nuisance (candidate sets and fantasy uniforms, replicate
ids rep_base + k) across candidates: common random numbers.

Two uses:
- `belief_values` at the labeled states of a round-1 label set: the candidate with the best model-based
  advantage is selected *without* the task and scored with its ground-truth label. The mean of that score
  is a deployable, first-order estimate of the gain of one improvement step (Stage 1a);
- `run_model_based`: the rollout algorithm itself on fresh tasks (Stage 1b): at every step, label-style
  candidates are scored by belief continuations and the best one is queried if its model-based advantage
  is > 0, else x0.
"""

import torch
from torch import Tensor

from ..bo import nuisance
from ..bo.engine import Engine
from ..bo.runner import sample_bar
from ..bo.tasks import TaskSet
from .api import CandidateSpec, candidate_set, choose_labeled, f_lanes, features_ei, pfn_pass, pi0


@torch.no_grad()
def belief_values(
    engine: Engine,
    X: Tensor,
    Y: Tensor,
    rows: Tensor,
    xq: Tensor,
    t: int,
    T: int,
    M: int,
    rep_base: Tensor,
    n_tasks: int,
) -> Tensor:
    """X [G, n, d], Y [G, n] (the state's data), xq [G, J, d] candidates, rep_base [G]. -> V [G, J, M]: the best
    y of each fantasized continuation (query xq at step t, then pi_0 to T)."""
    assert engine.pfn is not None
    bd = engine.pfn.bar_dist
    G, J, d = xq.shape
    lg, _ = pfn_pass(engine, X, Y, xq, with_z=False)  # [G, J, nb]
    rows_l = rows.repeat_interleave(J * M)
    reps_l = rep_base.repeat_interleave(J * M) + torch.arange(M, device=X.device).repeat(G * J)
    u = nuisance.stream(engine.base_seed, "fantasy", t, reps_l, rows_l, (), n_tasks, False, X.device)
    yq = sample_bar(bd, lg.repeat_interleave(M, 1).reshape(G * J * M, -1), u)
    Xl = torch.cat([X.repeat_interleave(J * M, 0), xq.repeat_interleave(M, 1).reshape(G * J * M, 1, d)], 1)
    Yl = torch.cat([Y.repeat_interleave(J * M, 0), yq[:, None]], 1)
    policy = pi0(engine, n_tasks)
    for s in range(t + 1, T):
        x = policy(Xl, Yl, rows_l, reps_l, s)
        lgx, _ = pfn_pass(engine, Xl, Yl, x[:, None], with_z=False)
        u = nuisance.stream(engine.base_seed, "fantasy", s, reps_l, rows_l, (), n_tasks, False, X.device)
        Xl = torch.cat([Xl, x[:, None]], 1)
        Yl = torch.cat([Yl, sample_bar(bd, lgx[:, 0], u)[:, None]], 1)
    return Yl.min(1).values.view(G, J, M)


def model_advantage(V: Tensor) -> tuple[Tensor, Tensor]:
    """V [G, J, M] (index 0 = x0) -> mean paired advantage over x0 [G, J] (positive = better) and its se."""
    diff = V[:, :1] - V
    M = V.shape[-1]
    se = diff.std(-1, unbiased=True) / M**0.5 if M > 1 else torch.zeros_like(diff[..., 0])
    return diff.mean(-1), se


@torch.no_grad()
def run_model_based(
    engine: Engine,
    spec: CandidateSpec,
    tasks: TaskSet,
    X0: Tensor,
    T: int,
    n_labeled: int,
    M: int,
    u_seed: int,
    rep_offset: int = 100,
) -> dict[str, Tensor]:
    """The model-based rollout policy on every task (replicate 0), with pi_0 in parallel lanes (CRN). At step t:
    C(s) as in round 1; the n_labeled candidates chosen as for labels (x0, heuristic argmaxes, top EI, random);
    belief continuations with replicates rep_offset + t * M + k (fresh per decision, shared across candidates); query
    argmax of the model-based advantage if > 0, else x0.
    -> F [B, 2, n] (0 = EI, 1 = model-based), deviated [B, T], adv_model of the chosen candidate [B, T]."""
    B, n_init, d = X0.shape
    dev = X0.device
    rows = torch.arange(B, device=dev)
    reps0 = torch.zeros(B, dtype=torch.long, device=dev)
    F0 = f_lanes(tasks, X0.reshape(-1, d), rows.repeat_interleave(n_init)).view(B, n_init)
    Xe, Fe = X0.clone(), F0.clone()  # EI lanes
    Xm, Fm = X0.clone(), F0.clone()  # model-based lanes
    pol = pi0(engine, B)
    devs, advs = [], []
    for t in range(T):
        xe = pol(Xe, Fe, rows, reps0, t)
        C, valid = candidate_set(engine, spec, Xm, Fm, rows, reps0, t, B)
        ei = features_ei(engine, Xm, Fm, C)
        gen = torch.Generator().manual_seed(u_seed * 1_009 + t)
        u = torch.rand(B, spec.size, generator=gen).to(dev)
        lab = choose_labeled(valid, ei, n_labeled, u)
        lab_valid = valid.gather(1, lab)
        xq = C.gather(1, lab[..., None].expand(-1, -1, d))
        V = belief_values(engine, Xm, Fm, rows, xq, t, T, M, rep_offset + t * M + reps0, B)
        adv, _ = model_advantage(V)
        adv = adv.masked_fill(~lab_valid, float("-inf"))
        adv[:, 0] = 0.0
        best = adv.max(1)
        deviated = best.values > 0
        j = torch.where(deviated, best.indices, torch.zeros_like(best.indices))
        xm = xq[rows, j]
        Xe = torch.cat([Xe, xe[:, None]], 1)
        Fe = torch.cat([Fe, f_lanes(tasks, xe, rows)[:, None]], 1)
        Xm = torch.cat([Xm, xm[:, None]], 1)
        Fm = torch.cat([Fm, f_lanes(tasks, xm, rows)[:, None]], 1)
        devs.append(deviated)
        advs.append(adv[rows, j])
    return {"F": torch.stack([Fe, Fm], 1), "deviated": torch.stack(devs, 1), "adv_model": torch.stack(advs, 1)}

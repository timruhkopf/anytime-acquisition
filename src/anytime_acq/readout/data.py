"""States and targets for amortized EI.

A state is a context (X, Y) of n points on a BNN-prior task, plus the budget left. Half the states
come from EI trajectories run with the frozen PFN (the distribution a BO policy meets); the other half
are uniform random contexts. The target is the Boltzmann distribution over candidates,
p(x) ∝ exp((EI(x) − max EI) / (λ · max EI)), on 4096 shifted-Sobol candidates plus incumbent
perturbations, with EI from the frozen PFN. Training uses the `n_top` heaviest candidates and their
renormalized weights; `mass_top` records how much of the weight they hold.
"""

from dataclasses import dataclass

import torch
from pfn_repro.acquisitions.standard import arm_scores

from ..bo.engine import CandidateConfig, Engine
from ..bo.runner import Budget, run_static
from ..bo.tasks import draw_tasks, task_seed


@dataclass
class States:
    X: torch.Tensor  # [S, N, d] padded with zeros
    Y: torch.Tensor  # [S, N]
    mask: torch.Tensor  # [S, N] bool, True = real point
    budget_left: torch.Tensor  # [S] in [0, 1]
    task: torch.Tensor  # [S] index into seeds
    seeds: list[int]

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, idx: torch.Tensor) -> "States":
        return States(self.X[idx], self.Y[idx], self.mask[idx], self.budget_left[idx], self.task[idx], self.seeds)


@dataclass
class Targets:
    x: torch.Tensor  # [S, n_top, d]
    w: torch.Tensor  # [S, n_top], sums to 1
    max_ei: torch.Tensor  # [S]
    mass_top: torch.Tensor  # [S]


def make_states(engine: Engine, d: int, split: int, n_tasks: int, n_init: int, T: int, device: str) -> States:
    """n_tasks EI trajectories (every prefix after the initial design is a state) and as many random
    contexts with n uniform on [1, n_init + T]."""
    seeds = [task_seed(split, d, i) for i in range(n_tasks)]
    tasks = draw_tasks(d, seeds, n_dense=2048, refine_steps=20).to(device)  # f* is not needed here
    run = run_static(engine, tasks, "ei", Budget(n_init, T, T), device)
    N = n_init + T
    Fx, X = run["F"], run["X"]
    n_traj = torch.arange(n_init, N)  # context sizes of trajectory states: n_init .. N-1
    g = torch.Generator().manual_seed(split * 7 + d)
    n_rand = torch.randint(1, N + 1, (n_tasks * len(n_traj),), generator=g)
    Xs, Ys, Ms, Bs, Ts = [], [], [], [], []
    for n in n_traj.tolist():  # trajectory prefixes
        m = torch.arange(N, device=device) < n
        Xs.append(X * m[None, :, None])
        Ys.append(Fx * m[None])
        Ms.append(m.expand(n_tasks, -1))
        Bs.append(torch.full((n_tasks,), (N - n) / T, device=device))
        Ts.append(torch.arange(n_tasks, device=device))
    task_r = torch.arange(len(n_rand)) % n_tasks
    Xr = torch.rand(len(n_rand), N, d, generator=g).to(device)
    Yr = tasks.f(Xr, task_r.to(device))
    mr = (torch.arange(N)[None] < n_rand[:, None]).to(device)
    Xs.append(Xr * mr[..., None])
    Ys.append(Yr * mr)
    Ms.append(mr)
    Bs.append(((N - n_rand).clamp(min=0) / T).clamp(max=1).to(device))
    Ts.append(task_r.to(device))
    return States(torch.cat(Xs), torch.cat(Ys), torch.cat(Ms), torch.cat(Bs).float(), torch.cat(Ts), seeds)


@torch.no_grad()
def ei_targets(
    pfn: torch.nn.Module,
    states: States,
    lam: float,
    n_top: int,
    candidates: CandidateConfig,
    seed: int,
    chunk: int = 64,  # 64 states x 4288 candidates x 1000 bins of logits: ~1.1 GB
) -> Targets:
    S, _, d = states.X.shape
    dev = states.X.device
    sob = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed).draw(candidates.n_sobol).to(dev)
    sig = torch.tensor(candidates.local_sigmas, device=dev).repeat_interleave(candidates.n_local)[None, :, None]
    g = torch.Generator().manual_seed(seed + 1)
    xs, ws, mx, mass = [], [], [], []
    for s in range(0, S, chunk):
        sl = slice(s, min(s + chunk, S))
        X, Y, m = states.X[sl], states.Y[sl], states.mask[sl]
        B = len(X)
        shift = torch.rand(B, 1, d, generator=g).to(dev)
        Ym = Y.masked_fill(~m, float("inf"))
        inc_y, j = Ym.min(1)
        inc_x = X[torch.arange(B, device=dev), j]
        z = torch.randn(B, sig.shape[1], d, generator=g).to(dev)
        C = torch.cat([(sob[None] + shift) % 1.0, (inc_x[:, None] + sig * z).clamp(0, 1)], 1)
        nf = torch.full((B,), d, device=dev, dtype=torch.long)
        kv = pfn.context_kv(X, Y, n_features=nf, train_key_padding_mask=m)
        lg = pfn.logits_from_context_kv(kv, C, n_features=nf, train_key_padding_mask=m).float()
        ei = arm_scores("ei", pfn.bar_dist, lg, inc_y, 0.95)  # [B, C]
        top = ei.max(1, keepdim=True).values.clamp(min=1e-12)
        logit = (ei - top) / (lam * top)
        w_all = logit.softmax(1)
        wt, it = w_all.topk(n_top, 1)
        xs.append(torch.gather(C, 1, it[..., None].expand(-1, -1, d)))
        ws.append(wt / wt.sum(1, keepdim=True))
        mx.append(top[:, 0])
        mass.append(wt.sum(1))
    return Targets(torch.cat(xs), torch.cat(ws), torch.cat(mx), torch.cat(mass))


@torch.no_grad()
def ei_at(pfn: torch.nn.Module, states: States, x: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """EI of the frozen PFN at one point per state: x [S, d] -> [S]."""
    out = []
    for s in range(0, len(states), chunk):
        sl = slice(s, min(s + chunk, len(states)))
        X, Y, m = states.X[sl], states.Y[sl], states.mask[sl]
        nf = torch.full((len(X),), X.shape[-1], device=X.device, dtype=torch.long)
        kv = pfn.context_kv(X, Y, n_features=nf, train_key_padding_mask=m)
        lg = pfn.logits_from_context_kv(kv, x[sl, None], n_features=nf, train_key_padding_mask=m).float()
        inc = Y.masked_fill(~m, float("inf")).min(1).values
        out.append(arm_scores("ei", pfn.bar_dist, lg, inc, 0.95)[:, 0])
    return torch.cat(out)

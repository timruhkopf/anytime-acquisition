"""Benchmark tasks: functions drawn from the PFN's own BNN prior, one per integer seed.

A task is identified by (d, seed) alone: its function is a batch-size-1 `BNNPrior` draw, made on the CPU
(device-independent; a device-side RNG would make the same seed a different function on a GPU), then
stacked. f* is shared by every method run on a task: a dense scrambled-Sobol search plus Adam refinement
through the noise-free prior, lowered later if any evaluated point is lower (`Tasks.lower_fstar`).
Minimization; y in [0, 1] on the training prior's ECDF scale.
"""

from dataclasses import dataclass

import torch
from pfn_repro.checkpoints import make_prior
from pfn_repro.priors.bnn import BNNPrior

# BNNPrior attributes that carry per-function state (dim 0 = batch); see pfn-repro priors/bnn.py.
PER_FUNCTION = (
    "depth",
    "width",
    "crit",
    "preactivation_noise_std",
    "output_noise_std",
    "width_mask",
    "layer_mask",
    "active_dim",
    "active_dim_mask",
    "relevant_mask",
    "W_in",
    "b_in",
    "W_h",
    "b_h",
    "W_out",
    "b_out",
    "ecdf_sorted",
)


def task_seed(split: int, d: int, i: int) -> int:
    """Disjoint seed ranges per split (0: tuning, 1: evaluation, 2: pilot, 3+: readout data) and d."""
    return split * 10_000_019 + d * 100_003 + i


def _select(prior: BNNPrior, rows: torch.Tensor) -> BNNPrior:
    """A BNNPrior whose slot b is `prior`'s function rows[b] (shares nothing mutable)."""
    out = BNNPrior.__new__(BNNPrior)
    out.__dict__.update(prior.__dict__)
    for name in PER_FUNCTION:
        setattr(out, name, getattr(prior, name).index_select(0, rows.to(getattr(prior, name).device)))
    out.B = len(rows)
    return out


@dataclass
class Tasks:
    d: int
    seeds: list[int]
    prior: BNNPrior  # batched: slot b = task b
    fstar: torch.Tensor  # [B]
    xstar: torch.Tensor  # [B, d]

    @property
    def B(self) -> int:
        return len(self.seeds)

    def f(self, x: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Noise-free f. x: [B, N, d] for all tasks, or [L, N, d] for lanes with task rows[l]."""
        prior = self.prior if rows is None else _select(self.prior, rows)
        return prior.evaluate(x, noise=False)

    def to(self, device: str) -> "Tasks":
        p = self.prior
        for name in PER_FUNCTION:
            setattr(p, name, getattr(p, name).to(device))
        p.device = device
        self.fstar, self.xstar = self.fstar.to(device), self.xstar.to(device)
        return self

    def lower_fstar(self, f_seen: torch.Tensor) -> None:
        """f_seen [B, ...]: any evaluated values; f* := min(f*, min f_seen) (shared f*, standard S6)."""
        self.fstar = torch.minimum(self.fstar, f_seen.reshape(self.B, -1).min(1).values.to(self.fstar.device))


def _stack(priors: list[BNNPrior]) -> BNNPrior:
    out = priors[0]
    for name in PER_FUNCTION:
        setattr(out, name, torch.cat([getattr(p, name) for p in priors], 0))
    out.B = len(priors)
    return out


def draw_tasks(
    d: int, seeds: list[int], n_dense: int = 16384, n_refine: int = 16, refine_steps: int = 200, lr: float = 0.01
) -> Tasks:
    """Draw on the CPU; f* from a dense Sobol search + Adam refinement of the best `n_refine` points."""
    prior = _stack([make_prior(x_dim=d, batch_size=1, seed=int(s)) for s in seeds])
    B = len(seeds)
    dense = torch.quasirandom.SobolEngine(d, scramble=True, seed=7).draw(n_dense)[None].expand(B, -1, -1)
    with torch.no_grad():
        f = torch.cat([prior.evaluate(dense[:, i : i + 4096], noise=False) for i in range(0, n_dense, 4096)], 1)
    top = f.topk(n_refine, dim=1, largest=False).indices
    x = torch.gather(dense, 1, top[..., None].expand(-1, -1, d)).clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(refine_steps):
        opt.zero_grad()
        prior.evaluate(x.clamp(0, 1), noise=False).sum().backward()
        opt.step()
    with torch.no_grad():
        x = x.clamp(0, 1)
        fx = prior.evaluate(x, noise=False)
        cand_f = torch.cat([fx, f.gather(1, top)], 1)
        cand_x = torch.cat([x, torch.gather(dense, 1, top[..., None].expand(-1, -1, d))], 1)
        j = cand_f.argmin(1)
        r = torch.arange(B)
        return Tasks(d, list(seeds), prior, cand_f[r, j], cand_x[r, j])

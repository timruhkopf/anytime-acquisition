"""BBOB benchmark tasks (COCO's noise-free suite, via `ioh`) behind the `bo.tasks.Tasks` interface.

A task is (function id, instance, repetition). Repetitions of one instance are the same function with a
different initial design: the design's shift comes from the task seed (`bo/runner.py::initial_design`),
and the seed is `task_seed(split, d, i)` for the i-th task of the split's list, as for prior tasks.

f is returned as f(x) - f_opt (computed in float64 by ioh, then cast), so f* = 0 exactly and regrets keep
their resolution near the optimum in float32. Every surrogate here sees y only through a rank transform,
which is invariant to this shift, so knowing f_opt gives the optimizer nothing.
x in [0, 1]^d is mapped affinely onto BBOB's domain [-5, 5]^d.
"""

from dataclasses import dataclass
from itertools import product

import ioh
import numpy as np
import torch

LO, HI = -5.0, 5.0


def task_list(fids: list[int], instances: list[int], reps: int) -> list[tuple[int, int, int]]:
    """(fid, instance, rep) in a fixed order: function-major, then instance, then repetition."""
    return [(f, i, r) for f, i, r in product(fids, instances, range(reps))]


@dataclass
class BBOBTasks:
    d: int
    seeds: list[int]
    ids: list[tuple[int, int, int]]  # (fid, instance, rep) per row
    fstar: torch.Tensor  # [B] zeros: f is shifted by f_opt

    def __post_init__(self) -> None:
        self._problems = [
            ioh.get_problem(fid, instance=inst, dimension=self.d, problem_class=ioh.ProblemClass.BBOB)
            for fid, inst, _ in self.ids
        ]
        self._fopt = [p.optimum.y for p in self._problems]

    @property
    def B(self) -> int:
        return len(self.seeds)

    def f(self, x: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, N, d] for all tasks, or [L, N, d] for lanes with task rows[l]. -> f - f_opt, [B|L, N]."""
        L, N, _ = x.shape
        rows_np = np.arange(L) if rows is None else rows.cpu().numpy()
        z = (LO + (HI - LO) * x.detach().double().cpu().numpy()).reshape(L * N, self.d)
        out = np.empty(L * N)
        lane_of = np.repeat(rows_np, N)
        for row in np.unique(rows_np):
            m = lane_of == row
            out[m] = np.asarray(self._problems[row](z[m].tolist())) - self._fopt[row]
        return torch.from_numpy(out.reshape(L, N)).to(dtype=torch.float32, device=x.device).clamp(min=0)

    def to(self, device: str) -> "BBOBTasks":
        self.fstar = self.fstar.to(device)
        return self

    def lower_fstar(self, f_seen: torch.Tensor) -> None:
        """f* = 0 is exact for BBOB; kept for the interface."""


def bbob_tasks(d: int, seeds: list[int], ids: list[tuple[int, int, int]]) -> BBOBTasks:
    if len(seeds) != len(ids):
        raise ValueError(f"{len(seeds)} seeds for {len(ids)} BBOB tasks")
    return BBOBTasks(d, list(seeds), list(ids), torch.zeros(len(ids)))

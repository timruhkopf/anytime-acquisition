"""BO runs: static arms, and the non-anticipating rollout controller.

The rollout controller (Bertsekas-style policy improvement over a base policy) decides at the start of
every segment:
1. copy the current state of each task into K × R lanes (K arms, R rollouts);
2. each lane follows its arm for one segment, then the base arm until the horizon. Lanes of one rollout
   index r share fresh nuisance across arms (CRN), independent of the evaluated trajectory's
   (replicate 0) and of every other decision's;
3. execute the arm with the best mean rollout value for the segment, on the evaluated trajectory.

Two value models:
- `privileged`: rollouts observe the true f. The value is the final regret. This bounds what a
  controller with task knowledge and a one-segment lookahead can gain.
- `belief`: rollouts observe y drawn sequentially from the PFN's posterior predictive, i.e. functions
  sampled from the belief (for the PFN's Bayes posterior this is the posterior predictive process). The
  value is the best fantasized y; f* doesn't enter, since it is the same for every arm. This needs no
  privileged information: it is what a policy amortized over the prior could learn to do.
Because the choice never sees the evaluated trajectory's future, the gain over the base arm on the same
tasks has no selection bias (unlike a hindsight oracle). An equal-arms portfolio (`base@0, base@1, …`)
must give a gain of zero in expectation: the null control.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .engine import Engine
from .tasks import Tasks


@dataclass
class Budget:
    n_init: int  # shared scrambled-Sobol initial design
    T: int  # BO steps after the initial design
    segment: int  # steps per controller decision

    @property
    def n_decisions(self) -> int:
        return -(-self.T // self.segment)


def initial_design(tasks: Tasks, n_init: int) -> torch.Tensor:
    """[B, n_init, d]: the first n_init points of a scrambled Sobol sequence, shifted per task (CRN for
    every method on a task)."""
    d = tasks.d
    base = torch.quasirandom.SobolEngine(d, scramble=True, seed=3).draw(n_init)
    shifts = torch.stack(
        [torch.rand(d, generator=torch.Generator().manual_seed(int(s) % (2**31))) for s in tasks.seeds]
    )
    return (base[None] + shifts[:, None]) % 1.0


def sample_bar(bd: torch.nn.Module, logits: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF sample from bar-distribution logits [L, nb] with uniforms u [L]."""
    p = torch.softmax(logits.float(), -1)
    cdf = p.cumsum(-1).clamp(max=1.0)
    j = torch.searchsorted(cdf, u[:, None].clamp(max=1 - 1e-7)).squeeze(-1).clamp(max=p.shape[-1] - 1)
    prev = torch.where(j > 0, cdf.gather(1, (j - 1).clamp(min=0)[:, None]).squeeze(-1), torch.zeros_like(u))
    frac = ((u - prev) / p.gather(1, j[:, None]).squeeze(-1).clamp(min=1e-12)).clamp(0, 1)
    lo = bd.borders[j]
    return lo + frac * bd.bucket_widths[j]


def simulate(
    engine: Engine,
    tasks: Tasks,
    X: torch.Tensor,
    Y: torch.Tensor,
    F: torch.Tensor,
    rows: torch.Tensor,
    reps: torch.Tensor,
    arm_at: Callable[[int], list[str]],
    t0: int,
    t1: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run lanes from step t0 to t1 (exclusive). X [L,n,d], Y [L,n] (what the PFN sees), F [L,n] (true f; for
    belief lanes after t0 these are fantasies). arm_at(t) -> each lane's arm. -> (X, Y, F) extended."""
    for t in range(t0, t1):
        out = engine.propose(X, Y, arm_at(t), rows, reps, t, tasks.B, return_logits=(mode == "belief"))
        x = out["x"]
        if mode == "privileged":
            y = tasks.f(x[:, None], rows)[:, 0]
            f = y
        elif mode == "belief":
            from . import nuisance

            u = nuisance.stream(engine.base_seed, "fantasy", t, reps, rows, (), tasks.B, False, X.device)
            y = sample_bar(engine.pfn.bar_dist, out["logits"], u)
            f = y
        else:
            raise ValueError(mode)
        X = torch.cat([X, x[:, None]], 1)
        Y = torch.cat([Y, y[:, None]], 1)
        F = torch.cat([F, f[:, None]], 1)
    return X, Y, F


def run_static(
    engine: Engine, tasks: Tasks, arm: str, budget: Budget, device: str | torch.device
) -> dict[str, torch.Tensor]:
    """Every task with one arm, replicate 0. -> {"X": [B, n, d], "F": [B, n]} (n = n_init + T)."""
    X = initial_design(tasks, budget.n_init).to(device)
    F = tasks.f(X)
    rows = torch.arange(tasks.B, device=device)
    reps = torch.zeros(tasks.B, dtype=torch.long, device=device)
    X, _, F = simulate(engine, tasks, X, F.clone(), F, rows, reps, lambda t: [arm] * tasks.B, 0, budget.T, "privileged")
    return {"X": X, "F": F}


def run_controller(
    engine: Engine,
    tasks: Tasks,
    arms: list[str],
    base: str,
    budget: Budget,
    n_rollouts: int,
    mode: str,
    device: str | torch.device,
    rollout_engine: Engine | None = None,
) -> dict[str, torch.Tensor]:
    """The rollout controller on every task (evaluated trajectory = replicate 0). `rollout_engine` (e.g.
    fewer candidates) runs the rollouts only: a cheaper simulator changes the controller's choices, never
    how the evaluated trajectory is optimized.
    -> {"X", "F": as run_static, "choice": [B, n_decisions] arm index, "values": [B, n_decisions, K]}."""
    B, K, R = tasks.B, len(arms), n_rollouts
    X = initial_design(tasks, budget.n_init).to(device)
    F = tasks.f(X)
    rows = torch.arange(B, device=device)
    reps0 = torch.zeros(B, dtype=torch.long, device=device)
    choices, values = [], []
    for j in range(budget.n_decisions):
        t0 = j * budget.segment
        t_seg = min(t0 + budget.segment, budget.T)
        # lanes: task b, arm k, rollout r -> index (b * K + k) * R + r
        Lr = B * K * R
        rows_r = rows.repeat_interleave(K * R)
        arm_of_lane = [arms[k] for _ in range(B) for k in range(K) for _ in range(R)]
        rep_r = 1 + j * R + torch.arange(R, device=device).repeat(B * K)  # fresh, shared across arms
        Xr = X.repeat_interleave(K * R, 0)
        Fr = F.repeat_interleave(K * R, 0)

        def arm_at(t: int, t_seg: int = t_seg, lanes: list[str] = arm_of_lane, n: int = Lr) -> list[str]:
            return lanes if t < t_seg else [base] * n

        sim = rollout_engine or engine
        _, Yr, Fr2 = simulate(sim, tasks, Xr, Fr.clone(), Fr, rows_r, rep_r, arm_at, t0, budget.T, mode)
        v = (Fr2 if mode == "privileged" else Yr).min(1).values.reshape(B, K, R).mean(-1)  # [B, K]
        c = v.argmin(1)  # ties -> the earlier arm (put the base first)
        choices.append(c)
        values.append(v)
        chosen = [arms[int(k)] for k in c]
        X, _, F = simulate(
            engine, tasks, X, F.clone(), F, rows, reps0, lambda t, chosen=chosen: chosen, t0, t_seg, "privileged"
        )
    return {"X": X, "F": F, "choice": torch.stack(choices, 1), "values": torch.stack(values, 1)}


def regret_curve(F: torch.Tensor, fstar: torch.Tensor, n_init: int) -> torch.Tensor:
    """[B, T + 1]: simple regret of the best evaluated point after the initial design and after each step."""
    best = F.cummin(1).values[:, n_init - 1 :]
    return (best - fstar[:, None]).clamp(min=0)

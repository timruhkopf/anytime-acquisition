"""BO runs on BNN-prior tasks with the frozen PFN: static arms, random search, rollout controllers.

One member = one d and one method:
- `static`: every arm in `arms` (plus random search, and `base@0` if `null_reference`) on the same
  tasks, replicate 0;
- `privileged` / `belief`: the rollout controller over `arms` with base arm `base` (bo/runner.py);
- `equal_arms` (the null control): the privileged controller over K copies of the base arm, each
  restricted to its own random half of the candidates (`base@k`). Its gain over the static `base@0`
  must be 0 in expectation.

Tables: `best` (task, seed, arm, t, best_f: best evaluated f after the initial design and each step),
`fstar` (the member's f* estimate per task; analysis lowers it across members), and for controllers
`decisions` (task, decision, chosen arm, mean rollout value per arm).

Off the prior (`task.benchmark=bbob`, experimental): BBOB tasks with the PFN on rank-transformed y
(`task.surrogate=pfn`) or a batched GP (`task.surrogate=gp`); the belief controller is not defined there
(its fantasies live on the transformed scale), and BBOB tasks also log a `tasks` table (fid, instance, rep).
"""

import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pfn_repro.checkpoints import load

from ..bo.engine import CandidateConfig, Engine
from ..bo.runner import Budget, initial_design, run_controller, run_static
from ..bo.tasks import TaskSet, draw_tasks, task_seed
from ..evaluation import metric_names as M
from ..experimental.bbob import bbob_tasks, task_list
from ..experimental.surrogates import GPEngine, RankPFNEngine
from ..tracking.sink import Run
from ..utils.seeding import seed_everything

METHODS = ("static", "privileged", "belief", "equal_arms")
RANDOM = "random"


def per_d(t: DictConfig, key: str) -> object:
    """task.<key>, unless task.<key>_by_d gives a value for this d."""
    by_d = t.get(f"{key}_by_d")
    return by_d[t.d] if by_d is not None and t.d in by_d else t[key]


def _candidates(c: DictConfig) -> CandidateConfig:
    d = OmegaConf.to_container(c)
    return CandidateConfig(d["n_sobol"], d["n_local"], tuple(d["local_sigmas"]))


def budget_for(t: DictConfig) -> Budget:
    return Budget(n_init=t.n_init_per_d * t.d, T=t.T_per_d * t.d, segment=t.segment_per_d * t.d)


def random_search(tasks: TaskSet, budget: Budget, device: str) -> torch.Tensor:
    """F [B, n_init + T]: the shared initial design, then uniform points (seeded by the task seed)."""
    X0 = initial_design(tasks, budget.n_init)
    U = torch.stack(
        [
            torch.rand(budget.T, tasks.d, generator=torch.Generator().manual_seed(int(s) % (2**31) + 1))
            for s in tasks.seeds
        ]
    )
    return tasks.f(torch.cat([X0, U], 1).to(device))


def best_table(tasks: TaskSet, arm: str, F: torch.Tensor, n_init: int) -> pd.DataFrame:
    best = F.cummin(1).values[:, n_init - 1 :].cpu().numpy()
    B, n_t = best.shape
    return pd.DataFrame(
        {
            "task": np.repeat(np.arange(B), n_t),
            "seed": np.repeat(np.asarray(tasks.seeds), n_t),
            "arm": arm,
            "t": np.tile(np.arange(n_t), B),
            "best_f": best.ravel(),
        }
    )


def log_regret_final(F: torch.Tensor, fstar: torch.Tensor, eps: float) -> float:
    return float(torch.log10((F.min(1).values - fstar).clamp(min=0) + eps).mean())


def bo(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    if t.method not in METHODS:
        raise ValueError(f"task.method must be one of {METHODS}, got {t.method}")
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(t.tf32)
    torch.backends.cudnn.allow_tf32 = bool(t.tf32)
    dev = t.device
    d, budget = int(t.d), budget_for(t)
    arms = list(per_d(t, "arms"))
    base = str(per_d(t, "base"))

    t0 = time.time()
    if t.benchmark == "prior":
        seeds = [task_seed(t.split, d, t.task_offset + i) for i in range(t.n_tasks)]
        tasks = draw_tasks(d, seeds, **OmegaConf.to_container(t.task_draw)).to(dev)
    elif t.benchmark == "bbob":
        if t.method == "belief":
            raise ValueError("the belief controller is not defined off the prior (fantasies on the rank scale)")
        ids = task_list(list(t.bbob.fids), list(t.bbob.instances_by_split[t.split]), t.bbob.reps)
        seeds = [task_seed(t.split, d, t.task_offset + i) for i in range(len(ids))]
        tasks = bbob_tasks(d, seeds, ids).to(dev)
        fid, inst, rep = zip(*ids, strict=True)
        ids_table = {"task": np.arange(len(ids)), "seed": seeds, "fid": fid, "instance": inst, "rep": rep}
        run.log_table("tasks", pd.DataFrame(ids_table))
    else:
        raise ValueError(f"task.benchmark must be prior or bbob, got {t.benchmark}")
    run.log_summary(**{M.TIME_TASKS_S: time.time() - t0})

    needs_alpha = any(a.split("@")[0] == "es" for a in [*arms, base or ""])
    alpha = load(t.alpha)[0].to(dev).eval() if needs_alpha else None
    if t.surrogate == "gp":
        cls, pfn, extra = GPEngine, None, {"fit_iters": t.gp_fit_iters}
    elif t.surrogate == "pfn":
        cls, pfn, extra = (Engine if t.benchmark == "prior" else RankPFNEngine), load(t.pfn)[0].to(dev).eval(), {}
    else:
        raise ValueError(f"task.surrogate must be pfn or gp, got {t.surrogate}")
    engine = cls(pfn, alpha, _candidates(t.candidates), base_seed=cfg.seed, chunk=t.chunk, **extra)
    rollout_engine = (
        None
        if t.rollout_candidates is None
        else cls(pfn, alpha, _candidates(t.rollout_candidates), base_seed=cfg.seed, chunk=t.chunk, **extra)
    )

    tables, Fs = [], {}
    if t.method == "static":
        if t.null_reference:  # the static counterpart of the equal_arms controller
            arms = [*arms, f"{base}@0"]
        for arm in arms:
            t0 = time.time()
            Fs[arm] = run_static(engine, tasks, arm, budget, dev)["F"]
            run.log_summary(**{f"{M.TIME_PREFIX}{arm}_s": time.time() - t0})
        if t.random_search:
            Fs[RANDOM] = random_search(tasks, budget, dev)
    else:
        if t.method == "equal_arms":
            arms = [f"{base}@{k}" for k in range(t.null_k)]
            base_arm, mode = f"{base}@0", "privileged"
        else:
            if arms[0] != base:
                raise ValueError(f"put the base arm first (ties go to it): arms={arms}, base={base}")
            base_arm, mode = base, t.method
        t0 = time.time()
        out = run_controller(engine, tasks, arms, base_arm, budget, t.n_rollouts, mode, dev, rollout_engine)
        run.log_summary(**{f"{M.TIME_PREFIX}controller_s": time.time() - t0})
        label = f"ctrl:{t.method}"
        Fs[label] = out["F"]
        choice, values = out["choice"].cpu().numpy(), out["values"].cpu().numpy()  # [B, J], [B, J, K]
        B, J = choice.shape
        dec = pd.DataFrame(
            {
                "task": np.repeat(np.arange(B), J),
                "seed": np.repeat(np.asarray(seeds), J),
                "decision": np.tile(np.arange(J), B),
                "chosen": np.asarray(arms)[choice.ravel()],
            }
        )
        for k, a in enumerate(arms):
            dec[f"value/{a}"] = values[..., k].ravel()
        run.log_table("decisions", dec)

    for F in Fs.values():
        tasks.lower_fstar(F)
    for arm, F in Fs.items():
        tables.append(best_table(tasks, arm, F, budget.n_init))
        run.log_summary(**{f"{M.FINAL_LOG_REGRET}/{arm}": log_regret_final(F, tasks.fstar, t.eps)})
    run.log_table("best", pd.concat(tables, ignore_index=True))
    run.log_table(
        "fstar", pd.DataFrame({"task": np.arange(tasks.B), "seed": seeds, "fstar": tasks.fstar.cpu().numpy()})
    )
    return log_regret_final(next(iter(Fs.values())), tasks.fstar, t.eps)

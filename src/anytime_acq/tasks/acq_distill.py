"""Distill the arms' acquisition rankings into the acquisition branch on the frozen PFN
(experimental/acq_branch.py, analysis/acq_distill.py; pre-registration: notebooks/16_2026-11-13_acq_distill.ipynb;
idea: research_log/decisions/0009).

One member = one input dimension d.
- Training states: prefixes of BO runs on prior tasks, one run per arm (deterministic arms, candidate seed
  `train.cand_seed`) and `train.n_random` runs of uniformly random points per task.
- Teacher (`acq_branch.teacher`): each arm's acquisition value on the state's seed-0 candidate set plus fresh
  queries. Loss: `acq_branch.rank_loss`, one output column per arm.
- Evaluation on held-out tasks (runs of the arms under seed 0, and random runs), at every step:
  agreement of each column with its arm on the deployment candidates (`cand0`), on a candidate set with unseen
  Sobol points (`cand1`) and on a dense set (`dense`: a grid for d <= 2, Sobol points otherwise);
  example states with the scores on the dense set (d <= 2); and closed loop: BO with the branch's column as the
  acquisition against the arm itself, on the same tasks and candidates.

Tables: `train` (loss curve), `agree` (agreement per kind of run, step, query set and arm, averaged over
states), `examples`, `examples_ctx`,
`closed` (per task and arm), `curves` (mean log10 regret per step), `summary`. Artifact: `branch.pt`.
"""

import math
import time
from typing import cast

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load

from ..analysis.acq_distill import GOOD_PICK, distill_verdict, summarize
from ..bo.engine import Engine
from ..bo.runner import initial_design
from ..bo.tasks import Tasks, draw_tasks, task_seed
from ..experimental.acq_branch import (
    AcqBranch,
    BranchSpec,
    agreement,
    branch_policy,
    branch_scores,
    budget_features,
    extra_queries,
    rank_loss,
    teacher,
)
from ..experimental.api import f_lanes, rollout
from ..experimental.options import option_policy as arm_policy
from ..experimental.options import seeded_candidates
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .api import _dict
from .bo import _candidates


def shape(t: DictConfig) -> tuple[int, int, int]:
    d = int(t.d)
    n_init, T = (int(v) for v in t.shape[f"d{d}"])
    return d, n_init, T


def task_chunks(t: DictConfig, spec: DictConfig, d: int, n_init: int):
    """Yields (tasks, X0 [B, n_init, d], F0 [B, n_init], first task index) per chunk of prior tasks."""
    N = int(spec.n_tasks)
    seeds = [task_seed(int(spec.split), d, int(spec.task_offset) + i) for i in range(N)]
    for c0 in range(0, N, int(t.task_chunk)):
        with torch.enable_grad():  # draw_tasks refines f* by gradient descent
            tasks = draw_tasks(d, seeds[c0 : c0 + int(t.task_chunk)], **_dict(t.task_draw)).to(t.device)
        rows = torch.arange(tasks.B, device=t.device)
        X0 = initial_design(tasks, n_init).to(t.device)
        F0 = f_lanes(tasks, X0.reshape(-1, d), rows.repeat_interleave(n_init)).view(tasks.B, n_init)
        yield tasks, X0, F0, c0


def _random_policy(gen: torch.Generator):
    def pol(X: torch.Tensor, Y: torch.Tensor, rows: torch.Tensor, reps: torch.Tensor, t: int) -> torch.Tensor:
        return torch.rand(X.shape[0], X.shape[-1], generator=gen).to(X.device)

    return pol


@torch.no_grad()
def runs(
    engine: Engine,
    tasks: Tasks,
    X0: torch.Tensor,
    F0: torch.Tensor,
    arms: list[str],
    T: int,
    s: int,
    n_random: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full runs: per task one lane per arm under candidate seed s, then n_random lanes of uniform points.
    -> X [L, n_init + T, d], F [L, n_init + T], kind [L] (arm index; len(arms) = random), row [L]."""
    B, A = tasks.B, len(arms)
    rows = torch.arange(B, device=X0.device)
    gi = rows.repeat(A)
    lane_arms = [a for a in arms for _ in range(B)]
    X, F = rollout(
        tasks, X0[gi], F0[gi], gi, gi * 0, 0, T, arm_policy(engine, lane_arms, torch.full((A * B,), T), s, arms[0])
    )
    kind = torch.arange(A, device=X0.device).repeat_interleave(B)
    if n_random:
        gr = rows.repeat(n_random)
        Xr, Fr = rollout(tasks, X0[gr], F0[gr], gr, gr * 0, 0, T, _random_policy(gen))
        X, F, gi = torch.cat([X, Xr]), torch.cat([F, Fr]), torch.cat([gi, gr])
        kind = torch.cat([kind, torch.full((len(gr),), A, device=X0.device)])
    return X, F, kind, gi


def train(cfg: DictConfig, engine: Engine, branch: AcqBranch, run: Run) -> pd.DataFrame:
    t, tr = cfg.task, cfg.task.train
    d, n_init, T = shape(t)
    arms = [str(a) for a in t.arms]
    gen = torch.Generator().manual_seed(int(cfg.seed) + 11)
    Xs, Fs = [], []
    for tasks, X0, F0, _ in task_chunks(t, tr.tasks, d, n_init):
        X, F, _, _ = runs(engine, tasks, X0, F0, arms, T, int(tr.cand_seed), int(tr.n_random), gen)
        Xs.append(X.cpu())
        Fs.append(F.cpu())
    X_all, F_all = torch.cat(Xs), torch.cat(Fs)
    steps, B = int(tr.steps), int(tr.batch)
    opt = torch.optim.AdamW(branch.parameters(), lr=float(tr.lr), weight_decay=float(tr.weight_decay))
    warm = int(tr.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / steps)))
    )
    rows = []
    branch.train()
    for step in range(steps):
        tb = int(torch.randint(0, T, (1,), generator=gen))
        n = n_init + tb
        idx = torch.randint(0, len(X_all), (B,), generator=gen)
        X, Y = X_all[idx, :n].to(t.device), F_all[idx, :n].to(t.device)
        xq, tsc, valid, _ = teacher(engine, arms, X, Y, extra_queries(X, Y, int(tr.n_extra), gen))
        m = int(tr.n_queries)  # train on a subset: every arm's top 16, random others
        if m < xq.shape[1]:
            key = torch.rand(xq.shape[:2], generator=gen).to(t.device)
            top = tsc.masked_fill(~valid[..., None], float("-inf")).topk(16, 1).indices.flatten(1)
            key.scatter_(1, top, -1.0)
            keep = key.topk(m, 1, largest=False).indices
            xq, valid = xq.gather(1, keep[..., None].expand(-1, -1, d)), valid.gather(1, keep)
            tsc = tsc.gather(1, keep[..., None].expand(-1, -1, tsc.shape[-1]))
        s = branch_scores(branch, engine, X, Y, xq, budget_features(tb, T, n, B, t.device), chunk=B)
        loss, parts = rank_loss(s, tsc, valid, gen)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % int(tr.log_every) == 0 or step == steps - 1:
            ag = agreement(s.detach(), tsc, valid)
            row = {
                "step": step,
                "loss": float(loss.detach()),
                **parts,
                "top1": float(ag["top1"].mean()),
                "good_pick": float((ag["rank_pct"] <= 0.01).float().mean()),
                "spearman": float(ag["spearman"].mean()),
            }
            rows.append(row)
            run.log_metrics({f"train/{k}": float(v) for k, v in row.items() if k != "step"}, step=step)
    branch.eval()
    return pd.DataFrame(rows)


def dense_set(d: int, n_grid: int, device: str) -> torch.Tensor:
    """[G, d]: a regular grid for d <= 2, scrambled Sobol points otherwise."""
    if d == 1:
        return torch.linspace(0, 1, n_grid, device=device)[:, None]
    if d == 2:
        m = round(math.sqrt(n_grid))
        g = torch.linspace(0, 1, m, device=device)
        return torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    return torch.quasirandom.SobolEngine(d, scramble=True, seed=99).draw(n_grid).to(device)


@torch.no_grad()
def evaluate(cfg: DictConfig, engine: Engine, branch: AcqBranch, run: Run) -> pd.DataFrame:
    t, ev = cfg.task, cfg.task.eval
    d, n_init, T = shape(t)
    dev = t.device
    arms = [str(a) for a in t.arms]
    A = len(arms)
    kinds = [*arms, "random"]
    gen = torch.Generator().manual_seed(int(cfg.seed) + 23)
    grid = dense_set(d, int(ev.n_dense), dev)
    st_rows, ex_rows, ctx_rows, cl_rows = [], [], [], []
    curves: dict[tuple[str, str], list[torch.Tensor]] = {}
    n_cl, n_ex = int(ev.n_closed_loop), int(ev.n_examples)
    ex_steps = sorted({1, T // 3, (2 * T) // 3})
    for tasks, X0, F0, c0 in task_chunks(t, ev.tasks, d, n_init):
        B = tasks.B
        X, F, kind, row = runs(engine, tasks, X0, F0, arms, T, 0, int(ev.n_random), gen)
        L = len(X)
        # --- agreement at every `eval_every`-th step of every run
        for tb in range(0, T, int(ev.every)):
            n = n_init + tb
            Xn, Yn = X[:, :n], F[:, :n]
            C1 = seeded_candidates(engine, Xn, Yn, 1)
            extra = torch.cat([C1, grid[None].expand(L, -1, -1)], 1)
            xq, tsc, valid, mu = teacher(engine, arms, Xn, Yn, extra)
            s = branch_scores(branch, engine, Xn, Yn, xq, budget_features(tb, T, n, L, dev))
            P0, P1 = xq.shape[1] - extra.shape[1], C1.shape[1]
            for qs, sl in (("cand0", slice(0, P0)), ("cand1", slice(P0, P0 + P1)), ("dense", slice(P0 + P1, None))):
                ag = {k: v.cpu().numpy() for k, v in agreement(s[:, sl], tsc[:, sl], valid[:, sl]).items()}
                for a, arm in enumerate(arms):
                    st_rows.append(
                        pd.DataFrame(
                            {
                                "task": (c0 + row).cpu().numpy(),
                                "run": np.asarray(kinds)[kind.cpu().numpy()],
                                "step": tb,
                                "query_set": qs,
                                "arm": arm,
                                **{k: v[:, a] for k, v in ag.items()},
                            }
                        )
                    )
            # --- examples: the dense set's scores at a few states of the first tasks (d <= 2)
            if d <= 2 and c0 == 0 and tb in ex_steps:
                for i in range(min(n_ex, B)):
                    for kname in ("ei", "random"):
                        if kname not in kinds:
                            continue
                        ln = int(((row == i) & (kind == kinds.index(kname))).nonzero()[0, 0])
                        eid = f"task{i}_{kname}_t{tb}"
                        f_true = tasks.f(grid[None], torch.tensor([i], device=dev))[0].cpu().numpy()
                        g = pd.DataFrame(grid.cpu().numpy(), columns=[f"x{j}" for j in range(d)])
                        g["example"], g["f"] = eid, f_true
                        g["pfn_mean"] = mu[ln, P0 + P1 :].cpu().numpy()
                        g["valid"] = valid[ln, P0 + P1 :].cpu().numpy()
                        for a, arm in enumerate(arms):
                            g[f"teacher:{arm}"] = tsc[ln, P0 + P1 :, a].cpu().numpy()
                            g[f"student:{arm}"] = s[ln, P0 + P1 :, a].cpu().numpy()
                        ex_rows.append(g)
                        c = pd.DataFrame(Xn[ln].cpu().numpy(), columns=[f"x{j}" for j in range(d)])
                        c["y"], c["example"], c["step"], c["run"] = Yn[ln].cpu().numpy(), eid, tb, kname
                        c["fstar"] = float(tasks.fstar[i])
                        ctx_rows.append(c)
            del xq, tsc, valid, mu, s
        # --- closed loop: the branch's column as the acquisition, against the arm (seed-0 candidates)
        m = min(B, max(0, n_cl - c0))
        if m > 0:
            ti = torch.arange(m, device=dev)
            gi = ti.repeat(A)
            cols = torch.arange(A, device=dev).repeat_interleave(m)
            Xs, Fs = rollout(tasks, X0[gi], F0[gi], gi, gi * 0, 0, T, branch_policy(branch, engine, cols, T, 0))
            teach = (kind < A) & (row < m)  # lanes ordered by arm, then task: the same order as (cols, gi)
            Xt, Ft = X[teach], F[teach]
            low = torch.full((B,), float("inf"), device=dev)
            low.scatter_reduce_(0, row, F.min(1).values, "amin")
            low.scatter_reduce_(0, gi, Fs.min(1).values, "amin")
            tasks.lower_fstar(low[:, None])
            fst = tasks.fstar[gi]
            same = (Xs[:, n_init:] - Xt[:, n_init:]).abs().amax(-1) < 1e-9  # [A m, T]
            same_steps = same.float().cumprod(1).sum(1)
            cl_rows.append(
                pd.DataFrame(
                    {
                        "task": (c0 + gi).cpu().numpy(),
                        "arm": np.asarray(arms)[cols.cpu().numpy()],
                        "regret_teacher": (Ft.min(1).values - fst).clamp(min=0).cpu().numpy(),
                        "regret_student": (Fs.min(1).values - fst).clamp(min=0).cpu().numpy(),
                        "same_steps": same_steps.cpu().numpy(),
                        "same_run": (same_steps == T).cpu().numpy(),
                        "same_picks": same.float().mean(1).cpu().numpy(),
                    }
                )
            )
            for name, Fx in (("teacher", Ft), ("student", Fs)):
                reg = (Fx.cummin(1).values[:, n_init - 1 :] - fst[:, None]).clamp(min=0)  # [A m, T + 1]
                lr = torch.log10(reg.double() + float(t.eps)).view(A, m, -1)
                for a, arm in enumerate(arms):
                    curves.setdefault((arm, name), []).append(lr[a].cpu())
    states = pd.concat(st_rows, ignore_index=True)
    states["good"] = states.rank_pct <= GOOD_PICK
    states["ok5"] = states.rank_pct <= 0.05
    keys = ["run", "step", "query_set", "arm"]
    cols = ["top1", "good", "ok5", "rank_pct", "gap", "spearman", "spearman_top"]
    grouped = states.astype({"good": float, "ok5": float}).groupby(keys)
    agree = pd.DataFrame(grouped[cols].mean()).reset_index()
    agree = agree.assign(n_states=grouped.size().to_numpy())
    closed = pd.concat(cl_rows, ignore_index=True)
    cur = [
        {"arm": arm, "policy": name, "step": i, "log_regret": float(v), "n_tasks": len(torch.cat(parts))}
        for (arm, name), parts in curves.items()
        for i, v in enumerate(torch.cat(parts).mean(0))
    ]
    run.log_table("agree", agree)
    run.log_table("closed", closed)
    run.log_table("curves", pd.DataFrame(cur))
    if ex_rows:
        ex = pd.concat(ex_rows, ignore_index=True)
        num = ex.select_dtypes("float").columns
        ex[num] = ex[num].astype("float32")
        run.log_table("examples", ex)
        run.log_table("examples_ctx", pd.concat(ctx_rows, ignore_index=True))
    summary = summarize(states, closed, float(t.eps), int(t.n_boot))
    run.log_table("summary", summary)
    return summary


def acq_distill(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(t.tf32)
    torch.backends.cudnn.allow_tf32 = bool(t.tf32)
    pfn = load(t.pfn)[0].to(t.device).eval()
    for p in pfn.parameters():
        p.requires_grad_(False)
    engine = Engine(pfn, None, _candidates(t.candidates), base_seed=cfg.seed, chunk=int(t.chunk))
    blocks = cast(torch.nn.ModuleList, pfn.blocks)
    d_pfn = int(cast(torch.nn.Linear, blocks[0].attn.q_proj).in_features)  # type: ignore[union-attr]
    b = {str(k): v for k, v in _dict(t.branch).items()}
    spec = BranchSpec(
        width=int(b["width"]),
        heads=int(b["heads"]),
        ff_mult=int(b["ff_mult"]),
        read=tuple(int(x) for x in b["read"]) if b["read"] is not None else None,
        self_attn=bool(b["self_attn"]),
        query_state=bool(b["query_state"]),
        dropout=float(b["dropout"]),
    )
    branch = AcqBranch(d_pfn, len(blocks), int(cast(int, pfn.max_x_dim)), len(t.arms), spec).to(t.device)
    run.log_summary(branch_params=float(sum(p.numel() for p in branch.parameters())))
    t0 = time.time()
    run.log_table("train", train(cfg, engine, branch, run))
    run.log_summary(**{"time/train_s": time.time() - t0})
    run.artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state": branch.state_dict(), "spec": _dict(t.branch), "arms": list(t.arms)}, run.artifacts / "branch.pt"
    )
    t0 = time.time()
    summary = evaluate(cfg, engine, branch, run)
    run.log_summary(**{"time/evaluate_s": time.time() - t0})
    for r in summary.to_dict("records"):
        run.log_summary(**{f"{k}/{r['arm']}": float(v) for k, v in r.items() if k != "arm"})
    verdict = distill_verdict(summary)
    run.log_summary(verdict_go=float(verdict == "GO"), verdict_nogo=float(verdict == "NO-GO"))
    return float(summary["cand0:good_pick"].to_numpy().min())

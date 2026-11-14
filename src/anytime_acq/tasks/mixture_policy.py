"""An in-context mixture of acquisition functions, trained by direct paired search on whole runs
(experimental/mixture.py; pre-registration: notebooks/17_2026-11-13_mixture_policy.ipynb).

One member = one d.
- Policy: `experimental/mixture.py`. theta [K, 3 + F]; the arms are its one-hot special cases.
- Objective J (lower = better), per task: w_final * log10(R_T + eps) + (1 - w_final) * mean_t log10(R_t + eps),
  R_t the regret after t steps, eps = `eps` (the floor).
- Search (`search`): antithetic perturbations theta +- sigma * e, both run on the same tasks and candidate sets
  (deterministic, so the difference is exact); the mean paired difference times e is the step direction (Adam).
  Every `val_every` generations the current theta is compared with the accepted one on validation tasks and
  accepted only if its mean J is lower.
  Phase 1, budget-only: the mask allows the constant and tau terms; it starts with most weight on the arm that is
  best on the validation tasks. Phase 2, context: starts from phase 1's accepted theta, the mask allows everything.
- Test (`test`), fresh tasks, candidate seeds 0 and 1: every arm, the uniform mixture, the budget-only policy and
  the context policy. Tables: `train`, `theta`, `test_per_task`, `test_curves`, `weights`, `summary`.
"""

import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load

from ..analysis.option_policy import policy_verdict
from ..analysis.stats import bootstrap_ci
from ..bo.engine import Engine
from ..bo.runner import initial_design
from ..bo.tasks import Tasks, draw_tasks, task_seed
from ..experimental.api import f_lanes, rollout
from ..experimental.mixture import FEATURES, N_TIME, mixture_policy, mixture_step, one_hot_theta
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .api import _dict
from .bo import _candidates

Chunk = tuple[Tasks, torch.Tensor, torch.Tensor]
Stats = tuple[torch.Tensor, torch.Tensor]


def draw(t: DictConfig, spec: DictConfig, d: int, n_init: int) -> list[Chunk]:
    N = int(spec.n_tasks)
    seeds = [task_seed(int(spec.split), d, int(spec.task_offset) + i) for i in range(N)]
    out = []
    for c0 in range(0, N, int(t.task_chunk)):
        with torch.enable_grad():  # draw_tasks refines f* by gradient descent
            tasks = draw_tasks(d, seeds[c0 : c0 + int(t.task_chunk)], **_dict(t.task_draw)).to(t.device)
        rows = torch.arange(tasks.B, device=t.device)
        X0 = initial_design(tasks, n_init).to(t.device)
        F0 = f_lanes(tasks, X0.reshape(-1, d), rows.repeat_interleave(n_init)).view(tasks.B, n_init)
        out.append((tasks, X0, F0))
    return out


@torch.no_grad()
def run_policies(
    t: DictConfig,
    engine: Engine,
    chunk: Chunk,
    theta: torch.Tensor,
    mask: torch.Tensor,
    stats: Stats,
    s: int = 0,
    record: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Best value so far [M, B, T + 1] (after 0..T steps) of every policy theta[m] on every task of the chunk."""
    tasks, X0, F0 = chunk
    B, M = tasks.B, len(theta)
    n_init, T = X0.shape[1], int(t.T)
    gi = torch.arange(B, device=X0.device).repeat(M)
    pol = torch.arange(M, device=X0.device).repeat_interleave(B)
    arms = [str(a) for a in t.arms]
    policy = mixture_policy(engine, arms, theta.to(X0.device), mask.to(X0.device), pol, T, s, stats, record)
    _, F = rollout(tasks, X0[gi], F0[gi], gi, gi * 0, 0, T, policy)
    return F.cummin(1).values[:, n_init - 1 :].view(M, B, T + 1)


def objective(best: torch.Tensor, fstar: torch.Tensor, eps: float, w_final: float) -> dict[str, torch.Tensor]:
    """best [M, B, T + 1], fstar [B] -> per policy and task: J, final and auc (log10 regret, lower = better)."""
    lr = torch.log10((best - fstar[None, :, None]).clamp(min=0).double() + eps)
    final, auc = lr[..., -1], lr[..., 1:].mean(-1)
    return {"J": w_final * final + (1 - w_final) * auc, "final": final, "auc": auc}


def _J(t: DictConfig, engine: Engine, chunks: list[Chunk], theta: torch.Tensor, mask: torch.Tensor, stats: Stats):
    """[M, n tasks]: J of every policy on all tasks of `chunks`; f* is lowered by what these runs find."""
    out = []
    for ch in chunks:
        best = run_policies(t, engine, ch, theta, mask, stats)
        fstar = torch.minimum(ch[0].fstar, best.amin((0, 2)))
        out.append(objective(best, fstar, float(t.eps), float(t.w_final))["J"])
    return torch.cat(out, 1)


def search(
    t: DictConfig,
    engine: Engine,
    train: list[Chunk],
    val: list[Chunk],
    theta0: torch.Tensor,
    mask: torch.Tensor,
    stats: Stats,
    phase: str,
    gen: torch.Generator,
    run: Run,
    step0: int,
) -> tuple[torch.Tensor, list[dict]]:
    sp = t.search
    P, sigma = int(sp.pairs), float(sp.sigma)
    theta = theta0.clone().requires_grad_(False)
    accepted = theta0.clone()
    j_acc = float(_J(t, engine, val, accepted[None], mask[None], stats).mean())
    m_adam, v_adam = torch.zeros_like(theta), torch.zeros_like(theta)
    rows = [{"phase": phase, "gen": -1, "J_train": float("nan"), "J_val": j_acc, "J_accepted": j_acc, "accepted": True}]
    for g in range(int(sp.generations)):
        e = torch.randn(P, *theta.shape, generator=gen) * mask
        th = torch.cat([theta + sigma * e, theta - sigma * e])
        J = _J(t, engine, [train[g % len(train)]], th, mask.expand(2 * P, -1), stats)  # [2P, B]
        diff = (J[:P] - J[P:]).mean(1).float().cpu()  # > 0: the + side is worse
        grad = (diff[:, None, None] * e).mean(0) / (diff.std().clamp(min=1e-12))
        m_adam = 0.9 * m_adam + 0.1 * grad
        v_adam = 0.99 * v_adam + 0.01 * grad**2
        theta = theta - float(sp.lr) * (m_adam / (1 - 0.9 ** (g + 1))) / (
            (v_adam / (1 - 0.99 ** (g + 1))).sqrt() + 1e-8
        )
        theta = theta * mask
        row = {"phase": phase, "gen": g, "J_train": float(J.mean()), "J_val": float("nan"), "accepted": False}
        if (g + 1) % int(sp.val_every) == 0 or g == int(sp.generations) - 1:
            jv = _J(t, engine, val, torch.stack([theta, accepted]), mask.expand(2, -1), stats).mean(1)
            row["J_val"] = float(jv[0])
            if float(jv[0]) < float(jv[1]):
                accepted, j_acc, row["accepted"] = theta.clone(), float(jv[0]), True
        row["J_accepted"] = j_acc
        rows.append(row)
        run.log_metrics({f"{phase}/{k}": float(v) for k, v in row.items() if k.startswith("J_")}, step=step0 + g)
    return accepted, rows


def mixture(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(t.tf32)
    torch.backends.cudnn.allow_tf32 = bool(t.tf32)
    pfn = load(t.pfn)[0].to(t.device).eval()
    engine = Engine(pfn, None, _candidates(t.candidates), base_seed=cfg.seed, chunk=int(t.chunk))
    d, n_init, T = int(t.d), int(t.n_init), int(t.T)
    arms = [str(a) for a in t.arms]
    K, nf = len(arms), N_TIME + len(FEATURES)
    gen = torch.Generator().manual_seed(int(cfg.seed) + 31)
    t0 = time.time()
    train, val = draw(t, t.train_tasks, d, n_init), draw(t, t.val_tasks, d, n_init)

    # --- standardization of the context features: states of the uniform mixture on the first training chunk
    tasks, X0, F0 = train[0]
    rows_ = torch.arange(tasks.B, device=t.device)
    ident: Stats = (torch.zeros(len(FEATURES), device=t.device), torch.ones(len(FEATURES), device=t.device))
    zero, const = torch.zeros(1, K, nf), torch.zeros(1, nf)
    const[0, 0] = 1.0
    pol0 = mixture_policy(engine, arms, zero.to(t.device), const.to(t.device), rows_ * 0, T, 0, ident)
    Xc, Fc = rollout(tasks, X0, F0, rows_, rows_ * 0, 0, T, pol0)
    fs = torch.cat([mixture_step(engine, arms, Xc[:, :n], Fc[:, :n], 0, T)["feats"] for n in range(n_init, n_init + T)])
    stats: Stats = (fs.mean(0), fs.std(0).clamp(min=1e-6))

    # --- the arms on the validation tasks: the best one is the comparator and the start of the search
    m_time = torch.zeros(nf)
    m_time[:N_TIME] = 1.0
    m_all = torch.ones(nf)
    th_arms = torch.stack([one_hot_theta(k, K) for k in range(K)])
    j_arms = _J(t, engine, val, th_arms, const.expand(K, -1), stats).mean(1)
    k_best = int(j_arms.argmin())
    run.log_summary(**{f"val_J/{a}": float(j) for a, j in zip(arms, j_arms, strict=True)})
    theta0 = torch.zeros(K, nf)
    theta0[k_best, 0] = float(t.search.init_bias)

    # --- search: budget-only, then context from it
    th_time, r1 = search(t, engine, train, val, theta0, m_time, stats, "budget", gen, run, 0)
    th_ctx, r2 = search(t, engine, train, val, th_time, m_all, stats, "context", gen, run, int(t.search.generations))
    run.log_table("train", pd.DataFrame(r1 + r2))
    run.log_summary(**{"time/search_s": time.time() - t0})
    cols = ["const", "tau", "tau2", *FEATURES]
    th_rows = [
        {"policy": p, "arm": arms[k], "term": cols[j], "theta": float(th[k, j])}
        for p, th in (("budget", th_time), ("context", th_ctx))
        for k in range(K)
        for j in range(nf)
    ]
    run.log_table("theta", pd.DataFrame(th_rows))
    run.artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"budget": th_time, "context": th_ctx, "stats": [s.cpu() for s in stats], "arms": arms, "features": FEATURES},
        run.artifacts / "policy.pt",
    )
    del train, val

    # --- test on fresh tasks, both candidate seeds
    t0 = time.time()
    names = [*arms, "uniform", "budget", "context"]
    theta = torch.cat([th_arms, zero, th_time[None], th_ctx[None]])
    mask = torch.cat([const.expand(K + 1, -1), m_time[None], m_all[None]])
    per, curves, wts = [], [], []
    off = 0
    for ch in draw(t, t.test_tasks, d, n_init):
        B = ch[0].B
        best, rec = {}, {}
        for s in (0, 1):
            rec[s] = []
            best[s] = run_policies(t, engine, ch, theta, mask, stats, s, rec[s])
        fstar = torch.minimum(ch[0].fstar, torch.minimum(best[0].amin((0, 2)), best[1].amin((0, 2))))
        for s in (0, 1):
            reg = (best[s] - fstar[None, :, None]).clamp(min=0)  # [M, B, T + 1]
            for m, name in enumerate(names):
                row = {
                    "task": off + np.arange(B),
                    "policy": name,
                    "seed": s,
                    "regret_final": reg[m, :, -1].cpu().numpy(),
                }
                for eps in t.report_eps:
                    o = objective(best[s][m : m + 1], fstar, float(eps), float(t.w_final))
                    row |= {f"{k}@{float(eps):g}": v[0].cpu().numpy() for k, v in o.items()}
                per.append(pd.DataFrame(row))
            curves.append(reg.cpu())
            W = torch.stack(rec[s], 0).view(T, len(names), B, K)  # per step: lanes ordered by policy, then task
            wts.append(W)
        off += B
    per_task = pd.concat(per, ignore_index=True)
    run.log_table("test_per_task", per_task)
    n_ch = len(curves) // 2
    cur_rows = []
    for s in (0, 1):
        reg = torch.cat(curves[s::2], 1)  # [M, N, T + 1]
        for eps in t.report_eps:
            lr = torch.log10(reg.double() + float(eps)).mean(1)
            share = (reg > float(eps)).float().mean(1)
            for m, name in enumerate(names):
                for i in range(T + 1):
                    cur_rows.append(
                        {"policy": name, "seed": s, "eps": float(eps), "step": i, "log_regret": float(lr[m, i]),
                         "share_above": float(share[m, i])}
                    )  # fmt: skip
    run.log_table("test_curves", pd.DataFrame(cur_rows))
    W = torch.cat(wts[0::2], 2).mean(2)  # seed 0: [T, M, K]
    w_rows = [
        {"policy": name, "step": i, "arm": arms[k], "weight": float(W[i, m, k])}
        for m, name in enumerate(names)
        if name in ("budget", "context")
        for i in range(T)
        for k in range(K)
    ]
    run.log_table("weights", pd.DataFrame(w_rows))
    assert n_ch >= 1

    # --- summary: contrasts on the objective at the training floor (positive = better than the comparator)
    eps0 = f"{float(t.eps):g}"
    rows = []
    best_arm = arms[k_best]
    wide = {
        m: per_task.pivot(index=["task", "seed"], columns="policy", values=f"{m}@{eps0}") for m in ("J", "final", "auc")
    }
    for s in (0, 1):
        for metric, wd in wide.items():
            w_s = wd.xs(s, level="seed")
            for variant in ("budget", "context", "uniform"):
                for contrast, ref in (("vs_base", best_arm), ("vs_static", "budget"), ("vs_ei", "ei")):
                    if variant == ref or ref not in w_s or (contrast == "vs_static" and variant != "context"):
                        continue
                    x = (w_s[ref] - w_s[variant]).to_numpy()
                    lo, hi = bootstrap_ci(x, n_boot=int(t.n_boot))
                    name = variant if metric == "J" else f"{variant}:{metric}"
                    rows.append(
                        {"variant": name, "metric": metric, "seed": s, "contrast": contrast, "ref": ref,
                         "mean": float(x.mean()), "lo": lo, "hi": hi, "n_tasks": len(x)}
                    )  # fmt: skip
            for a in arms:
                lvl = float(w_s[a].to_numpy().mean())
                rows.append(
                    {"variant": f"arm:{a}", "metric": metric, "seed": s, "contrast": "level", "ref": "",
                     "mean": lvl, "lo": float("nan"), "hi": float("nan"), "n_tasks": len(w_s)}
                )  # fmt: skip
    summary = pd.DataFrame(rows)
    run.log_table("summary", summary)
    run.log_summary(**{"time/test_s": time.time() - t0, "best_arm_index": float(k_best)})
    s0 = summary[(summary.seed == 0) & (summary.metric == "J") & (summary.contrast != "level")]
    for variant, contrast, mean in zip(s0.variant, s0.contrast, s0["mean"], strict=True):
        run.log_summary(**{f"gain/{variant}/{contrast}": float(mean)})
    verdict = policy_verdict(pd.DataFrame(summary[summary.metric == "J"]), "context", float(t.worthwhile_log10))
    run.log_summary(verdict_go=float(verdict == "GO"), verdict_nogo=float(verdict == "NO-GO"))
    primary = (s0.variant == "context") & (s0.contrast == "vs_base")
    return float(s0.loc[primary, "mean"].to_numpy()[0])

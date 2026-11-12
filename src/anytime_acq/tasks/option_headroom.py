"""Where is the headroom over EI? Basin decomposition of static arms and exact, luck-corrected option labels
(experimental/options.py; research_log/decisions/0008; pre-registration: notebooks/12_2026-11-12_option_headroom.ipynb).

One member = one d. Every policy picks from deterministic candidate sets (replicate seed s), the tasks are
noise-free, so one continuation is an exact outcome; the seeds are the luck replicates.
- Part A (split `part_a.split`): full BO runs of every static arm under every seed. Per (task, arm, seed, step):
  the best f, the log10 regret, and whether the incumbent drains (local descent on the true f) to a basin whose
  minimum is within delta bins of f* (`good<delta>`).
- Part B (split `part_b.split`, fresh tasks): one branch state per task (EI under seed 0 to a uniform step
  t <= t_max). From it, under every seed: the base (EI to T) and every option "arm a for k steps, then EI to T",
  the arms being the static arms other than EI and null arms ``ei@j``. Per lane: final log10 regret, the final
  incumbent's basin, the first step its incumbent is in a good basin, whether its first pick differs from the
  base's, and the belief at t + k (`experimental/options.py::belief_at_optimum`).

Tables: `part_a` (one row per task x arm x seed x step), `part_b` (one row per state x option x seed),
`descent_check` (good-basin agreement between two descent settings), `headroom` (cross-fitted headroom per k and
option family), `arms_b` (mean advantage per option). Verdict: `analysis/options.py::option_verdict`.
"""

import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load

from ..analysis.options import crossfit_table, option_verdict
from ..bo.engine import Engine
from ..bo.runner import initial_design
from ..bo.tasks import Tasks, draw_tasks, task_seed
from ..experimental.api import f_lanes, rollout
from ..experimental.options import DescentSpec, belief_at_optimum, descend, option_policy
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .api import _dict
from .bo import _candidates

BIN = 1e-3  # pfn_large's bar width on the ECDF scale


def _setup(cfg: DictConfig) -> tuple[Engine, DescentSpec, DescentSpec]:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(t.tf32)
    torch.backends.cudnn.allow_tf32 = bool(t.tf32)
    pfn = load(t.pfn)[0].to(t.device).eval()
    engine = Engine(pfn, None, _candidates(t.candidates), base_seed=cfg.seed, chunk=int(t.chunk))
    ds = DescentSpec(int(t.descent.steps), float(t.descent.lr), int(t.descent.chunk))
    dc = DescentSpec(int(t.descent.check_steps), float(t.descent.check_lr), int(t.descent.chunk))
    return engine, ds, dc


def _tasks(t: DictConfig, part: DictConfig, d: int) -> tuple[Tasks, list[int]]:
    seeds = [task_seed(int(part.split), d, int(part.task_offset) + i) for i in range(int(part.n_tasks))]
    with torch.enable_grad():  # draw_tasks refines f* by gradient descent
        tasks = draw_tasks(d, seeds, **_dict(t.task_draw)).to(t.device)
    return tasks, seeds


def _incumbents(X: torch.Tensor, F: torch.Tensor, n_init: int) -> torch.Tensor:
    """[L, T+1, d]: the incumbent after the initial design (step 0) and after every BO step."""
    L, N, _ = X.shape
    out = []
    for n in range(n_init, N + 1):
        j = F[:, :n].argmin(1)
        out.append(X[torch.arange(L, device=X.device), j])
    return torch.stack(out, 1)


def part_a(cfg: DictConfig, engine: Engine, ds: DescentSpec, dc: DescentSpec, run: Run) -> pd.DataFrame:
    t = cfg.task
    dev, d = t.device, int(t.d)
    n_init, T = int(t.n_init_per_d * d), int(t.T_per_d * d)
    tasks, seeds = _tasks(t, t.part_a, d)
    B = tasks.B
    X0 = initial_design(tasks, n_init).to(dev)
    rows = torch.arange(B, device=dev)
    F0 = f_lanes(tasks, X0.reshape(-1, d), rows.repeat_interleave(n_init)).view(B, n_init)
    trajs = {}
    for arm in t.arms:
        for s in t.seeds:
            pol = option_policy(engine, [str(arm)] * B, torch.full((B,), T), int(s))
            trajs[(str(arm), int(s))] = rollout(tasks, X0, F0, rows, rows * 0, 0, T, pol)
    for _, F in trajs.values():
        tasks.lower_fstar(F)
    out = []
    for (arm, s), (X, F) in trajs.items():
        inc = _incumbents(X, F, n_init)  # [B, T+1, d]
        best = torch.stack([F[:, : n_init + k].min(1).values for k in range(T + 1)], 1)
        floc = descend(tasks, inc.reshape(-1, d), rows.repeat_interleave(T + 1), ds).view(B, T + 1)
        reg = (best - tasks.fstar[:, None]).clamp(min=0)
        df = pd.DataFrame(
            {
                "task": np.repeat(np.arange(B), T + 1),
                "seed": np.repeat(seeds, T + 1),
                "arm": arm,
                "rep": s,
                "t": np.tile(np.arange(T + 1), B),
                "best_f": best.flatten().cpu().numpy(),
                "log_regret": torch.log10(reg + float(t.eps)).flatten().cpu().numpy(),
                "regret_bins": (reg / BIN).flatten().cpu().numpy(),
                "basin_gap_bins": ((floc - tasks.fstar[:, None]) / BIN).flatten().cpu().numpy(),
            }
        )
        out.append(df)
    A = pd.concat(out, ignore_index=True)
    # robustness of the basin label: the final incumbents of every run, descended with a second setting
    X, F = trajs[(str(t.arms[0]), int(t.seeds[0]))]
    inc = _incumbents(X, F, n_init)[:, -1]
    a1 = descend(tasks, inc, rows, ds) - tasks.fstar
    a2 = descend(tasks, inc, rows, dc) - tasks.fstar
    chk = [
        {"delta_bins": float(dl), "agree": float(((a1 <= dl * BIN) == (a2 <= dl * BIN)).float().mean())}
        for dl in t.deltas
    ]
    run.log_table("descent_check", pd.DataFrame(chk))
    return A


def _branch_steps(seeds: list[int], t_max: int) -> torch.Tensor:
    u = torch.stack([torch.rand(1, generator=torch.Generator().manual_seed(int(s) % (2**31) + 17)) for s in seeds])
    return (u[:, 0] * (t_max + 1)).long().clamp(max=t_max)


def part_b(cfg: DictConfig, engine: Engine, ds: DescentSpec) -> pd.DataFrame:
    t = cfg.task
    dev, d = t.device, int(t.d)
    n_init, T = int(t.n_init_per_d * d), int(t.T_per_d * d)
    tasks, seeds = _tasks(t, t.part_b, d)
    B = tasks.B
    X0 = initial_design(tasks, n_init).to(dev)
    rows = torch.arange(B, device=dev)
    F0 = f_lanes(tasks, X0.reshape(-1, d), rows.repeat_interleave(n_init)).view(B, n_init)
    bt = _branch_steps(seeds, int(t.part_b.t_max)).to(dev)
    tmax = int(bt.max())
    # the shared prefix: EI under seed 0 up to each state's branch step
    Xp, Fp = rollout(tasks, X0, F0, rows, rows * 0, 0, tmax, option_policy(engine, ["ei"] * B, torch.full((B,), T), 0))
    opts = [("base", "ei", 0)]
    opts += [(f"{a}_k{k}", str(a), int(k)) for a in t.arms if str(a) != "ei" for k in t.ks]
    opts += [(f"null{j}_k{k}", f"ei@{j}", int(k)) for j in range(int(t.n_null)) for k in t.ks]
    lanes = []  # (state, option name, arm, k, seed)
    res: dict[str, list] = {k: [] for k in ("X", "F")}
    meta = []
    for tb in bt.unique().tolist():
        g = (bt == tb).nonzero().squeeze(1)
        n = n_init + tb
        for s in t.seeds:
            arms = [a for _, a, _ in opts for _ in g]
            until = torch.tensor([tb + k for _, _, k in opts for _ in g], device=dev)
            gi = g.repeat(len(opts))
            X, F = Xp[gi, :n], Fp[gi, :n]
            X, F = rollout(tasks, X, F, gi, gi * 0, tb, T, option_policy(engine, arms, until, int(s)))
            res["X"].append(X)
            res["F"].append(F)
            for name, arm, k in opts:
                for gg in g.tolist():
                    meta.append((gg, name, arm, k, int(s), tb))
            lanes.append(gi)
    gi_all = torch.cat(lanes)
    F_all = torch.cat(res["F"])
    fin = F_all.min(1).values
    low = torch.full((B,), float("inf"), device=dev).scatter_reduce(0, gi_all, fin, "amin")
    tasks.lower_fstar(torch.minimum(low, Fp.min(1).values)[:, None])
    M = pd.DataFrame(meta, columns=["state", "option", "arm", "k", "rep", "t"])
    reg = (fin - tasks.fstar[gi_all]).clamp(min=0)
    M["log_regret"] = torch.log10(reg + float(t.eps)).cpu().numpy()
    M["regret_bins"] = (reg / BIN).cpu().numpy()
    X_all = torch.cat(res["X"])
    inc_T = X_all[torch.arange(len(X_all), device=dev), F_all.argmin(1)]
    M["basin_gap_bins"] = ((descend(tasks, inc_T, gi_all, ds) - tasks.fstar[gi_all]) / BIN).cpu().numpy()
    # the first pick after the branch (disagreement with the base's)
    first = X_all[torch.arange(len(X_all), device=dev), torch.tensor(M.t.to_numpy() + n_init, device=dev)]
    M["first_x"] = list(first.cpu().numpy())
    # belief at the end of the option (t + k): P(y(x*) < state incumbent), distance(belief argmin, x*)
    st_inc = torch.stack([Fp[b, : n_init + int(bt[b])].min() for b in range(B)])
    thr = st_inc[gi_all]
    pis, dists = torch.empty(len(M), device=dev), torch.empty(len(M), device=dev)
    for (tb, k), idx in M.groupby(["t", "k"]).groups.items():
        ii = torch.tensor(np.asarray(idx), device=dev)
        n = n_init + int(tb) + max(int(k), 1)  # the base's belief is read after one step, as k = 1
        n = min(n, X_all.shape[1])
        for s in t.seeds:
            jj = ii[torch.tensor((M.rep.to_numpy()[np.asarray(idx)] == int(s)), device=dev)]
            if len(jj) == 0:
                continue
            b = belief_at_optimum(engine, X_all[jj, :n], F_all[jj, :n], tasks.xstar[gi_all[jj]], thr[jj], int(s))
            pis[jj], dists[jj] = b["pi_xstar"], b["dist_argmin_xstar"]
    M["pi_xstar"] = pis.cpu().numpy()
    M["dist_argmin_xstar"] = dists.cpu().numpy()
    M["seed"] = [seeds[i] for i in M.state]
    return M


def option_headroom(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    engine, ds, dc = _setup(cfg)
    t0 = time.time()
    A = part_a(cfg, engine, ds, dc, run)
    run.log_summary(**{"time/part_a_s": time.time() - t0})
    run.log_table("part_a", A)
    t0 = time.time()
    M = part_b(cfg, engine, ds)
    run.log_summary(**{"time/part_b_s": time.time() - t0})
    fx = np.stack(M.pop("first_x").to_numpy())
    base = M[M.option == "base"].set_index(["state", "rep"])
    bx = fx[M.option.to_numpy() == "base"]
    bmap = {key: bx[i] for i, key in enumerate(base.index)}
    M["disagree"] = [
        float(np.linalg.norm(fx[i] - bmap[(s, r)])) > 1e-7 for i, (s, r) in enumerate(zip(M.state, M.rep, strict=True))
    ]
    M["adv"] = [base.loc[(s, r), "log_regret"] for s, r in zip(M.state, M.rep, strict=True)] - M.log_regret
    run.log_table("part_b", M)
    head, arms_b = crossfit_table(M, [int(k) for k in t.ks], int(t.n_boot))
    run.log_table("headroom", head)
    run.log_table("arms_b", arms_b)
    verdict = option_verdict(head, int(t.primary_k), float(t.worthwhile_log10))
    for r in head.to_dict("records"):
        run.log_summary(**{f"H/{r['family']}/k{r['k']}": float(r["H"])})
    run.log_summary(verdict_go=float(verdict == "GO"))
    return float(head.loc[(head.family == "arms") & (head.k == int(t.primary_k)), "H"].iloc[0])

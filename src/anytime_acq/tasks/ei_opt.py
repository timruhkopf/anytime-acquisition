"""How well is EI optimized, and does a deterministic refined EI change BO? (experimental/ei_opt.py; issue #13;
pre-registration: notebooks/11_2026-11-12_ei_opt.ipynb.)

One member = one d.
- Part A (states): at the labeled states of the pinned round-1 label set, every variant's pick under `n_reps`
  replicate candidate sets: its EI relative to the best EI found by any variant, the distance between replicate
  picks, and the true f at the picks (in PFN bins relative to the state's incumbent).
- Part B (rollouts): full BO runs (n_init, T) of every variant on `n_tasks` tasks of split `split`, `n_reps`
  replicates each (replicate = the candidate-set nuisance stream; the fixed variants ignore it). Final regret
  per (task, variant, replicate).

Tables: `states` (one row per state x variant x replicate), `final` (one row per task x variant x replicate),
`spread` (per task x variant: sd over replicates), `summary` (by variant), `paired` (variant - sampled, paired over
tasks on the replicate-mean, bootstrap CI). The verdict: `analysis/ei_opt.py::ei_opt_verdict`.
"""

import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load

from ..analysis.api import paired_delta
from ..analysis.ei_opt import ei_opt_verdict, summarize
from ..bo.engine import Engine
from ..bo.runner import initial_design
from ..bo.tasks import draw_tasks, task_seed
from ..experimental.api import f_lanes, normalized_regret, rollout, task_median
from ..experimental.ei_opt import VARIANTS, RefineSpec, ei_policy, pick
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .api import _dict
from .api_gate import load_labels
from .bo import _candidates

BIN = 1e-3  # pfn_large's bar width on the ECDF scale


def _spec(t: DictConfig) -> RefineSpec:
    r = t.refine
    return RefineSpec(int(r.k), int(r.steps), float(r.lr))


def state_part(cfg: DictConfig, engine: Engine, run: Run) -> pd.DataFrame:
    t = cfg.task
    dev = t.device
    d, n_init = int(t.d), int(t.n_init_per_d * t.d)
    L = load_labels(cfg)
    S = min(int(t.states.n_states), len(L["seed"]))
    R = int(t.n_reps)
    spec = _spec(t)
    with torch.enable_grad():  # draw_tasks refines f* by gradient descent
        tasks = draw_tasks(d, L["seed"][:S].tolist(), **_dict(t.states.task_draw)).to(dev)
    out = []
    for tb in L["branch_t"][:S].unique().tolist():
        idx = (L["branch_t"][:S] == tb).nonzero().squeeze(1)
        for s in range(0, len(idx), int(t.group)):
            g = idx[s : s + int(t.group)]
            X, F = L["X"][g, : n_init + tb].to(dev), L["F"][g, : n_init + tb].to(dev)
            rows = g.to(dev)
            inc = F.min(1).values
            picks = {}
            for v in VARIANTS:
                for rep in range(R):
                    reps = torch.full((len(g),), rep, device=dev, dtype=torch.long)
                    picks[(v, rep)] = pick(engine, v, X, F, rows, reps, tb, S, spec)
            best = torch.stack([e for _, e in picks.values()]).max(0).values
            for (v, rep), (x, e) in picks.items():
                f = f_lanes(tasks, x, rows)
                out.append(
                    pd.DataFrame(
                        {
                            "state": g.numpy(),
                            "t": tb,
                            "variant": v,
                            "rep": rep,
                            "ei": e.cpu().numpy(),
                            "ei_ratio": (e / best.clamp_min(1e-30)).cpu().numpy(),
                            "f_rel_bins": ((f - inc) / BIN).cpu().numpy(),
                            **{f"x{i}": x[:, i].cpu().numpy() for i in range(d)},
                        }
                    )
                )
    st = pd.concat(out, ignore_index=True)
    gap = (torch.stack([L["F"][i, : n_init + int(L["branch_t"][i])].min() for i in range(S)]) - L["fstar"][:S]) / BIN
    st["gap_bins"] = gap.numpy()[st.state.to_numpy()]
    return st


def rollout_part(cfg: DictConfig, engine: Engine) -> pd.DataFrame:
    t = cfg.task
    dev = t.device
    d, n_init, T = int(t.d), int(t.n_init_per_d * t.d), int(t.T_per_d * t.d)
    B, R = int(t.rollouts.n_tasks), int(t.n_reps)
    spec = _spec(t)
    seeds = [task_seed(int(t.rollouts.split), d, int(t.rollouts.task_offset) + i) for i in range(B)]
    with torch.enable_grad():
        tasks = draw_tasks(d, seeds, **_dict(t.rollouts.task_draw)).to(dev)
    X0 = initial_design(tasks, n_init).to(dev)
    rows0 = torch.arange(B, device=dev)
    F0 = f_lanes(tasks, X0.reshape(-1, d), rows0.repeat_interleave(n_init)).view(B, n_init)
    Fs = {}
    for v in VARIANTS:
        for rep in range(R):
            reps = torch.full((B,), rep, device=dev, dtype=torch.long)
            _, F = rollout(tasks, X0, F0, rows0, reps, 0, T, ei_policy(engine, v, B, spec))
            Fs[(v, rep)] = F
    for F in Fs.values():
        tasks.lower_fstar(F)
    fmed = task_median(tasks)
    rows = []
    for (v, rep), F in Fs.items():
        reg = (F.min(1).values - tasks.fstar).clamp(min=0)
        rows.append(
            pd.DataFrame(
                {
                    "task": np.arange(B),
                    "seed": seeds,
                    "variant": v,
                    "rep": rep,
                    "norm_regret": normalized_regret(F, tasks.fstar, fmed).cpu().numpy(),
                    "regret_bins": (reg / BIN).cpu().numpy(),
                    "log_regret": torch.log10(reg + float(t.log_eps)).cpu().numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def ei_opt(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(t.tf32)
    torch.backends.cudnn.allow_tf32 = bool(t.tf32)
    pfn = load(t.pfn)[0].to(t.device).eval()
    engine = Engine(pfn, None, _candidates(t.candidates), base_seed=cfg.seed, chunk=int(t.chunk))
    t0 = time.time()
    st = state_part(cfg, engine, run)
    run.log_summary(**{"time/states_s": time.time() - t0})
    run.log_table("states", st)
    t0 = time.time()
    fin = rollout_part(cfg, engine)
    run.log_summary(**{"time/rollouts_s": time.time() - t0})
    run.log_table("final", fin)
    spread, summary = summarize(st, fin)
    run.log_table("spread", spread)
    run.log_table("summary", summary)
    cols = ["norm_regret", "regret_bins", "log_regret"]
    mean = pd.DataFrame(fin.groupby(["task", "variant"], as_index=False)[cols].mean()).rename(
        columns={"variant": "arm"}
    )
    paired = []
    for v in VARIANTS[1:]:
        for col in ("norm_regret", "regret_bins", "log_regret"):
            paired.append({"metric": col, **paired_delta(mean, v, "sampled", col=col)})
    pt = pd.DataFrame(paired)
    run.log_table("paired", pt)
    verdict = ei_opt_verdict(summary, pt)
    for r in summary.to_dict("records"):
        for k, val in r.items():
            if k != "variant":
                run.log_summary(**{f"{r['variant']}/{k}": float(val)})
    run.log_summary(verdict_go=float(verdict == "GO"))
    return float(pt.loc[(pt.a == "fixed+refine") & (pt.metric == "norm_regret"), "rel"].iloc[0])

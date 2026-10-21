"""Amortized EI with a PFN-VLA action expert (readout/): train on EI-Boltzmann targets from the frozen PFN,
then evaluate one-step quality (EI ratio) and closed-loop BO against EI itself.

One member = one d and one trunk (`pfn_all`, `pfn_last`, `d_only`; optionally probes). The dataset is
generated once per data config and cached under `paths.data/readout/`, so all trunks train on the same
states (its hash is a run tag).

Tables: `ei_ratio` (test state, source, n, ratio of the mode and of a sample), `closed_loop` (task,
seed, method, final best f, f*), `cheap` (the EI ratio of EI maximized over a cost-matched 256-point
random set: what one-pass amortization competes with).
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pfn_repro.checkpoints import load

from ..bo.engine import CandidateConfig, Engine
from ..bo.runner import Budget, initial_design, run_static
from ..bo.tasks import draw_tasks, task_seed
from ..evaluation import metric_names as M
from ..readout.data import States, Targets, ei_at, ei_targets, make_states
from ..readout.expert import ActionExpert, log_prob, mode, sample
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def _data(cfg: DictConfig, pfn: torch.nn.Module, dev: str) -> tuple[dict, str]:
    t = cfg.task
    spec = {"pfn": t.pfn, "d": t.d, **OmegaConf.to_container(t.data)}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(cfg.paths.data) / "readout" / f"{key}.pt"
    if path.exists():
        return torch.load(path, map_location=dev, weights_only=False), key
    dc = t.data
    cand = CandidateConfig(dc.n_sobol, dc.n_local, tuple(dc.local_sigmas))
    engine = Engine(pfn, candidates=cand, base_seed=17)
    n_init, T = dc.n_init_per_d * t.d, dc.T_per_d * t.d
    out = {}
    for name, split, n_tasks in (("train", dc.split_train, dc.n_tasks_train), ("test", dc.split_test, dc.n_tasks_test)):
        st = make_states(engine, t.d, split, n_tasks, n_init, T, dev)
        tg = ei_targets(pfn, st, dc.lam, dc.n_top, cand, seed=split)
        out[name] = (st, tg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out, key


def _batch(st: States, tg: Targets, idx: torch.Tensor) -> tuple:
    return st.X[idx], st.Y[idx], st.mask[idx], st.budget_left[idx], tg.x[idx], tg.w[idx]


def _loss(expert: ActionExpert, b: tuple) -> torch.Tensor:
    X, Y, m, bl, tx, tw = b
    return -(tw * log_prob(expert(X, Y, m, bl), tx)).sum(1).mean()


@torch.no_grad()
def _predict(expert: ActionExpert, st: States, chunk: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    modes, samples = [], []
    g = torch.Generator(device=st.X.device).manual_seed(0)
    for s in range(0, len(st), chunk):
        sl = slice(s, min(s + chunk, len(st)))
        mix = expert(st.X[sl], st.Y[sl], st.mask[sl], st.budget_left[sl])
        modes.append(mode(mix))
        samples.append(sample(mix, g))
    return torch.cat(modes), torch.cat(samples)


@torch.no_grad()
def _closed_loop(expert: ActionExpert, tasks, budget: Budget, how: str, dev: str) -> torch.Tensor:
    """BO where every step queries the expert's mode or a sample. -> F [B, n_init + T]."""
    B, d, N = tasks.B, tasks.d, budget.n_init + budget.T
    X = torch.zeros(B, N, d, device=dev)
    F = torch.zeros(B, N, device=dev)
    X[:, : budget.n_init] = initial_design(tasks, budget.n_init).to(dev)
    F[:, : budget.n_init] = tasks.f(X[:, : budget.n_init])
    g = torch.Generator(device=dev).manual_seed(1)
    for t in range(budget.T):
        n = budget.n_init + t
        m = (torch.arange(N, device=dev) < n).expand(B, -1)
        mix = expert(X, F * m, m, torch.full((B,), (budget.T - t) / budget.T, device=dev))
        x = mode(mix) if how == "mode" else sample(mix, g)
        X[:, n] = x
        F[:, n] = tasks.f(x[:, None])[:, 0]
    return F


def ei_readout(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = t.device
    pfn = load(t.pfn)[0].to(dev).eval()

    t0 = time.time()
    data, key = _data(cfg, pfn, dev)
    run.set_tags(data_hash=key)
    run.log_summary(**{M.TIME_PREFIX + "data_s": time.time() - t0})
    (st, tg), (st_te, tg_te) = data["train"], data["test"]
    run.log_summary(
        **{
            "data/n_train": float(len(st)),
            "data/n_test": float(len(st_te)),
            "data/mass_top_mean": float(tg.mass_top.mean()),
        }
    )

    expert = ActionExpert(t.trunk, pfn, t.d_expert, t.n_heads, t.n_mix, t.n_probe).to(dev)
    params = [p for p in expert.parameters() if p.requires_grad]
    run.log_summary(**{"model/trainable_params": float(sum(p.numel() for p in params))})
    opt = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t.lr, total_steps=t.steps, pct_start=0.05)
    g = torch.Generator().manual_seed(cfg.seed)
    val_idx = torch.randperm(len(st_te), generator=g)[: t.val_size].to(dev)
    t0 = time.time()
    for step in range(1, t.steps + 1):
        idx = torch.randint(0, len(st), (t.batch,), generator=g).to(dev)
        loss = _loss(expert, _batch(st, tg, idx))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % t.log_every == 0 or step == t.steps:
            run.log_metric(M.TRAIN_LOSS, float(loss), step)
        if step % t.val_every == 0 or step == t.steps:
            with torch.no_grad():
                vl = np.mean([float(_loss(expert, _batch(st_te, tg_te, i))) for i in val_idx.split(512)])
            run.log_metric(M.VAL_LOSS, vl, step)
    run.log_summary(**{M.TIME_PREFIX + "train_s": time.time() - t0, M.FINAL_VAL_LOSS: vl})
    expert.eval()

    # one-step quality on test states (tasks of a split never used for training)
    x_mode, x_samp = _predict(expert, st_te)
    r_mode = (ei_at(pfn, st_te, x_mode) / tg_te.max_ei).cpu().numpy()
    r_samp = (ei_at(pfn, st_te, x_samp) / tg_te.max_ei).cpu().numpy()
    n_ctx = st_te.mask.sum(1).cpu().numpy()
    n_traj = len(st_te) // 2
    ratio = pd.DataFrame(
        {
            "state": np.arange(len(st_te)),
            "source": np.where(np.arange(len(st_te)) < n_traj, "ei_trajectory", "random"),
            "n": n_ctx,
            "ratio_mode": r_mode,
            "ratio_sample": r_samp,
        }
    )
    run.log_table("ei_ratio", ratio)
    run.log_summary(
        **{
            M.EI_RATIO_MODE_MEDIAN: float(np.median(r_mode)),
            M.EI_RATIO_MODE_P10: float(np.quantile(r_mode, 0.1)),
            M.EI_RATIO_SAMPLE_MEDIAN: float(np.median(r_samp)),
            M.EI_RATIO_SAMPLE_P10: float(np.quantile(r_samp, 0.1)),
        }
    )
    # cost-matched reference: EI maximized over 256 random points
    cheap = ei_targets(pfn, st_te, t.data.lam, 1, CandidateConfig(256, 0, (0.01,)), seed=99)
    run.log_table(
        "cheap", pd.DataFrame({"state": np.arange(len(st_te)), "ratio": (cheap.max_ei / tg_te.max_ei).cpu().numpy()})
    )

    # closed loop on fresh tasks: the expert (mode, sample) vs EI with the full candidate optimizer
    c = t.closed_loop
    seeds = [task_seed(c.split, t.d, i) for i in range(c.n_tasks)]
    tasks = draw_tasks(t.d, seeds).to(dev)
    budget = Budget(c.n_init_per_d * t.d, c.T_per_d * t.d, c.T_per_d * t.d)
    engine = Engine(
        pfn,
        candidates=CandidateConfig(
            **{**OmegaConf.to_container(c.candidates), "local_sigmas": tuple(c.candidates.local_sigmas)}
        ),
    )
    Fs = {"ei": run_static(engine, tasks, "ei", budget, dev)["F"]}
    for how in ("mode", "sample"):
        Fs[f"expert_{how}"] = _closed_loop(expert, tasks, budget, how, dev)
    for F in Fs.values():
        tasks.lower_fstar(F)
    cl = pd.concat(
        [
            pd.DataFrame(
                {
                    "task": np.arange(tasks.B),
                    "seed": seeds,
                    "method": k,
                    "best_f": F.min(1).values.cpu().numpy(),
                    "fstar": tasks.fstar.cpu().numpy(),
                }
            )
            for k, F in Fs.items()
        ],
        ignore_index=True,
    )
    run.log_table("closed_loop", cl)
    for k, F in Fs.items():
        lr = torch.log10((F.min(1).values - tasks.fstar).clamp(min=0) + 1e-4).mean()
        run.log_summary(**{f"{M.FINAL_LOG_REGRET}/{k}": float(lr)})
    if t.save_checkpoint:
        torch.save({k: v for k, v in expert.state_dict().items() if not k.startswith("pfn.")}, run.dir / "expert.pt")
    return float(np.median(r_mode))

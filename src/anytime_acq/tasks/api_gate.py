"""The model-based gate (experimental/api_gate.py; research_log/decisions/0003 §8.3, §8.4 Stages 1a and 1b).

One member = one d, in two parts:
1. *states* (Stage 1a): at the first `states.n_states` labeled states of a round-1 label set (an artifact of an
   api run, pinned by sha256), belief continuations score the same labeled candidates; the candidate with the
   best model-based advantage is selected without the task and scored with its ground-truth label;
2. *policy* (Stage 1b): the model-based rollout policy against EI (pi_0) on fresh tasks, paired (CRN).

Tables: `gate_states` (one row per labeled candidate: state, seed, t, j, valid, model advantage and se,
ground-truth label), `test` (final normalized regret and log regret per task and arm), `best` (best f per
step), `decisions` (per task and step: deviated, model advantage of the queried candidate), `fstar`.
"""

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load

from ..bo.engine import Engine
from ..bo.runner import initial_design
from ..bo.tasks import draw_tasks, task_seed
from ..experimental.api import CandidateSpec, task_median
from ..experimental.api_gate import belief_values, model_advantage, run_model_based
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .api import _arm_means, _dict, final_table
from .bo import _candidates, best_table


def runs_root(cfg: DictConfig) -> Path:
    """The runs root of the machine (members run from the sweep's frozen worktree, so not repo_root/runs)."""
    sd = cfg.tracking.get("sweep_dir")
    return Path(sd).parents[1] if sd else Path(cfg.paths.runs)


def load_labels(cfg: DictConfig) -> dict[str, torch.Tensor]:
    path = runs_root(cfg) / str(cfg.task.labels.path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != str(cfg.task.labels.sha256):
        raise ValueError(f"{path}: sha256 {digest} != pinned {cfg.task.labels.sha256}")
    return torch.load(path)


def state_part(cfg: DictConfig, engine: Engine, run: Run) -> pd.DataFrame:
    t = cfg.task
    n_init, T = int(t.n_init_per_d * t.d), int(t.T_per_d * t.d)
    L = load_labels(cfg)
    S = min(int(t.states.n_states), len(L["seed"]))
    M = int(t.states.n_cont)
    dev = t.device
    rows_all = []
    for tb in L["branch_t"][:S].unique().tolist():
        idx = (L["branch_t"][:S] == tb).nonzero().squeeze(1)
        X, F = L["X"][idx, : n_init + tb].to(dev), L["F"][idx, : n_init + tb].to(dev)
        lab = L["lab"][idx].to(dev)
        xq = L["C"][idx].to(dev).gather(1, lab[..., None].expand(-1, -1, X.shape[-1]))
        rows = idx.to(dev)
        rep = torch.full((len(idx),), int(t.states.rep_base), device=dev, dtype=torch.long)
        V = belief_values(engine, X, F, rows, xq, tb, T, M, rep, S)
        a_m, se_m = (v.cpu() for v in model_advantage(V))
        nl = lab.shape[1]
        rows_all.append(
            pd.DataFrame(
                {
                    "state": np.repeat(idx.numpy(), nl),
                    "seed": np.repeat(L["seed"][idx].numpy(), nl),
                    "t": tb,
                    "j": np.tile(np.arange(nl), len(idx)),
                    "valid": L["lab_valid"][idx].numpy().ravel(),
                    "adv_model": a_m.numpy().ravel(),
                    "se_model": se_m.numpy().ravel(),
                    "adv_true": L["adv"][idx].numpy().ravel(),
                }
            )
        )
    tab = pd.concat(rows_all, ignore_index=True)
    sel = selected_gain(tab, 0.0)
    run.log_summary(
        **{
            "states/n": float(S),
            "states/gain_tau0": float(sel.adv_true.mean()),
            "states/deviation_tau0": float((sel.j > 0).mean()),
        }
    )
    return tab


def selected_gain(tab: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Per state: the candidate with the best model advantage among valid ones if that advantage exceeds
    tau * se_model (else x0), and its ground-truth label."""
    v = tab.loc[tab.valid].copy()
    v["score"] = np.where(v.j == 0, 0.0, v.adv_model - tau * v.se_model)
    best = v.loc[v.groupby("state").score.idxmax()]
    best = best.assign(j=np.where(best.score > 0, best.j, 0))
    x0 = v.loc[v.j == 0].set_index("state").adv_true
    best["adv_true"] = np.where(best.j == 0, best.state.map(x0), best.adv_true)
    return best[["state", "t", "j", "adv_true"]]


def api_gate(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(t.tf32)
    torch.backends.cudnn.allow_tf32 = bool(t.tf32)
    pfn = load(t.pfn)[0].to(t.device).eval()
    engine = Engine(pfn, None, _candidates(t.candidates), base_seed=cfg.seed, chunk=int(t.chunk))
    ac = t.api_candidates
    spec = CandidateSpec(
        int(ac.n_global), int(ac.n_top), int(ac.n_local_per_point), tuple(float(s) for s in ac.local_sigmas)
    )
    d, n_init, T = int(t.d), int(t.n_init_per_d * t.d), int(t.T_per_d * t.d)

    t0 = time.time()
    run.log_table("gate_states", state_part(cfg, engine, run))
    run.log_summary(**{"time/states_s": time.time() - t0})

    t0 = time.time()
    p = t.policy
    seeds = [task_seed(int(p.split), d, int(p.task_offset) + i) for i in range(int(p.n_tasks))]
    tasks = draw_tasks(d, seeds, **_dict(t.task_draw)).to(t.device)
    X0 = initial_design(tasks, n_init).to(t.device)
    out = run_model_based(engine, spec, tasks, X0, T, int(p.n_labeled), int(p.n_cont), int(cfg.seed), int(p.rep_offset))
    Fs = {"ei": out["F"][:, 0], "mb": out["F"][:, 1]}
    for F in Fs.values():
        tasks.lower_fstar(F)
    fmed = task_median(tasks)
    test = final_table(Fs, tasks, fmed, seeds, float(t.eps))
    run.log_table("test", test)
    run.log_table("best", pd.concat([best_table(tasks, a, F, n_init) for a, F in Fs.items()], ignore_index=True))
    B = len(seeds)
    run.log_table(
        "decisions",
        pd.DataFrame(
            {
                "task": np.repeat(np.arange(B), T),
                "seed": np.repeat(np.asarray(seeds), T),
                "step": np.tile(np.arange(T), B),
                "deviated": out["deviated"].cpu().numpy().ravel(),
                "adv_model": out["adv_model"].cpu().numpy().ravel(),
            }
        ),
    )
    run.log_table(
        "fstar",
        pd.DataFrame(
            {"task": np.arange(B), "seed": seeds, "fstar": tasks.fstar.cpu().numpy(), "fmed": fmed.cpu().numpy()}
        ),
    )
    m = _arm_means(test, "norm_regret")
    run.log_summary(**{f"test/norm_regret/{a}": v for a, v in m.items()})
    run.log_summary(**{f"test/log_regret/{a}": v for a, v in _arm_means(test, "log_regret").items()})
    run.log_summary(
        **{"test/deviation_rate/mb": float(out["deviated"].float().mean()), "time/policy_s": time.time() - t0}
    )
    return m["mb"] - m["ei"]

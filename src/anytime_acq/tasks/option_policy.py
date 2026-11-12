"""Can a policy that reads only the decision-time context choose options that beat the base policy?
(experimental/option_policy.py, analysis/option_policy.py; pre-registration:
notebooks/13_2026-11-12_option_policy.ipynb; research_log/decisions/0008 §4 step 3.)

One member = one d and one base arm.
- Labels (`collect`): per task, the base arm under seed 0 for the whole run (the prefix). At every decision step
  t in `steps`, from the prefix's data: the base and every option "arm a for k steps (or the rest), then the
  base to T", under every candidate-set seed. Deterministic arms on noise-free tasks: each lane is an exact
  outcome. The advantage of an option is log10 R_T(base) - log10 R_T(option) under the same seed.
- Context (`collect`): at every (task, t), features of the data, of the PFN's belief, and the PFN's state.
- Policy (`evaluate`): Q(state, option) regressed on the seed-mean advantage, cross-fitted over tasks; the
  one-deviation policy commits at the first decision step where max_o Q > c, with c chosen inside the training
  folds (c = inf, the base, is allowed). Its value on a held-out task is the exact label of its decision.

Tables: `summary` (gains and contrasts with CIs), `per_task` (decisions and values per variant), `usage` (which
step and option each variant chose), `fit` (held-out fit of Q per option), `options` (mean advantage per step and
option). Artifact: `labels.pt` (raw regrets and features). Verdict: `analysis/option_policy.py::policy_verdict`.
"""

import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load

from ..analysis.option_policy import (
    choose_threshold,
    decision_value,
    first_trigger,
    oracle_crossfit,
    policy_verdict,
    shuffled_value,
    static_choice,
    summarize,
)
from ..bo.engine import Engine
from ..bo.runner import initial_design
from ..bo.tasks import draw_tasks, task_seed
from ..experimental.api import f_lanes, rollout
from ..experimental.option_policy import FitSpec, belief_features, crossfit_q, data_features, pca, pfn_state
from ..experimental.options import option_policy as arm_policy
from ..experimental.probes import spearman
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .api import _dict
from .bo import _candidates

BIN = 1e-3


def option_list(t: DictConfig) -> list[tuple[str, str, int]]:
    """(name, arm, k) of every option; k = T means "for the rest of the run"."""
    T = int(t.T_per_d * t.d)
    out = [(f"{a}_k{int(k)}", str(a), int(k)) for a in t.arms if str(a) != str(t.base) for k in t.ks]
    if bool(t.rest):
        out += [(f"{a}_rest", str(a), T) for a in t.arms if str(a) != str(t.base)]
    return out


def collect(cfg: DictConfig, engine: Engine, run: Run | None = None) -> dict:
    """With `run`: after every task chunk, the labels so far go to the artifact `labels_partial.pt` and the
    progress to the run's metrics, so a long collection can be evaluated before it ends (`evaluate` on it)."""
    t = cfg.task
    dev, d = t.device, int(t.d)
    n_init, T = int(t.n_init_per_d * d), int(t.T_per_d * d)
    steps = [int(s) for s in t.steps]
    seeds = [int(s) for s in t.seeds]
    base = str(t.base)
    opts = [("base", base, 0), *option_list(t)]
    N, S, n_opt = int(t.tasks.n_tasks), len(steps), len(opts)
    best = torch.empty(N, S, n_opt, len(seeds))
    state_best = torch.empty(N, S)
    fstar = torch.empty(N)
    fd, fb, fz, names = [], [], [], {}
    all_seeds = [task_seed(int(t.tasks.split), d, int(t.tasks.task_offset) + i) for i in range(N)]

    def pack(n: int) -> dict:
        return {
            "regret": (best[:n] - fstar[:n, None, None, None]).clamp(min=0),  # [n, S, 1 + options, seeds]; 0 = base
            "state_regret": (state_best[:n] - fstar[:n, None]).clamp(min=0),  # [n, S], privileged, diagnostic
            "data": torch.cat(fd),
            "belief": torch.cat(fb),
            "z": torch.cat(fz).half(),
            "names": names,
            "options": [o[0] for o in opts[1:]],
            "steps": steps,
            "task_seeds": all_seeds[:n],
        }

    for c0 in range(0, N, int(t.task_chunk)):
        ts = all_seeds[c0 : c0 + int(t.task_chunk)]
        with torch.enable_grad():  # draw_tasks refines f* by gradient descent
            tasks = draw_tasks(d, ts, **_dict(t.task_draw)).to(dev)
        B = tasks.B
        rows = torch.arange(B, device=dev)
        X0 = initial_design(tasks, n_init).to(dev)
        F0 = f_lanes(tasks, X0.reshape(-1, d), rows.repeat_interleave(n_init)).view(B, n_init)
        Xp, Fp = rollout(
            tasks, X0, F0, rows, rows * 0, 0, T, arm_policy(engine, [base] * B, torch.full((B,), T), seeds[0], base)
        )
        low = Fp.min(1).values
        cb = torch.empty(B, S, n_opt, len(seeds), device=dev)
        cd, cbf, cz = [], [], []
        for j, tb in enumerate(steps):
            n = n_init + tb
            gi = rows.repeat(n_opt)
            arms = [a for _, a, _ in opts for _ in range(B)]
            until = torch.tensor([tb + k for _, _, k in opts for _ in range(B)], device=dev)
            for si, s in enumerate(seeds):
                _, F = rollout(
                    tasks, Xp[gi, :n], Fp[gi, :n], gi, gi * 0, tb, T, arm_policy(engine, arms, until, s, base)
                )
                cb[:, j, :, si] = F.min(1).values.view(n_opt, B).T
            f1, names["data"] = data_features(Xp[:, :n], Fp[:, :n], tb, T, n_init)
            f2, names["belief"] = belief_features(engine, Xp[:, :n], Fp[:, :n], seeds[0])
            cd.append(f1)
            cbf.append(f2)
            cz.append(pfn_state(engine, Xp[:, :n], Fp[:, :n]))
            state_best[c0 : c0 + B, j] = Fp[:, :n].min(1).values.cpu()
        tasks.lower_fstar(torch.minimum(low, cb.flatten(1).min(1).values)[:, None])
        best[c0 : c0 + B] = cb.cpu()
        fstar[c0 : c0 + B] = tasks.fstar.cpu()
        fd.append(torch.stack(cd, 1).cpu())
        fb.append(torch.stack(cbf, 1).cpu())
        fz.append(torch.stack(cz, 1).cpu())
        if run is not None and c0 + B < N:
            run.artifacts.mkdir(parents=True, exist_ok=True)
            torch.save(pack(c0 + B), run.artifacts / "labels_partial.pt")
            run.log_metric("collect/tasks_done", float(c0 + B), step=c0 + B)
    return pack(N)


def advantages(regret: torch.Tensor, eps: float) -> torch.Tensor:
    """[N, S, options, seeds]: log10(R_base + eps) - log10(R_option + eps)."""
    lr = torch.log10(regret.double() + eps)
    return (lr[:, :, :1] - lr[:, :, 1:]).float()


def evaluate(L: dict, cfg: DictConfig, run: Run) -> pd.DataFrame:
    t = cfg.task
    dev = t.device
    A = advantages(L["regret"], float(t.eps))  # [N, S, n_opt, seeds]
    N, S, n_opt, _ = A.shape
    names: list[str] = L["options"]
    target = A.mean(-1)
    A0, A1 = A[..., 0].numpy(), A[..., 1].numpy()
    K = int(t.n_folds)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(int(cfg.seed) + 5))
    fold_task = torch.empty(N, dtype=torch.long)
    fold_task[perm] = torch.arange(N) % K
    task_row = torch.arange(N).repeat_interleave(S).to(dev)
    fold_row = fold_task.repeat_interleave(S).to(dev)
    Y = target.reshape(N * S, n_opt).to(dev)
    data, belief = L["data"].reshape(N * S, -1), L["belief"].reshape(N * S, -1)
    zp = pca(L["z"].float().reshape(N * S, -1), int(t.z_pca))
    sets = {
        "time": data[:, :1],
        "data": data,
        "belief": torch.cat([data, belief], 1),
        "pfn": torch.cat([data, belief, zp], 1),
    }
    spec = FitSpec(**_dict(t.fit))
    models = [(fs, "mlp") for fs in t.feature_sets] + [("belief", "ridge")]
    portfolios = {
        str(k): [names.index(o) for o in names if o.rsplit("_", 1)[0] in list(v)] for k, v in t.portfolios.items()
    }
    per = pd.DataFrame({"task": np.arange(N), "task_seed": L["task_seeds"], "fold": fold_task.numpy()})
    tgt = target.numpy()
    R = L["regret"].numpy()
    fit_rows, usage_rows, variants = [], [], []
    static = {}
    for pf, idx in portfolios.items():
        v0, v1 = np.zeros(N), np.zeros(N)
        for k in range(K):
            tr, te = (fold_task != k).numpy(), (fold_task == k).numpy()
            j, o = static_choice(tgt[tr][:, :, idx])
            jj, oo = np.full(te.sum(), j), np.full(te.sum(), o)
            v0[te], v1[te] = decision_value(A0[te][:, :, idx], jj, oo), decision_value(A1[te][:, :, idx], jj, oo)
            usage_rows.append(
                {
                    "variant": f"{pf}/static",
                    "fold": k,
                    "step": j,
                    "option": names[idx[o]] if j >= 0 else "base",
                    "n": int(te.sum()),
                }
            )
        static[pf] = (v0, v1)
        per[f"{pf}:oracle"] = oracle_crossfit(A0[:, :, idx], A1[:, :, idx])
    for fs, learner in models:
        Q_out, Q_in = crossfit_q(
            sets[fs].to(dev).float(), Y, task_row, fold_row, learner, int(t.n_inner), int(cfg.seed), spec
        )
        Qo = Q_out.view(N, S, n_opt).cpu().numpy()
        Qi = Q_in.view(K, N, S, n_opt).cpu().numpy()
        for oi, name in enumerate(names):
            fit_rows.append(
                {
                    "features": fs,
                    "learner": learner,
                    "option": name,
                    "spearman": spearman(Qo[:, :, oi].ravel(), tgt[:, :, oi].ravel()),
                    "r2": 1 - float(((Qo[:, :, oi] - tgt[:, :, oi]) ** 2).mean() / tgt[:, :, oi].var()),
                    "seed_corr": float(np.corrcoef(A0[:, :, oi].ravel(), A1[:, :, oi].ravel())[0, 1]),
                }
            )
        for pf, idx in portfolios.items():
            v = f"{pf}/{fs}/{learner}"
            variants.append(v)
            j, o, c = np.zeros(N, dtype=int), np.zeros(N, dtype=int), np.zeros(N)
            sh0, sh1 = np.zeros(N), np.zeros(N)
            for k in range(K):
                tr, te = (fold_task != k).numpy(), (fold_task == k).numpy()
                c[te] = choose_threshold(Qi[k][tr][:, :, idx], tgt[tr][:, :, idx])
                j[te], o[te] = first_trigger(Qo[te][:, :, idx], float(c[te][0]))
                sh0[te] = shuffled_value(A0[te][:, :, idx], j[te], o[te])
                sh1[te] = shuffled_value(A1[te][:, :, idx], j[te], o[te])
            og = np.asarray(idx)[o]  # option index among all options
            jc = np.maximum(j, 0)
            cols = {
                f"{v}:j": j,
                f"{v}:option": np.where(j >= 0, np.asarray(names)[og], "base"),
                f"{v}:c": c,
                f"{v}:v0": decision_value(A0[:, :, idx], j, o),
                f"{v}:v1": decision_value(A1[:, :, idx], j, o),
                f"{v}:static0": static[pf][0],
                f"{v}:static1": static[pf][1],
                f"{v}:shuffled0": sh0,
                f"{v}:shuffled1": sh1,
            }
            for s in (0, 1):  # raw regrets (ECDF units) of the base and of the decision, for other eps
                cols[f"{v}:regret_base{s}"] = R[np.arange(N), jc, 0, s]
                cols[f"{v}:regret{s}"] = np.where(j >= 0, R[np.arange(N), jc, og + 1, s], R[np.arange(N), jc, 0, s])
            per = pd.concat([per, pd.DataFrame(cols)], axis=1)
            st = np.where(j >= 0, np.asarray(L["steps"])[jc], -1).tolist()
            for (step, op), n in Counter(zip(st, cols[f"{v}:option"].tolist(), strict=True)).items():
                usage_rows.append({"variant": v, "fold": -1, "step": step, "option": op, "n": n})
    summary = summarize(per, variants, int(t.n_boot))
    run.log_table("summary", summary)
    run.log_table("per_task", per)
    run.log_table("usage", pd.DataFrame(usage_rows))
    run.log_table("fit", pd.DataFrame(fit_rows))
    om = [
        {
            "step": L["steps"][j],
            "option": names[o],
            "adv0": float(A0[:, j, o].mean()),
            "adv1": float(A1[:, j, o].mean()),
            "share_pos": float((tgt[:, j, o] > 0).mean()),
            "share_neg": float((tgt[:, j, o] < 0).mean()),
        }
        for j in range(S)
        for o in range(n_opt)
    ]
    run.log_table("options", pd.DataFrame(om))
    return summary


def option_policy(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(t.tf32)
    torch.backends.cudnn.allow_tf32 = bool(t.tf32)
    pfn = load(t.pfn)[0].to(t.device).eval()
    engine = Engine(pfn, None, _candidates(t.candidates), base_seed=cfg.seed, chunk=int(t.chunk))
    t0 = time.time()
    L = collect(cfg, engine, run)
    run.log_summary(**{"time/collect_s": time.time() - t0})
    run.artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(L, run.artifacts / "labels.pt")
    t0 = time.time()
    summary = evaluate(L, cfg, run)
    run.log_summary(**{"time/evaluate_s": time.time() - t0})
    s0 = summary.loc[summary.seed == 0]
    for variant, contrast, mean in zip(s0.variant, s0.contrast, s0["mean"], strict=True):
        run.log_summary(**{f"gain/{variant}/{contrast}": float(mean)})
    verdict = policy_verdict(summary, str(t.primary), float(t.worthwhile_log10))
    run.log_summary(verdict_go=float(verdict == "GO"), verdict_nogo=float(verdict == "NO-GO"))
    primary = (s0.variant == str(t.primary)) & (s0.contrast == "vs_static")
    return float(s0.loc[primary, "mean"].to_numpy()[0])

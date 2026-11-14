"""Amortized policy improvement, round 1 (experimental/api.py; research_log/decisions/0003).

One member = one d, in five steps:
1. labels on `label.split` (training instances): behaviour rollouts, one branch state each, branch
   continuations with pi_0 (EI) on the true task;
2. features of every labeled state (frozen PFN), split by instance into train / validation;
3. the value head, trained by Gaussian NLL with the label's standard error, early-stopped on validation
   NLL; and the null head: the same on labels permuted across the candidates of each state (destroys
   the candidate-label link, keeps the label distribution);
4. calibration on `calib.split`: pi_1 for each c in `calib.cs` paired against pi_0 (EI); c* = the c
   with the lowest mean final normalized regret (ties to the larger c) -- chosen before any test run;
5. test on `test.split`: EI, pi_1 at every c (c* is the pre-registered one), the null head at c*.

Tables: `labels` (one row per labeled candidate: state, t, behaviour, source, label, se, R per
continuation, split; prediction and sigma for validation states), `calib` and `test` (final normalized
regret and log regret per task and arm), `best` (best f per step, as in the bo pipeline), `decisions`
(per test task, arm and step: deviated, chosen source, A_hat), `fstar` (f*, f_med per task and split).
"""

import math
import time
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pfn_repro.checkpoints import load

from ..bo.engine import Engine
from ..bo.runner import initial_design
from ..bo.tasks import Tasks, draw_tasks, task_seed
from ..experimental.api import (
    SOURCES,
    CandidateSpec,
    LabelSpec,
    ValueHead,
    advantages,
    features,
    generate_labels,
    improved_pick,
    label_nll,
    normalized_regret,
    run_improved,
    task_median,
)
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .bo import _candidates, best_table

INF = float("inf")


def _dict(c: Any) -> dict[str, Any]:
    """A config node as a plain dict with str keys."""
    return cast(dict[str, Any], OmegaConf.to_container(c))


def _arm(c: float) -> str:
    return "ei" if math.isinf(c) else f"api_c{c:g}"


def _per_task_uniforms(seeds: list[int], n: int) -> torch.Tensor:
    """[B, n] uniforms from each task's own seed (independent of chunking and of other tasks)."""
    return torch.stack([torch.rand(n, generator=torch.Generator().manual_seed(int(s) % (2**31) + 17)) for s in seeds])


def make_labels(cfg: DictConfig, engine: Engine, spec: CandidateSpec, run: Run) -> dict[str, torch.Tensor]:
    t, lab = cfg.task, cfg.task.label
    d, n_init, T = int(t.d), int(t.n_init_per_d * t.d), int(t.T_per_d * t.d)
    lspec = LabelSpec(int(lab.n_labeled), int(lab.n_cont), dict(_dict(lab.behaviour)))
    arms, probs = list(lspec.behaviour), torch.tensor(list(lspec.behaviour.values()))
    cum = probs.cumsum(0) / probs.sum()
    parts = []
    for s in range(0, int(lab.n_tasks), int(lab.chunk_tasks)):
        t0 = time.time()
        seeds = [
            task_seed(int(lab.split), d, int(lab.task_offset) + i)
            for i in range(s, min(s + int(lab.chunk_tasks), int(lab.n_tasks)))
        ]
        tasks = draw_tasks(d, seeds, **_dict(lab.task_draw)).to(t.device)
        u = _per_task_uniforms(seeds, 2)
        behaviour = [arms[int((cum <= ui).sum().clamp(max=len(arms) - 1))] for ui in u[:, 0]]
        branch_t = (u[:, 1] * T).long().clamp(max=T - 1).to(t.device)
        X0 = initial_design(tasks, n_init).to(t.device)
        out = generate_labels(
            engine, spec, lspec, tasks, n_init, T, X0, behaviour, branch_t, label_u_seed=int(cfg.seed)
        )
        out = {k: v.cpu() for k, v in out.items()}
        out["seed"] = torch.tensor(seeds)
        out["behaviour"] = torch.tensor([arms.index(b) for b in behaviour])
        parts.append(out)
        run.log_metric("labels/chunk_s", time.time() - t0, step=s)
        del tasks
    L = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    L["adv"], L["se"] = advantages(L["R"], L["lab_valid"])
    L["behaviour_names"] = arms  # type: ignore[assignment]
    return L


@torch.no_grad()
def state_features(cfg: DictConfig, engine: Engine, spec: CandidateSpec, L: dict) -> list[dict[str, torch.Tensor]]:
    """Features of every labeled state, grouped by branch step t (equal context length); on device."""
    dev = cfg.task.device
    n_init, T = int(cfg.task.n_init_per_d * cfg.task.d), int(cfg.task.T_per_d * cfg.task.d)
    groups = []
    for tb in L["branch_t"].unique().tolist():
        idx = (L["branch_t"] == tb).nonzero().squeeze(1)
        Xs = L["X"][idx, : n_init + tb].to(dev)
        Fs = L["F"][idx, : n_init + tb].to(dev)
        C, valid = L["C"][idx].to(dev), L["valid"][idx].to(dev)
        ft = features(engine, spec, Xs, Fs, C, valid, tb, T)
        ft.pop("ei")
        ft.update(
            valid=valid,
            idx=idx.to(dev),
            lab=L["lab"][idx].to(dev),
            lab_valid=L["lab_valid"][idx].to(dev),
            adv=L["adv"][idx].to(dev),
            se=L["se"][idx].to(dev),
        )
        groups.append(ft)
    return groups


def _batches(groups: list[dict], mask_of: dict[int, torch.Tensor], bs: int, gen: torch.Generator) -> list:
    """(group, rows) batches over the states selected by mask_of[group], shuffled."""
    out = []
    for gi in range(len(groups)):
        rows = mask_of[gi].nonzero().squeeze(1).cpu()
        rows = rows[torch.randperm(len(rows), generator=gen)]
        out += [(gi, rows[s : s + bs]) for s in range(0, len(rows), bs)]
    return [out[i] for i in torch.randperm(len(out), generator=gen).tolist()]


def _loss(
    head: ValueHead, g: dict, rows: torch.Tensor, labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    adv, se, lab_valid = (x[rows] for x in labels)
    A, sig = head(g["z"][rows], g["cand"][rows], g["glob"][rows], g["valid"][rows])
    lab = g["lab"][rows][:, 1:]
    A_l, s_l = A.gather(1, lab), sig.gather(1, lab)
    m = lab_valid[:, 1:]
    return label_nll(A_l[m], s_l[m], adv[:, 1:][m], se[:, 1:][m])


def train_head(
    cfg: DictConfig, groups: list[dict], is_val: dict[int, torch.Tensor], permute: bool, run: Run, prefix: str
) -> ValueHead:
    """AdamW + cosine decay, early stopping on validation NLL; `permute`: the null head's labels."""
    h, tr = cfg.task.head, cfg.task.train
    dev = cfg.task.device
    torch.manual_seed(int(cfg.seed) + (1 if permute else 0))
    head = ValueHead(groups[0]["z"].shape[-1], int(h.width), int(h.layers), int(h.heads)).to(dev)
    tr_mask = {gi: ~is_val[gi] for gi in range(len(groups))}
    head.set_normalization(
        torch.cat([g["cand"][tr_mask[gi]] for gi, g in enumerate(groups)]),
        torch.cat([g["glob"][tr_mask[gi]] for gi, g in enumerate(groups)]),
        torch.cat([g["valid"][tr_mask[gi]] for gi, g in enumerate(groups)]),
    )
    gen = torch.Generator().manual_seed(int(cfg.seed) * 31 + int(permute))
    labels = []
    for g in groups:
        adv, se, lv = g["adv"], g["se"], g["lab_valid"]
        if permute:  # permute labels across the non-base candidates of each state
            perm = torch.argsort(torch.rand(adv.shape[0], adv.shape[1] - 1, generator=gen), 1).to(dev) + 1
            perm = torch.cat([torch.zeros_like(perm[:, :1]), perm], 1)
            adv, se, lv = adv.gather(1, perm), se.gather(1, perm), lv.gather(1, perm)
        labels.append((adv, se, lv))
    n_steps = int(tr.epochs) * len(_batches(groups, tr_mask, int(tr.batch), torch.Generator().manual_seed(0)))
    opt = torch.optim.AdamW(head.parameters(), lr=float(tr.lr), weight_decay=float(tr.wd))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(n_steps, 1))
    best, best_state, bad = INF, None, 0
    for ep in range(int(tr.epochs)):
        head.train()
        tot, n = 0.0, 0
        for gi, rows in _batches(groups, tr_mask, int(tr.batch), gen):
            rows = rows.to(dev)
            loss = _loss(head, groups[gi], rows, labels[gi])
            if loss.numel() == 0:
                continue
            opt.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + float(loss.detach().sum()), n + loss.numel()
        head.eval()
        with torch.no_grad():
            vl = [_loss(head, groups[gi], is_val[gi].nonzero().squeeze(1), labels[gi]) for gi in range(len(groups))]
            v = torch.cat(vl)
            val = float(v.mean())
        run.log_metrics({f"{prefix}/train_nll": tot / max(n, 1), f"{prefix}/val_nll": val}, step=ep)
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: x.detach().clone() for k, x in head.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tr.patience):
                break
    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    run.log_summary(**{f"{prefix}/best_val_nll": best, f"{prefix}/epochs": float(ep + 1)})
    return head


@torch.no_grad()
def predict_val(head: ValueHead, groups: list[dict], is_val: dict[int, torch.Tensor]) -> dict[int, tuple]:
    """Prediction and sigma at the labeled candidates of validation states: {state index: (A, sigma)}."""
    out = {}
    for gi, g in enumerate(groups):
        rows = is_val[gi].nonzero().squeeze(1)
        if len(rows) == 0:
            continue
        A, sig = head(g["z"][rows], g["cand"][rows], g["glob"][rows], g["valid"][rows])
        lab = g["lab"][rows]
        for r, si in enumerate(g["idx"][rows].tolist()):
            out[si] = (A[r].gather(0, lab[r]).cpu(), sig[r].gather(0, lab[r]).cpu())
    return out


def labels_table(L: dict, spec: CandidateSpec, is_val_state: torch.Tensor, pred: dict) -> pd.DataFrame:
    B, nl, M = L["R"].shape
    src = spec.source_ids()[L["lab"]]  # [B, nl]
    df = pd.DataFrame(
        {
            "state": np.repeat(np.arange(B), nl),
            "seed": np.repeat(L["seed"].numpy(), nl),
            "t": np.repeat(L["branch_t"].numpy(), nl),
            "behaviour": np.repeat(np.asarray(L["behaviour_names"])[L["behaviour"].numpy()], nl),
            "j": np.tile(np.arange(nl), B),
            "cand": L["lab"].numpy().ravel(),
            "source": np.asarray(SOURCES)[src.numpy().ravel()],
            "valid": L["lab_valid"].numpy().ravel(),
            "adv": L["adv"].numpy().ravel(),
            "se": L["se"].numpy().ravel(),
            "split": np.repeat(np.where(is_val_state.numpy(), "val", "train"), nl),
        }
    )
    for k in range(M):
        df[f"R{k}"] = L["R"][..., k].numpy().ravel()
    A = np.full((B, nl), np.nan)
    S = np.full((B, nl), np.nan)
    for si, (a, s) in pred.items():
        A[si], S[si] = a.numpy(), s.numpy()
    df["A_hat"], df["sigma"] = A.ravel(), S.ravel()
    return df


def evaluate(
    cfg: DictConfig,
    engine: Engine,
    spec: CandidateSpec,
    heads: dict[str, tuple[ValueHead, list[float]]],
    split: int,
    n_tasks: int,
    offset: int,
) -> tuple[dict[str, torch.Tensor], Tasks, torch.Tensor, list[int], dict]:
    """Run every head at its cs on the same tasks (EI = c = inf of the first head). -> F per arm, tasks,
    f_med, seeds, decisions per arm."""
    t = cfg.task
    d, n_init, T = int(t.d), int(t.n_init_per_d * t.d), int(t.T_per_d * t.d)
    seeds = [task_seed(split, d, offset + i) for i in range(n_tasks)]
    tasks = draw_tasks(d, seeds, **_dict(t.task_draw)).to(t.device)
    X0 = initial_design(tasks, n_init).to(t.device)
    Fs, dec = {}, {}
    for name, (head, cs) in heads.items():
        out = run_improved(engine, spec, head, tasks, X0, T, cs)
        for k, c in enumerate(cs):
            arm = _arm(c) if name == "api" else f"{name}_c{c:g}"
            Fs[arm] = out["F"][:, k]
            dec[arm] = {key: out[key][:, k].cpu() for key in ("deviated", "source", "A_hat")}
    for F in Fs.values():
        tasks.lower_fstar(F)
    return Fs, tasks, task_median(tasks), seeds, dec


def _arm_means(fin: pd.DataFrame, col: str) -> dict[str, float]:
    return {str(a): float(g[col].to_numpy().mean()) for a, g in fin.groupby("arm")}


def final_table(Fs: dict, tasks: Tasks, fmed: torch.Tensor, seeds: list[int], eps: float) -> pd.DataFrame:
    rows = []
    for arm, F in Fs.items():
        R = normalized_regret(F, tasks.fstar, fmed).cpu().numpy()
        reg = (F.min(1).values - tasks.fstar).clamp(min=0).cpu().numpy()
        rows.append(
            pd.DataFrame(
                {
                    "task": np.arange(len(seeds)),
                    "seed": seeds,
                    "arm": arm,
                    "norm_regret": R,
                    "regret": reg,
                    "log_regret": np.log10(reg + eps),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def api(cfg: DictConfig, run: Run) -> float:
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

    t0 = time.time()
    L = make_labels(cfg, engine, spec, run)
    run.log_summary(**{"time/labels_s": time.time() - t0})
    run.artifacts.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in L.items() if isinstance(v, torch.Tensor)}, run.artifacts / "labels.pt")
    B = len(L["seed"])
    is_val_state = _per_task_uniforms(L["seed"].tolist(), 3)[:, 2] < float(t.label.val_frac)
    adv_ok = L["adv"][:, 1:][L["lab_valid"][:, 1:]]
    run.log_summary(
        **{
            "labels/n_states": float(B),
            "labels/mean_adv": float(adv_ok.mean()),
            "labels/frac_positive": float((adv_ok > 0).float().mean()),
            "labels/max_adv_mean": float(L["adv"].max(1).values.mean()),  # in-sample best-of-n (optimistic)
        }
    )

    t0 = time.time()
    groups = state_features(cfg, engine, spec, L)
    is_val = {gi: is_val_state.to(t.device)[g["idx"]] for gi, g in enumerate(groups)}
    head = train_head(cfg, groups, is_val, permute=False, run=run, prefix="head")
    null = train_head(cfg, groups, is_val, permute=True, run=run, prefix="null")
    run.log_summary(**{"time/train_s": time.time() - t0})
    pred = predict_val(head, groups, is_val)
    run.log_table("labels", labels_table(L, spec, is_val_state, pred))
    torch.save({"head": head.state_dict(), "null": null.state_dict()}, run.artifacts / "heads.pt")
    del groups

    # calibration of c (0003 §5.4), never on the test split
    t0 = time.time()
    cs = [float(c) for c in t.calib.cs]
    c = t.calib
    Fs, tasks, fmed, seeds, _ = evaluate(
        cfg, engine, spec, {"api": (head, [*cs, INF])}, int(c.split), int(c.n_tasks), int(c.task_offset)
    )
    cal = final_table(Fs, tasks, fmed, seeds, float(t.eps))
    run.log_table("calib", cal)
    means = _arm_means(cal, "norm_regret")
    c_star = min(cs, key=lambda x: (round(float(means[_arm(x)]), 9), -x))
    run.log_summary(**{"calib/c_star": c_star, "time/calib_s": time.time() - t0})
    run.log_summary(**{f"calib/norm_regret/{a}": v for a, v in means.items()})
    del tasks

    # test (split 1): EI, pi_1 at every c, the null head at c*
    t0 = time.time()
    Fs, tasks, fmed, seeds, dec = evaluate(
        cfg,
        engine,
        spec,
        {"api": (head, [*cs, INF]), "null": (null, [c_star])},
        int(t.test.split),
        int(t.test.n_tasks),
        int(t.test.task_offset),
    )
    test = final_table(Fs, tasks, fmed, seeds, float(t.eps))
    run.log_table("test", test)
    n_init = int(t.n_init_per_d * t.d)
    run.log_table("best", pd.concat([best_table(tasks, a, F, n_init) for a, F in Fs.items()], ignore_index=True))
    T = int(t.T_per_d * t.d)
    drows = []
    for arm, dd in dec.items():
        Bt = dd["deviated"].shape[0]
        drows.append(
            pd.DataFrame(
                {
                    "task": np.repeat(np.arange(Bt), T),
                    "seed": np.repeat(np.asarray(seeds), T),
                    "arm": arm,
                    "step": np.tile(np.arange(T), Bt),
                    "deviated": dd["deviated"].numpy().ravel(),
                    "source": np.asarray(SOURCES)[dd["source"].numpy().ravel()],
                    "A_hat": dd["A_hat"].numpy().ravel(),
                }
            )
        )
    run.log_table("decisions", pd.concat(drows, ignore_index=True))
    run.log_table(
        "fstar",
        pd.DataFrame(
            {"task": np.arange(tasks.B), "seed": seeds, "fstar": tasks.fstar.cpu().numpy(), "fmed": fmed.cpu().numpy()}
        ),
    )
    m = {col: _arm_means(test, col) for col in ("norm_regret", "log_regret")}
    run.log_summary(**{f"test/norm_regret/{a}": v for a, v in m["norm_regret"].items()})
    run.log_summary(**{f"test/log_regret/{a}": v for a, v in m["log_regret"].items()})
    run.log_summary(**{f"test/deviation_rate/{a}": float(dd["deviated"].float().mean()) for a, dd in dec.items()})
    run.log_summary(**{"time/test_s": time.time() - t0})
    return float(m["norm_regret"][_arm(c_star)] - m["norm_regret"]["ei"])


__all__ = ["api", "improved_pick"]

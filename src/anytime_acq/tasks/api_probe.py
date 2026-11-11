"""Readout probes (experimental/probes.py; research_log/decisions/0006 §1).

One member = one d. At the labeled states of a round-1 label set (pinned by sha256): ground-truth targets
from the tasks (redrawn from their seeds), KG by re-encoding, the frozen PFN's pooled per-layer context
states and per-layer candidate query states, and the round-1 head's hand features. Probes (ridge, MLP) are
fit on 80% of the states (split by task) and evaluated on the rest.

Tables: `probes` (target, level, input, model, metric, value, bootstrap CI over test states), `contrasts`
(all layers − last layer, all layers − hand features: paired bootstrap over test states), `reliability`
(split-half reliability of the candidate labels), `targets` (per state: the state-level targets).
"""

import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load

from ..bo.engine import Engine
from ..bo.tasks import draw_tasks
from ..experimental.api import CandidateSpec, features
from ..experimental.probes import (
    auc,
    kg_reencode,
    mlp_fit_predict,
    pooled,
    query_layer_states,
    r2,
    ridge_fit_predict,
    roughness,
    spearman,
    within_spearman,
)
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .api import _dict
from .api_gate import load_labels
from .bo import _candidates

STATE_TARGETS = ("log_gap", "xstar", "n_relevant", "rough", "headroom", "headroom_bin")
CAND_TARGETS = ("adv", "kg")


def crossfit_headroom(R: torch.Tensor, lab_valid: torch.Tensor) -> torch.Tensor:
    """[S]: the best labeled candidate selected on half of the continuation replicates and scored on the other
    half, both ways, averaged (the clairvoyant advantage map of analysis/api.py, vectorized)."""
    M = R.shape[-1]
    h = M // 2
    adv = R[:, :1] - R  # [S, J, M]
    out = []
    for sel, sc in ((slice(0, h), slice(h, M)), (slice(h, M), slice(0, h))):
        a = adv[..., sel].mean(-1).masked_fill(~lab_valid, float("-inf"))
        j = a.argmax(1)
        out.append(adv[torch.arange(len(R)), j][:, sc].mean(-1))
    return torch.stack(out).mean(0)


@torch.no_grad()
def collect(cfg: DictConfig, engine: Engine, spec: CandidateSpec, L: dict, S: int) -> dict[str, torch.Tensor]:
    t = cfg.task
    dev = t.device
    d, n_init, T = int(t.d), int(t.n_init_per_d * t.d), int(t.T_per_d * t.d)
    pfn = engine.pfn
    assert pfn is not None
    out: dict[str, list] = {k: [] for k in ("idx", "pool", "qs", "hand_s", "hand_c", "kg")}
    for tb in L["branch_t"][:S].unique().tolist():
        idx = (L["branch_t"][:S] == tb).nonzero().squeeze(1)
        for s in range(0, len(idx), int(t.group)):
            ii = idx[s : s + int(t.group)]
            X, F = L["X"][ii, : n_init + tb].to(dev), L["F"][ii, : n_init + tb].to(dev)
            C, valid, lab = L["C"][ii].to(dev), L["valid"][ii].to(dev), L["lab"][ii].to(dev)
            xq = C.gather(1, lab[..., None].expand(-1, -1, d))
            ft = features(engine, spec, X, F, C, valid, tb, T)
            out["idx"].append(ii)
            out["pool"].append(pooled(pfn, X, F).cpu())
            out["qs"].append(query_layer_states(pfn, X, F, xq).cpu())
            out["hand_s"].append(ft["glob"].cpu())
            out["hand_c"].append(ft["cand"].gather(1, lab[..., None].expand(-1, -1, ft["cand"].shape[-1])).cpu())
            out["kg"].append(kg_reencode(engine, X, F, xq, C, int(t.kg_n_y)).cpu())
    order = torch.cat(out.pop("idx")).argsort()
    return {k: torch.cat(v)[order] for k, v in out.items()}


def state_targets(cfg: DictConfig, L: dict, S: int) -> dict[str, torch.Tensor]:
    t = cfg.task
    d, n_init = int(t.d), int(t.n_init_per_d * t.d)
    seeds = L["seed"][:S].tolist()
    xs, rel, rough = [], [], []
    for s in range(0, S, int(t.draw_chunk)):
        tasks = draw_tasks(d, seeds[s : s + int(t.draw_chunk)], **_dict(t.task_draw)).to(t.device)
        xs.append(tasks.xstar.cpu())
        rel.append(tasks.prior.relevant_mask.sum(1).float().cpu())
        rough.append(roughness(tasks).cpu())
    bt = L["branch_t"][:S]
    inc = torch.stack([L["F"][i, : n_init + int(bt[i])].min() for i in range(S)])
    fstar, fmed = L["fstar"][:S], L["fmed"][:S]
    head = crossfit_headroom(L["R"][:S], L["lab_valid"][:S])
    gap = 100 * (inc - fstar).clamp(min=0) / (fmed - fstar).clamp(min=1e-9)
    # both are heavy-tailed (api_probe sanity): probe log10 of the gap and asinh of the headroom
    return {
        "log_gap": torch.log10(gap + 1e-3),
        "xstar": torch.cat(xs),
        "n_relevant": torch.cat(rel),
        "rough": torch.cat(rough),
        "headroom": torch.asinh(head),
        "headroom_bin": (head > float(t.headroom_threshold)).float(),
    }


def _boot(fn, n: int, B: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def api_probe(cfg: DictConfig, run: Run) -> float:
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
    L = load_labels(cfg)
    S = min(int(t.n_states), len(L["seed"]))
    t0 = time.time()
    tg = state_targets(cfg, L, S)
    run.log_summary(**{"time/targets_s": time.time() - t0})
    t0 = time.time()
    rep = collect(cfg, engine, spec, L, S)
    run.log_summary(**{"time/collect_s": time.time() - t0})

    dev = t.device
    gen = torch.Generator().manual_seed(int(cfg.seed) + 7)
    is_test = torch.rand(S, generator=gen) < float(t.test_frac)
    tr, te = (~is_test).nonzero().squeeze(1), is_test.nonzero().squeeze(1)
    pool = rep["pool"]  # [S, L+1, 3dm]
    inputs_s = {"hand": rep["hand_s"], "last": pool[:, -1], "all": pool.flatten(1)}
    J = rep["qs"].shape[1]
    lab_valid = L["lab_valid"][:S]
    cmask = lab_valid.clone()
    cmask[:, 0] = False  # x0's label is 0 by construction
    inputs_c = {
        "hand": torch.cat([rep["hand_c"], rep["hand_s"][:, None].expand(-1, J, -1)], -1),
        "last": torch.cat([rep["qs"][:, :, -1], pool[:, None, -1].expand(-1, J, -1)], -1),
        "all": torch.cat([rep["qs"].flatten(2), pool.flatten(1)[:, None].expand(-1, J, -1)], -1),
    }
    # asinh: the labels are heavy-tailed; monotone, so the within-state ranking is unchanged
    targets_c = {"adv": torch.asinh(L["adv"][:S]), "kg": rep["kg"]}

    rows, contrasts = [], []
    B = int(t.n_boot)
    for name in STATE_TARGETS:
        y = tg[name]
        kind = "bin" if name == "headroom_bin" else "reg"
        preds = {}
        for inp, Xs in inputs_s.items():
            for model in ("ridge", "mlp"):
                Xtr, Xte = Xs[tr].to(dev), Xs[te].to(dev)
                if model == "ridge":
                    p, _ = ridge_fit_predict(Xtr, y[tr].to(dev), Xte)
                else:
                    p = mlp_fit_predict(Xtr, y[tr].to(dev), Xte, kind, seed=int(cfg.seed))
                p, yt = p.cpu().numpy(), y[te].numpy()
                if kind == "bin":
                    metrics = {"auc": lambda ii, p=p, yt=yt: auc(p[ii], yt[ii])}
                elif yt.ndim > 1:
                    metrics = {"r2": lambda ii, p=p, yt=yt: r2(p[ii].ravel(), yt[ii].ravel())}
                else:
                    metrics = {
                        "r2": lambda ii, p=p, yt=yt: r2(p[ii], yt[ii]),
                        "spearman": lambda ii, p=p, yt=yt: spearman(p[ii], yt[ii]),
                    }
                for mname, metric in metrics.items():
                    lo, hi = _boot(metric, len(yt), B, 0)
                    rows.append(
                        {
                            "target": name,
                            "level": "state",
                            "input": inp,
                            "model": model,
                            "metric": mname,
                            "value": metric(np.arange(len(yt))),
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                # contrasts on the last metric: auc, spearman (rank, robust to the tails), r2 for x*
                preds[(inp, model)] = (metric, len(yt), mname)
        for model in ("ridge", "mlp"):
            for a, b in (("all", "last"), ("all", "hand"), ("last", "hand")):
                ma, n, mname = preds[(a, model)]
                mb, _, _ = preds[(b, model)]
                lo, hi = _boot(lambda ii, ma=ma, mb=mb: ma(ii) - mb(ii), n, B, 1)
                contrasts.append(
                    {
                        "target": name,
                        "model": model,
                        "metric": mname,
                        "a": a,
                        "b": b,
                        "diff": ma(np.arange(n)) - mb(np.arange(n)),
                        "lo": lo,
                        "hi": hi,
                    }
                )

    st_te = te.numpy()
    for name, y in targets_c.items():
        m_tr, m_te = cmask[tr], cmask[te]
        group_te = np.repeat(st_te, J).reshape(len(te), J)[m_te.numpy()]
        yt = y[te][m_te].numpy()
        preds = {}
        for inp, Xc in inputs_c.items():
            for model in ("ridge", "mlp"):
                Xtr, Xte = Xc[tr][m_tr].to(dev), Xc[te][m_te].to(dev)
                gtr = torch.arange(len(tr))[:, None].expand(-1, J)[m_tr]
                if model == "ridge":
                    p, _ = ridge_fit_predict(Xtr, y[tr][m_tr].to(dev), Xte, groups=gtr)
                else:
                    p = mlp_fit_predict(Xtr, y[tr][m_tr].to(dev), Xte, "reg", seed=int(cfg.seed), groups=gtr)
                p = p.cpu().numpy()
                rho = within_spearman(p, yt, group_te)
                rows.append(
                    {
                        "target": name,
                        "level": "candidate",
                        "input": inp,
                        "model": model,
                        "metric": "within_spearman",
                        "value": float(np.nanmean(rho)),
                        "lo": _boot(lambda ii, rho=rho: np.nanmean(rho[ii]), len(rho), B, 0)[0],
                        "hi": _boot(lambda ii, rho=rho: np.nanmean(rho[ii]), len(rho), B, 0)[1],
                    }
                )
                rows.append(
                    {
                        "target": name,
                        "level": "candidate",
                        "input": inp,
                        "model": model,
                        "metric": "r2",
                        "value": r2(p, yt),
                        "lo": np.nan,
                        "hi": np.nan,
                    }
                )
                preds[(inp, model)] = rho
        for model in ("ridge", "mlp"):
            for a, b in (("all", "last"), ("all", "hand"), ("last", "hand")):
                ra, rb = preds[(a, model)], preds[(b, model)]
                lo, hi = _boot(lambda ii, ra=ra, rb=rb: np.nanmean(ra[ii] - rb[ii]), len(ra), B, 1)
                contrasts.append(
                    {
                        "target": name,
                        "model": model,
                        "metric": "within_spearman",
                        "a": a,
                        "b": b,
                        "diff": float(np.nanmean(ra - rb)),
                        "lo": lo,
                        "hi": hi,
                    }
                )

    # split-half reliability of the ground-truth candidate labels (the ceiling for any probe of `adv`)
    R = L["R"][:S]
    h = R.shape[-1] // 2
    a1, a2 = (R[:, :1, :h] - R[:, :, :h]).mean(-1), (R[:, :1, h:] - R[:, :, h:]).mean(-1)
    grp = np.repeat(np.arange(S), J).reshape(S, J)[cmask.numpy()]
    rel = within_spearman(a1[cmask].numpy(), a2[cmask].numpy(), grp)
    run.log_table(
        "reliability",
        pd.DataFrame(
            {
                "metric": ["within_spearman_split_half"],
                "value": [float(np.nanmean(rel))],
                "n_states": [int(np.isfinite(rel).sum())],
            }
        ),
    )
    probes = pd.DataFrame(rows)
    run.log_table("probes", probes)
    run.log_table("contrasts", pd.DataFrame(contrasts))
    tt = pd.DataFrame({k: v.numpy() for k, v in tg.items() if v.dim() == 1})
    tt["seed"], tt["t"], tt["split"] = (
        L["seed"][:S].numpy(),
        L["branch_t"][:S].numpy(),
        np.where(is_test.numpy(), "test", "train"),
    )
    run.log_table("targets", tt)
    for r in rows:
        run.log_summary(**{f"probe/{r['target']}/{r['input']}/{r['model']}/{r['metric']}": float(r["value"])})
    run.log_summary(**{"reliability/adv": float(np.nanmean(rel)), "n_states": float(S), "n_test": float(len(te))})
    return float(probes.loc[(probes.target == "headroom_bin") & (probes.input == "all")].value.max())

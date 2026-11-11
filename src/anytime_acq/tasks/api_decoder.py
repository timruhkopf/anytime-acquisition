"""Amortized policy improvement, round 2: the 0006 value decoder, Phase A (experimental/decoder.py;
research_log/decisions/0006 §3).

One member = one d:
1. labels: the round-1 label set (pinned by sha256), plus `label.n_tasks` new states made exactly as in
   round 1 (tasks/api.py::make_labels) on the same split at `label.task_offset` (more data for a larger
   model); 0 = only the pinned set;
2. decoder inputs at every labeled state (frozen PFN), split by instance into train / validation;
3. the decoder (bar-distribution NLL on single-seed paired advantages, every seed a target) and the null
   decoder (targets permuted across the non-base candidates of each state, as round 1's null head);
4. calibration on `calib.split`: the decoder policy for each rule in `calib.rules` (mean, quantile levels),
   paired against pi_0 (EI); rule* = the best mean normalized regret (ties: the earlier, more conservative rule);
5. test on `test.split`: EI, the decoder at every rule (rule* is the pre-registered one), the null at rule*.

Tables: `val` (validation states' labeled candidates: label, predicted mean and quantiles), `calib`, `test`,
`best`, `decisions`, `fstar`. Artifacts: labels.pt (if new labels were made), decoders.pt.
"""

import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load

from ..bo.engine import Engine
from ..bo.runner import initial_design
from ..bo.tasks import Tasks, draw_tasks, task_seed
from ..experimental.api import CandidateSpec, task_median
from ..experimental.decoder import Bars, ValueDecoder, run_decoder, state_inputs
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .api import _arm_means, _dict, _per_task_uniforms, final_table, make_labels
from .api_gate import load_labels
from .bo import _candidates, best_table

INF = float("inf")
QUANTILES = (0.1, 0.3, 0.5)


def all_labels(cfg: DictConfig, engine: Engine, spec: CandidateSpec, run: Run) -> dict[str, torch.Tensor]:
    L = {k: v for k, v in load_labels(cfg).items() if isinstance(v, torch.Tensor)}
    if int(cfg.task.label.n_tasks) > 0:
        t0 = time.time()
        new = make_labels(cfg, engine, spec, run)
        run.log_summary(**{"time/labels_s": time.time() - t0})
        new = {k: v for k, v in new.items() if isinstance(v, torch.Tensor)}
        run.artifacts.mkdir(parents=True, exist_ok=True)
        torch.save(new, run.artifacts / "labels.pt")
        L = {k: torch.cat([L[k], new[k]]) for k in L}
    return L


@torch.no_grad()
def decoder_data(cfg: DictConfig, engine: Engine, spec: CandidateSpec, L: dict) -> list[dict[str, torch.Tensor]]:
    """Decoder inputs at the labeled candidates of every state, grouped by branch step (equal context length);
    targets a [S, J, M] = R(x0, k) − R(c, k) (single seeds)."""
    t = cfg.task
    dev = t.device
    n_init, T = int(t.n_init_per_d * t.d), int(t.T_per_d * t.d)
    read = [int(r) for r in t.decoder.read]
    src_ids = spec.source_ids()
    groups = []
    for tb in L["branch_t"].unique().tolist():
        idx = (L["branch_t"] == tb).nonzero().squeeze(1)
        parts = []
        for s in range(0, len(idx), int(t.group)):
            ii = idx[s : s + int(t.group)]
            X, F = L["X"][ii, : n_init + tb].to(dev), L["F"][ii, : n_init + tb].to(dev)
            C, valid, lab = L["C"][ii].to(dev), L["valid"][ii].to(dev), L["lab"][ii].to(dev)
            xq = C.gather(1, lab[..., None].expand(-1, -1, C.shape[-1]))
            inp = state_inputs(engine, spec, X, F, C, valid, xq, src_ids.to(dev)[lab], tb, T, read)
            parts.append(inp)
        g = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        R = L["R"][idx].to(dev)
        g.update(idx=idx.to(dev), a=R[:, :1] - R, lab_valid=L["lab_valid"][idx].to(dev))
        groups.append(g)
    return groups


def _batches(groups: list[dict], mask_of: dict[int, torch.Tensor], bs: int, gen: torch.Generator) -> list:
    out = []
    for gi in range(len(groups)):
        rows = mask_of[gi].nonzero().squeeze(1).cpu()
        rows = rows[torch.randperm(len(rows), generator=gen)]
        out += [(gi, rows[s : s + bs]) for s in range(0, len(rows), bs)]
    return [out[i] for i in torch.randperm(len(out), generator=gen).tolist()]


def _forward(dec: ValueDecoder, g: dict, rows: torch.Tensor) -> torch.Tensor:
    return dec(g["q"][rows], g["ctx"][rows], g["glob"][rows], g["bud"][rows], g["q0"][rows])


def _nll(dec: ValueDecoder, g: dict, rows: torch.Tensor, a: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
    """NLL of every single-seed target a [S, J, M] of the valid labeled candidates (x0 included: target 0)."""
    lg = _forward(dec, g, rows)  # [s, J, nb]
    M = a.shape[-1]
    nll = dec.bars.nll(lg[:, :, None].expand(-1, -1, M, -1), a[rows])
    return nll[lv[rows]].flatten()


def train_decoder(
    cfg: DictConfig, groups: list[dict], is_val: dict[int, torch.Tensor], bars: Bars, permute: bool, run: Run,
    prefix: str,
) -> ValueDecoder:  # fmt: skip
    """AdamW + cosine decay, early stopping on validation NLL; `permute`: the null decoder's targets."""
    dc, tr = cfg.task.decoder, cfg.task.train
    dev = cfg.task.device
    torch.manual_seed(int(cfg.seed) + (1 if permute else 0))
    g0 = groups[0]
    dec = ValueDecoder(
        g0["q"].shape[-1], g0["glob"].shape[-1], g0["ctx"].shape[-1], bars, int(dc.width), int(dc.blocks),
        int(dc.heads),
    ).to(dev)  # fmt: skip
    tr_mask = {gi: ~is_val[gi] for gi in range(len(groups))}
    dec.set_normalization(
        torch.cat([g["q"][tr_mask[gi]][g["lab_valid"][tr_mask[gi]]] for gi, g in enumerate(groups)]),
        torch.cat([g["glob"][tr_mask[gi]] for gi, g in enumerate(groups)]),
    )
    gen = torch.Generator().manual_seed(int(cfg.seed) * 31 + int(permute))
    targets = []
    for g in groups:
        a, lv = g["a"], g["lab_valid"]
        if permute:  # permute targets across the non-base candidates of each state (as round 1's null head)
            perm = torch.argsort(torch.rand(a.shape[0], a.shape[1] - 1, generator=gen), 1).to(dev) + 1
            perm = torch.cat([torch.zeros_like(perm[:, :1]), perm], 1)
            a, lv = a.gather(1, perm[..., None].expand(-1, -1, a.shape[-1])), lv.gather(1, perm)
        targets.append((a, lv))
    n_steps = int(tr.epochs) * len(_batches(groups, tr_mask, int(tr.batch), torch.Generator().manual_seed(0)))
    opt = torch.optim.AdamW(dec.parameters(), lr=float(tr.lr), weight_decay=float(tr.wd))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(n_steps, 1))
    best, best_state, bad, ep = INF, None, 0, 0
    for ep in range(int(tr.epochs)):
        dec.train()
        tot, n = 0.0, 0
        for gi, rows in _batches(groups, tr_mask, int(tr.batch), gen):
            loss = _nll(dec, groups[gi], rows.to(dev), *targets[gi])
            if loss.numel() == 0:
                continue
            opt.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + float(loss.detach().sum()), n + loss.numel()
        dec.eval()
        with torch.no_grad():
            v = torch.cat([_nll(dec, g, is_val[gi].nonzero().squeeze(1), *targets[gi]) for gi, g in enumerate(groups)])
            val = float(v.mean())
        run.log_metrics({f"{prefix}/train_nll": tot / max(n, 1), f"{prefix}/val_nll": val}, step=ep)
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: x.detach().clone() for k, x in dec.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tr.patience):
                break
    assert best_state is not None
    dec.load_state_dict(best_state)
    dec.eval()
    run.log_summary(**{f"{prefix}/best_val_nll": best, f"{prefix}/epochs": float(ep + 1)})
    return dec


@torch.no_grad()
def val_table(dec: ValueDecoder, groups: list[dict], is_val: dict[int, torch.Tensor], L: dict) -> pd.DataFrame:
    """Validation states' labeled candidates: seed-averaged label `adv` and the decoder's mean and quantiles."""
    out = []
    for gi, g in enumerate(groups):
        rows = is_val[gi].nonzero().squeeze(1)
        if len(rows) == 0:
            continue
        lg = _forward(dec, g, rows)
        idx = g["idx"][rows].cpu()
        S, J = lg.shape[:2]
        df = pd.DataFrame(
            {
                "state": np.repeat(idx.numpy(), J),
                "seed": np.repeat(L["seed"][idx].numpy(), J),
                "t": np.repeat(L["branch_t"][idx].numpy(), J),
                "j": np.tile(np.arange(J), S),
                "valid": g["lab_valid"][rows].cpu().numpy().ravel(),
                "adv": g["a"][rows].mean(-1).cpu().numpy().ravel(),
                "mean": dec.bars.mean(lg).cpu().numpy().ravel(),
            }
        )
        for q in QUANTILES:
            df[f"q{q:g}"] = dec.bars.quantile(lg, q).cpu().numpy().ravel()
        out.append(df)
    return pd.concat(out, ignore_index=True)


def evaluate(
    cfg: DictConfig, engine: Engine, spec: CandidateSpec, arms: dict[str, tuple[ValueDecoder, list[str]]],
    split: int, n_tasks: int, offset: int,
) -> tuple[dict[str, torch.Tensor], Tasks, torch.Tensor, list[int], dict]:  # fmt: skip
    """Every decoder at its rules on the same tasks (CRN). Arm names: "ei", "dec_<rule>", "<name>_<rule>"."""
    t = cfg.task
    d, n_init, T = int(t.d), int(t.n_init_per_d * t.d), int(t.T_per_d * t.d)
    read = [int(r) for r in t.decoder.read]
    seeds = [task_seed(split, d, offset + i) for i in range(n_tasks)]
    tasks = draw_tasks(d, seeds, **_dict(t.task_draw)).to(t.device)
    X0 = initial_design(tasks, n_init).to(t.device)
    Fs, dec_out = {}, {}
    for name, (dec, rules) in arms.items():
        out = run_decoder(engine, spec, dec, tasks, X0, T, rules, read)
        for k, r in enumerate(rules):
            arm = "ei" if r == "ei" else f"{name}_{r}"
            Fs[arm] = out["F"][:, k]
            dec_out[arm] = {key: out[key][:, k].cpu() for key in ("deviated", "source", "score")}
    for F in Fs.values():
        tasks.lower_fstar(F)
    return Fs, tasks, task_median(tasks), seeds, dec_out


def api_decoder(cfg: DictConfig, run: Run) -> float:
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
    L = all_labels(cfg, engine, spec, run)
    S = len(L["seed"])
    is_val_state = _per_task_uniforms(L["seed"].tolist(), 3)[:, 2] < float(t.label.val_frac)

    t0 = time.time()
    groups = decoder_data(cfg, engine, spec, L)
    run.log_summary(**{"time/inputs_s": time.time() - t0, "labels/n_states": float(S)})
    is_val = {gi: is_val_state.to(t.device)[g["idx"]] for gi, g in enumerate(groups)}
    a_tr = torch.cat([g["a"][~is_val[gi]][g["lab_valid"][~is_val[gi]]].flatten() for gi, g in enumerate(groups)])
    bars = Bars.from_targets(a_tr.cpu(), int(t.decoder.n_bins), float(t.decoder.zero_eps)).to(t.device)
    t0 = time.time()
    dec = train_decoder(cfg, groups, is_val, bars, permute=False, run=run, prefix="dec")
    null = train_decoder(cfg, groups, is_val, bars, permute=True, run=run, prefix="null")
    run.log_summary(**{"time/train_s": time.time() - t0, "bars/n_bins": float(bars.n_bins)})
    vt = val_table(dec, groups, is_val, L)
    run.log_table("val", vt)
    run.artifacts.mkdir(parents=True, exist_ok=True)
    torch.save({"dec": dec.state_dict(), "null": null.state_dict()}, run.artifacts / "decoders.pt")
    v = vt.loc[vt.valid & (vt.j > 0)]
    rho = v.groupby("state").apply(lambda d: d["mean"].corr(d["adv"], method="spearman"), include_groups=False)
    run.log_summary(**{"val/within_spearman_mean": float(rho.mean())})
    del groups

    # calibration of the rule (0006 §3 "Decision rule"), never on the test split
    t0 = time.time()
    rules = [str(r) for r in t.calib.rules]
    c = t.calib
    Fs, tasks, fmed, seeds, _ = evaluate(
        cfg, engine, spec, {"dec": (dec, [*rules, "ei"])}, int(c.split), int(c.n_tasks), int(c.task_offset)
    )
    cal = final_table(Fs, tasks, fmed, seeds, float(t.eps))
    run.log_table("calib", cal)
    means = _arm_means(cal, "norm_regret")
    rule_star = min(rules, key=lambda r: (round(float(means[f"dec_{r}"]), 9), rules.index(r)))
    run.log_summary(**{"calib/rule_star_index": float(rules.index(rule_star)), "time/calib_s": time.time() - t0})
    run.log_summary(**{f"calib/norm_regret/{a}": v for a, v in means.items()})
    del tasks

    # test (split 1): EI, the decoder at every rule, the null decoder at rule*
    t0 = time.time()
    Fs, tasks, fmed, seeds, dout = evaluate(
        cfg, engine, spec, {"dec": (dec, [*rules, "ei"]), "null": (null, [rule_star])}, int(t.test.split),
        int(t.test.n_tasks), int(t.test.task_offset),
    )  # fmt: skip
    test = final_table(Fs, tasks, fmed, seeds, float(t.eps))
    run.log_table("test", test)
    n_init, T = int(t.n_init_per_d * t.d), int(t.T_per_d * t.d)
    run.log_table("best", pd.concat([best_table(tasks, a, F, n_init) for a, F in Fs.items()], ignore_index=True))
    B = len(seeds)
    run.log_table(
        "decisions",
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "task": np.repeat(np.arange(B), T),
                        "seed": np.repeat(np.asarray(seeds), T),
                        "arm": arm,
                        "step": np.tile(np.arange(T), B),
                        "deviated": dd["deviated"].numpy().ravel(),
                        "source": dd["source"].numpy().ravel(),
                        "score": dd["score"].numpy().ravel(),
                    }
                )
                for arm, dd in dout.items()
            ],
            ignore_index=True,
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
    run.log_summary(**{f"test/deviation_rate/{a}": float(dd["deviated"].float().mean()) for a, dd in dout.items()})
    run.log_summary(**{"time/test_s": time.time() - t0})
    return m[f"dec_{rule_star}"] - m["ei"]

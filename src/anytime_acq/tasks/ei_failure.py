"""Can the context predict how the base policy's run will end, and does gating a deviation by that prediction
beat the best static schedule? (analysis/ei_failure.py; pre-registration: notebooks/15_2026-11-12_ei_failure.ipynb.)

Labels and context come from `option_policy.collect` (base = EI; options: UCB for k steps or for the rest, at every
decision step, under two candidate-set seeds; features at every (task, step) of the EI run).
- Target g: log10 terminal regret of the full base run under seed 0. Its luck replicate: the base lane continued
  from step j under seed 1 (same data up to j, another candidate set afterwards).
- Probe: g-hat from each feature set, one ridge per decision step and one MLP over all steps, cross-fitted over
  tasks. Table `probe`: Spearman and R2 per step, with privileged references; `probe_diff`: paired differences.
- Gate: deviate at step j to option o iff score_j >= c; (j, o, c) chosen on the training folds (for g-hat: from
  out-of-inner-fold predictions). Scores: g-hat (deployable); the seed-1 base lane (privileged, luck-corrected:
  chosen and valued on seed 0); the seed-0 outcome itself (privileged and mechanical: it is part of the label);
  the current true regret (privileged). Tables `summary`, `gates`, `strata`, `per_task`.
"""

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load

from ..analysis.ei_failure import gate_choice, gate_value, spearman_ci
from ..analysis.option_policy import decision_value, policy_verdict, static_choice
from ..analysis.stats import bootstrap_ci
from ..bo.engine import Engine
from ..experimental.option_policy import FitSpec, crossfit_q, pca
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .api import _dict
from .bo import _candidates
from .option_policy import advantages, collect


def predict_outcome(
    X: torch.Tensor, g: torch.Tensor, fold: torch.Tensor, learner: str, n_inner: int, seed: int, spec: FitSpec
) -> tuple[np.ndarray, np.ndarray]:
    """X [N, S, p], g [N] -> (G_out [N, S], G_in [K, N, S]) as `crossfit_q`: ridge is fitted per decision step,
    the MLP once over all steps."""
    N, S, _ = X.shape
    K = int(fold.max()) + 1
    task = torch.arange(N, device=X.device)
    if learner == "ridge":
        outs = [crossfit_q(X[:, j], g[:, None], task, fold, learner, n_inner, seed, spec) for j in range(S)]
        return (
            torch.cat([o[0] for o in outs], 1).cpu().numpy(),
            torch.cat([o[1] for o in outs], 2).cpu().numpy(),
        )
    q_out, q_in = crossfit_q(
        X.reshape(N * S, -1),
        g.repeat_interleave(S)[:, None],
        task.repeat_interleave(S),
        fold.repeat_interleave(S),
        learner,
        n_inner,
        seed,
        spec,
    )
    return q_out.view(N, S).cpu().numpy(), q_in.view(K, N, S).cpu().numpy()


def evaluate(L: dict, cfg: DictConfig, run: Run) -> pd.DataFrame:
    t = cfg.task
    dev, eps, nb = t.device, float(t.eps), int(t.n_boot)
    R = L["regret"]
    N, S = R.shape[:2]
    names, steps = list(L["options"]), list(L["steps"])
    assert steps[0] == 0
    lr = torch.log10(R.double() + eps)
    g0 = lr[:, 0, 0, 0].float()  # the full base run under seed 0
    g1 = lr[:, :, 0, 1].float().numpy()  # [N, S]: the base continued from step j under seed 1
    cur = torch.log10(L["state_regret"].double() + eps).float().numpy()
    A = advantages(R, eps)
    A0, A1, tgt = A[..., 0].numpy(), A[..., 1].numpy(), A.mean(-1).numpy()
    K = int(t.n_folds)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(int(cfg.seed) + 5))
    fold = torch.empty(N, dtype=torch.long)
    fold[perm] = torch.arange(N) % K
    data, belief = L["data"].float(), L["belief"].float()
    zp = pca(L["z"].float().reshape(N * S, -1), int(t.z_pca)).view(N, S, -1)
    sets = {"data": data, "belief": torch.cat([data, belief], 2), "pfn": torch.cat([data, belief, zp], 2)}
    spec = FitSpec(**_dict(t.fit))
    y = g0.numpy()
    pred: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fs in t.feature_sets:
        for learner in t.learners:
            pred[f"{fs}/{learner}"] = predict_outcome(
                sets[fs].to(dev), g0.to(dev), fold.to(dev), str(learner), int(t.n_inner), int(cfg.seed), spec
            )

    # --- probe: how well is the outcome predicted, per decision step
    rows = []
    refs = {"priv:other_seed": g1, "priv:current_regret": cur}
    for name, G in {**{k: v[0] for k, v in pred.items()}, **refs}.items():
        for j in range(S):
            rho, lo, hi = spearman_ci(G[:, j], y, n_boot=int(t.n_boot_corr))
            r2 = 1 - float(((G[:, j] - y) ** 2).mean() / y.var()) if not name.startswith("priv:") else float("nan")
            rows.append({"score": name, "step": steps[j], "spearman": rho, "lo": lo, "hi": hi, "r2": r2})
    run.log_table("probe", pd.DataFrame(rows))
    rows = []
    for learner in t.learners:
        for a, b in (("belief", "data"), ("pfn", "belief"), ("pfn", "data")):
            if f"{a}/{learner}" in pred and f"{b}/{learner}" in pred:
                for j in range(S):
                    d, lo, hi = spearman_ci(
                        pred[f"{a}/{learner}"][0][:, j], y, pred[f"{b}/{learner}"][0][:, j], int(t.n_boot_corr)
                    )
                    rows.append({"a": a, "b": b, "learner": learner, "step": steps[j], "diff": d, "lo": lo, "hi": hi})
    run.log_table("probe_diff", pd.DataFrame(rows))

    # --- strata: the options' advantage by tercile of a score at the decision step (valued on seed 0)
    rows = []
    primary_score = str(t.primary).split("/", 1)[1]
    for name, G in {
        primary_score: pred[primary_score][0],
        "priv:other_seed": g1,
        "priv:same_seed": y[:, None] + 0 * g1,
    }.items():
        for j in range(S):
            grp = np.digitize(G[:, j], np.quantile(G[:, j], [1 / 3, 2 / 3]))
            for q in range(3):
                for o, on in enumerate(names):
                    x = A0[grp == q][:, j, o]
                    lo, hi = bootstrap_ci(x, n_boot=nb)
                    rows.append(
                        {
                            "score": name,
                            "step": steps[j],
                            "tercile": q,
                            "option": on,
                            "adv0": float(x.mean()),
                            "lo": lo,
                            "hi": hi,
                        }
                    )
    run.log_table("strata", pd.DataFrame(rows))

    # --- gates, cross-fitted over tasks
    per = pd.DataFrame({"task": np.arange(N), "task_seed": L["task_seeds"], "fold": fold.numpy(), "g0": y})
    st0, st1 = np.zeros(N), np.zeros(N)
    gates = []
    f = fold.numpy()
    for k in range(K):
        tr, te = f != k, f == k
        j, o = static_choice(tgt[tr])
        jj, oo = np.full(te.sum(), j), np.full(te.sum(), o)
        st0[te], st1[te] = decision_value(A0[te], jj, oo), decision_value(A1[te], jj, oo)
        gates.append(
            {"variant": "static", "fold": k, "step": steps[j] if j >= 0 else -1, "option": names[o], "c": -np.inf}
        )
    per["static:v0"], per["static:v1"] = st0, st1
    # (score for choosing on the training tasks of fold k, score on the test tasks, advantage to choose on)
    same = y[:, None] + 0 * g1
    scores: dict[str, tuple[Callable[[int], np.ndarray], np.ndarray, np.ndarray]] = {}
    for name, (g_out, g_in) in pred.items():
        scores[f"gate/{name}"] = (lambda k, g_in=g_in: g_in[k], g_out, tgt)
    scores["gate/priv:other_seed"] = (lambda k: g1, g1, A0)  # chosen on seed-0 advantages, scored by seed 1
    scores["gate/priv:same_seed"] = (lambda k: same, same, A0)  # mechanical: the target is part of the label
    scores["gate/priv:current_regret"] = (lambda k: cur, cur, tgt)
    variants = []
    for v, (s_in, s_out, obj) in scores.items():
        v0, v1, dv = np.zeros(N), np.zeros(N), np.zeros(N, dtype=bool)
        for k in range(K):
            tr, te = f != k, f == k
            gate = gate_choice(s_in(k)[tr], obj[tr])
            v0[te], dv[te] = gate_value(s_out[te], A0[te], gate)
            v1[te], _ = gate_value(s_out[te], A1[te], gate)
            gates.append(
                {
                    "variant": v,
                    "fold": k,
                    "step": steps[gate[0]] if gate[0] >= 0 else -1,
                    "option": names[gate[1]],
                    "c": gate[2],
                }
            )
        per[f"{v}:v0"], per[f"{v}:v1"], per[f"{v}:deviates"] = v0, v1, dv
        variants.append(v)
    rows = []

    def add(variant: str, seed: int, contrast: str, x: np.ndarray, dev_share: float) -> None:
        lo, hi = bootstrap_ci(x, n_boot=nb)
        rows.append(
            {"variant": variant, "seed": seed, "contrast": contrast, "mean": float(x.mean()), "lo": lo, "hi": hi,
             "n_tasks": len(x), "deviates": dev_share}
        )  # fmt: skip

    for s in (0, 1):
        add("static", s, "vs_base", per[f"static:v{s}"].to_numpy(), float("nan"))
        for v in variants:
            x, share = per[f"{v}:v{s}"].to_numpy(), float(per[f"{v}:deviates"].to_numpy().mean())
            add(v, s, "vs_base", x, share)
            add(v, s, "vs_static", x - per[f"static:v{s}"].to_numpy(), share)
    summary = pd.DataFrame(rows)
    run.log_table("summary", summary)
    run.log_table("gates", pd.DataFrame(gates))
    run.log_table("per_task", per)
    om = [
        {"step": steps[j], "option": names[o], "adv0": float(A0[:, j, o].mean()), "adv1": float(A1[:, j, o].mean())}
        for j in range(S)
        for o in range(len(names))
    ]
    run.log_table("options", pd.DataFrame(om))
    return summary


def ei_failure(cfg: DictConfig, run: Run) -> float:
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

"""option_policy: context features are decision-time measurable and match the arms' picks, the one-deviation
policy and its comparators, the rule, and the pipeline end to end."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from anytime_acq.analysis.option_policy import (
    choose_threshold,
    decision_value,
    first_trigger,
    oracle_crossfit,
    policy_verdict,
    shuffled_value,
    static_choice,
)
from anytime_acq.bo.engine import CandidateConfig, Engine
from anytime_acq.experimental.option_policy import FitSpec, belief_features, crossfit_q, data_features
from anytime_acq.experimental.options import arm_pick
from anytime_acq.tasks.option_policy import option_policy
from anytime_acq.tracking.sink import Run


def _engine() -> Engine:
    return Engine(load("pfn_smoke")[0].eval(), None, CandidateConfig(64, 4, (0.01, 0.1)), base_seed=0, chunk=4)


def test_features_match_the_arms_and_ignore_the_future() -> None:
    eng = _engine()
    g = torch.Generator().manual_seed(1)
    X, Y = torch.rand(6, 7, 2, generator=g), torch.rand(6, 7, generator=g)
    fd, nd = data_features(X, Y, 3, 10, 4)
    fb, nb = belief_features(eng, X, Y, 0)
    assert fd.shape == (6, len(nd)) and fb.shape == (6, len(nb)) and nd[0] == "t_frac"
    assert bool(torch.isfinite(fd).all()) and bool(torch.isfinite(fb).all())
    xi = X[torch.arange(6), Y.argmin(1)]
    for arm in ("ei", "ucb0.995", "mes"):  # the feature's pick is the pick the arm makes
        d = (arm_pick(eng, arm, X, Y, 0, 3) - xi).norm(dim=-1)
        assert torch.allclose(fb[:, nb.index(f"{arm}:dist_inc")], d, atol=1e-6)
    f0, _ = data_features(X[:, :4], Y[:, :4], 0, 10, 4)  # t = 0: no BO step yet
    assert bool(torch.isfinite(f0).all()) and bool((f0[:, nd.index("since_improved")] == 0).all())
    # an improvement at the last step is seen as such
    Y2 = Y.clone()
    Y2[:, -1] = Y[:, :-1].min(1).values - 0.1
    f2, _ = data_features(X, Y2, 3, 10, 4)
    assert bool((f2[:, nd.index("since_improved")] == 0).all())
    assert torch.allclose(f2[:, nd.index("log_last_improvement")], torch.full((6,), -1.0), atol=1e-3)


def test_one_deviation_policy_and_comparators() -> None:
    Q = np.zeros((3, 4, 2))
    Q[0, 2, 1] = 1.0  # task 0: triggers at step 2, option 1
    Q[1, 1, 0], Q[1, 3, 1] = 0.6, 2.0  # task 1: the first trigger wins, not the largest
    j, o = first_trigger(Q, 0.5)
    assert j.tolist() == [2, 1, -1] and o[:2].tolist() == [1, 0]
    A = np.arange(24, dtype=float).reshape(3, 4, 2)
    assert decision_value(A, j, o).tolist() == [A[0, 2, 1], A[1, 1, 0], 0.0]
    assert first_trigger(Q, np.inf)[0].tolist() == [-1, -1, -1]
    # the shuffled null: every task gets the average of all tasks' decisions
    sh = shuffled_value(A, j, o)
    assert np.isclose(sh[2], (A[2, 2, 1] + A[2, 1, 0] + 0.0) / 3)
    g = np.random.default_rng(0)
    luck0, luck1 = g.normal(size=(500, 4, 3)), g.normal(size=(500, 4, 3))
    assert abs(oracle_crossfit(luck0, luck1).mean()) < 0.1  # pure luck does not transfer across seeds
    sig = np.zeros((500, 4, 3))
    sig[:, 1, 2] = 1.0
    assert static_choice(sig + 0.01 * luck0) == (1, 2) and static_choice(-np.abs(luck0)) == (-1, 0)
    # a useless Q never beats the base on the data the threshold is chosen on: c = inf is available
    assert choose_threshold(g.normal(size=(200, 4, 3)), -np.abs(luck0[:200])) == np.inf
    assert np.isfinite(choose_threshold(sig, sig))


def test_crossfit_learns_a_contextual_signal() -> None:
    g = torch.Generator().manual_seed(0)
    n_tasks, S = 120, 3
    X = torch.randn(n_tasks * S, 5, generator=g)
    Y = torch.stack([X[:, 0].clamp(min=0), -X[:, 0].clamp(min=0)], 1) + 0.05 * torch.randn(n_tasks * S, 2, generator=g)
    task = torch.arange(n_tasks).repeat_interleave(S)
    fold = task % 3
    Qo, Qi = crossfit_q(X, Y, task, fold, "mlp", 2, 0, FitSpec(epochs=60, n_ens=1, batch=128))
    assert bool(torch.isfinite(Qo).all())
    assert float(torch.corrcoef(torch.stack([Qo[:, 0], Y[:, 0]]))[0, 1]) > 0.8
    for k in range(3):  # inner predictions exist exactly off fold k
        assert bool(torch.isnan(Qi[k][fold == k]).all()) and bool(torch.isfinite(Qi[k][fold != k]).all())
    Qr, _ = crossfit_q(X, Y, task, fold, "ridge", 2, 0)
    assert bool(torch.isfinite(Qr).all())


def test_verdict() -> None:
    def s(base: tuple, ctx: tuple) -> pd.DataFrame:
        rows = [("vs_base", *base), ("vs_static", *ctx)]
        return pd.DataFrame(
            [{"variant": "p", "seed": 0, "contrast": c, "mean": m, "lo": lo, "hi": hi} for c, m, lo, hi in rows]
        )

    assert policy_verdict(s((0.2, 0.1, 0.3), (0.05, 0.01, 0.09)), "p", 0.022) == "GO"
    assert policy_verdict(s((0.2, 0.1, 0.3), (0.0, -0.02, 0.015)), "p", 0.022) == "NO-GO"  # static only
    assert policy_verdict(s((0.2, 0.1, 0.3), (0.02, -0.01, 0.05)), "p", 0.022) == "INCONCLUSIVE"


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=option_policy",
                "task.device=cpu",
                "task.pfn=pfn_smoke",
                "task.T_per_d=3",
                "task.candidates.n_sobol=32",
                "task.candidates.n_local=2",
                "task.ks=[1]",
                "task.steps=[0,2]",
                "task.tasks.n_tasks=12",
                "task.task_chunk=7",
                "task.task_draw.n_dense=64",
                "task.task_draw.refine_steps=2",
                "task.n_folds=3",
                "task.n_inner=2",
                "task.z_pca=4",
                "task.fit.epochs=5",
                "task.fit.n_ens=1",
                "task.n_boot=20",
                "task.chunk=8",
            ],
        )
    run = Run(tmp_path)
    option_policy(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert {"summary", "per_task", "usage", "fit", "options"} <= set(tables)
    L = torch.load(run.artifacts / "labels.pt", weights_only=False)
    part = torch.load(run.artifacts / "labels_partial.pt", weights_only=False)  # written after the first chunk
    assert part["regret"].shape[0] == 7 and torch.equal(part["data"], L["data"][:7])
    R = L["regret"]
    assert R.shape == (12, 2, 1 + 4 * 2, 2) and len(L["options"]) == 8
    # under seed 0 the base lane from any step is the prefix itself: one value per task
    assert bool((R[:, 0, 0, 0] == R[:, 1, 0, 0]).all())
    assert float(R.min()) == 0.0  # f* is lowered by the runs
    per, summ = tables["per_task"], tables["summary"]
    assert len(per) == 12 and {"nomes/belief/mlp", "all/pfn/mlp", "all/belief/ridge", "nomes/static"} <= set(
        summ.variant
    )
    assert bool((per["full:base0"].to_numpy() == R[:, 0, 0, 0].numpy()).all()) and "full:mes_rest1" in per
    v = "nomes/belief/mlp"
    assert not per[f"{v}:option"].str.startswith("mes").any()  # the portfolio excludes MES
    never = per[f"{v}:j"] < 0
    assert bool((per.loc[never, f"{v}:v0"] == 0).all())
    # the value is the label of the decision: recomputed from the raw regrets
    eps = float(cfg.task.eps)
    re = np.log10(per[f"{v}:regret_base0"] + eps) - np.log10(per[f"{v}:regret0"] + eps)
    assert np.allclose(re, per[f"{v}:v0"], atol=1e-5)

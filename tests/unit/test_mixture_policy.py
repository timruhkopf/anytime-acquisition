from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from pfn_repro.checkpoints import load

from anytime_acq.bo.engine import Engine
from anytime_acq.experimental.mixture import FEATURES, N_TIME, mixture_policy, mixture_step, one_hot_theta
from anytime_acq.experimental.options import arm_pick
from anytime_acq.tasks.bo import _candidates
from anytime_acq.tasks.mixture_policy import mixture, objective
from anytime_acq.tracking.sink import Run

ARMS = ["ei", "pi0.01", "ucb0.995", "mes", "mean"]


def test_one_hot_mixture_is_the_arm_and_weights_follow_theta() -> None:
    pfn = load("pfn_smoke")[0].eval()
    cand = _candidates(OmegaConf.create({"n_sobol": 64, "n_local": 4, "local_sigmas": [0.01, 0.2]}))
    eng = Engine(pfn, None, cand, chunk=4)
    g = torch.Generator().manual_seed(0)
    X, Y = torch.rand(6, 7, 2, generator=g), torch.rand(6, 7, generator=g)
    st = mixture_step(eng, ARMS, X, Y, 0, 10)
    assert st["Z"].shape == (6, 72, 5) and st["feats"].shape == (6, len(FEATURES))
    assert float(st["Z"].masked_fill(~st["valid"][..., None], -1.0).amax(1).abs().max()) == 0.0  # each arm's best is 0
    assert bool(torch.isfinite(st["feats"]).all())
    nf = N_TIME + len(FEATURES)
    stats = (torch.zeros(len(FEATURES)), torch.ones(len(FEATURES)))
    theta = torch.stack([one_hot_theta(k, 5) for k in range(5)])
    mask = torch.zeros(5, nf)
    mask[:, 0] = 1
    rows = torch.arange(6)
    for k, arm in enumerate(ARMS):
        rec: list[torch.Tensor] = []
        pol = mixture_policy(eng, ARMS, theta, mask, torch.full((6,), k), 10, 0, stats, rec)
        x = pol(X, Y, rows, rows * 0, 3)
        assert torch.allclose(x, arm_pick(eng, arm, X, Y, 0, 3)), arm
        assert torch.allclose(rec[0][:, k], torch.ones(6))
    # a context-dependent theta gives different weights in different states; masked out, it does not
    th = torch.zeros(1, 5, nf)
    th[0, 0, N_TIME:] = 1.0
    full = torch.ones(1, nf)
    rec = []
    mixture_policy(eng, ARMS, th, full, rows * 0, 10, 0, stats, rec)(X, Y, rows, rows * 0, 3)
    assert float(rec[0][:, 0].std()) > 0
    rec = []
    mixture_policy(eng, ARMS, th, mask[:1], rows * 0, 10, 0, stats, rec)(X, Y, rows, rows * 0, 3)
    assert torch.allclose(rec[0], torch.full((6, 5), 0.2))


def test_objective() -> None:
    best = torch.tensor([[[0.5, 0.11, 0.101]], [[0.5, 0.5, 0.1]]])  # [2 policies, 1 task, T + 1 = 3]
    o = objective(best, torch.tensor([0.1]), 1e-3, 0.5)
    assert o["final"].shape == (2, 1)
    assert float(o["final"][1, 0]) < float(o["final"][0, 0])  # policy 1 ends better
    assert float(o["auc"][0, 0]) < float(o["auc"][1, 0])  # policy 0 got there sooner
    assert abs(float(o["final"][1, 0]) - (-3.0)) < 1e-6  # regret 0 -> the floor


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=mixture_policy",
                "task.device=cpu",
                "task.pfn=pfn_smoke",
                "task.T=3",
                "task.candidates.n_sobol=32",
                "task.candidates.n_local=2",
                "task.search.generations=2",
                "task.search.pairs=2",
                "task.search.val_every=1",
                "task.train_tasks.n_tasks=6",
                "task.val_tasks.n_tasks=4",
                "task.test_tasks.n_tasks=7",
                "task.task_chunk=4",
                "task.task_draw.n_dense=64",
                "task.task_draw.refine_steps=2",
                "task.n_boot=20",
                "task.chunk=8",
            ],
        )
    run = Run(tmp_path)
    mixture(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert {"train", "theta", "test_per_task", "test_curves", "weights", "summary"} <= set(tables)
    per, summ, w, tr = tables["test_per_task"], tables["summary"], tables["weights"], tables["train"]
    assert len(per) == 7 * 8 * 2 and set(per.policy) == {*ARMS, "uniform", "budget", "context"}
    assert {"J@0.001", "final@1e-06", "auc@0.0001", "regret_final"} <= set(per.columns)
    assert bool((per.regret_final >= 0).all())
    s = summ[(summ.variant == "context") & (summ.seed == 0)].set_index("contrast")
    assert {"vs_base", "vs_static", "vs_ei"} <= set(s.index)
    tot = w.groupby(["policy", "step"]).weight.sum()
    assert bool(((tot - 1).abs() < 1e-5).all()) and set(w.step) == {0, 1, 2}
    assert set(tr.phase) == {"budget", "context"} and len(tr) == 2 * 3
    th = tables["theta"]
    assert float(th[(th.policy == "budget") & ~th.term.isin(["const", "tau", "tau2"])].theta.abs().max()) == 0.0
    assert (run.artifacts / "policy.pt").exists()

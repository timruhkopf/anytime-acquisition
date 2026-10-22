"""BBOB tasks, the rank transform, the batched GP and its arms, and the bo pipeline off the prior."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_module

from anytime_acq.experimental.bbob import HI, LO, bbob_tasks, task_list
from anytime_acq.experimental.surrogates import fit_gp, gp_scores, rank_uniform
from anytime_acq.tasks.bo import bo
from anytime_acq.tracking.sink import Run


def test_rank_uniform_is_monotone_with_average_ties() -> None:
    u = rank_uniform(torch.tensor([[3.0, 1.0, 2.0, 1.0]]))
    assert torch.allclose(u, torch.tensor([[0.875, 0.25, 0.625, 0.25]]))


def test_bbob_is_shifted_to_zero_at_the_optimum_and_rows_select_tasks() -> None:
    ids = task_list([1, 8], [1], 1)
    tasks = bbob_tasks(2, [10, 11], ids)
    xopt = torch.tensor([[(v - LO) / (HI - LO) for v in p.optimum.x] for p in tasks._problems], dtype=torch.float32)
    assert tasks.f(xopt[:, None]).abs().max() < 1e-3  # float32 x at the optimum
    x = torch.rand(2, 5, 2)
    assert (tasks.f(x) >= 0).all()
    lanes = tasks.f(x[[1, 0, 1]], rows=torch.tensor([1, 0, 1]))
    assert torch.equal(lanes, tasks.f(x)[[1, 0, 1]])


def test_gp_fits_a_smooth_function() -> None:
    g = torch.Generator().manual_seed(0)
    X = torch.rand(4, 30, 2, generator=g)
    f = lambda x: torch.sin(6 * x[..., 0]) + x[..., 1] ** 2  # noqa: E731
    z = f(X)
    gp = fit_gp(X, (z - z.mean(1, keepdim=True)) / z.std(1, keepdim=True))
    Xt = torch.rand(4, 50, 2, generator=g)
    zt = (f(Xt) - z.mean(1, keepdim=True)) / z.std(1, keepdim=True)
    mu, _ = gp.predict(Xt.double())
    assert (mu - zt).abs().mean() < 0.1
    assert gp.predict(X.double())[1].max() < 0.1  # small sd at the data


def test_gp_arms_prefer_what_they_should() -> None:
    mu = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
    sd = torch.tensor([[0.1, 1.0, 0.1]], dtype=torch.float64)
    inc = torch.tensor([0.0], dtype=torch.float64)
    assert gp_scores("mean", mu, sd, inc).argmax() == 2
    assert gp_scores("ucb0.99", mu, sd, inc).argmax() == 1
    assert gp_scores("std", mu, sd, inc).argmax() == 1
    for arm in ("ei", "pi0.01", "mes"):
        s = gp_scores(arm, mu, sd, inc)
        assert torch.isfinite(s).all() and s[0, 1] > s[0, 0]  # same mean, more spread: better


TINY = [
    "experiment=bbob_switching",
    "task.device=cpu",
    "task.benchmark=bbob",
    "task.bbob.fids=[1,15]",
    "task.bbob.reps=2",
    "task.split=0",
    "task.T_per_d=2",
    "task.segment_per_d=1",
    "task.n_rollouts=2",
    "task.null_k=2",
    "task.candidates.n_sobol=64",
    "task.candidates.n_local=4",
    "task.d=2",
    "task.surrogate=gp",
    "task.gp_fit_iters=20",
]


def _run(overrides: list[str], tmp_path: Path) -> Run:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[*TINY, *overrides])
    run = Run(tmp_path)
    bo(cfg, run)
    return run


def test_bbob_static_and_null_with_the_gp(tmp_path: Path) -> None:
    run = _run(["task.arms=[ei,ucb0.99]", "task.null_reference=true", "task.base=ei"], tmp_path / "s")
    tables = run.artifacts / "tables"
    best, tasks = pd.read_parquet(tables / "best.parquet"), pd.read_parquet(tables / "tasks.parquet")
    assert set(best.arm) == {"ei", "ucb0.99", "ei@0", "random"}
    assert len(tasks) == 4 and set(tasks.fid) == {1, 15} and set(tasks.rep) == {0, 1}
    assert (pd.read_parquet(tables / "fstar.parquet").fstar == 0).all()
    null = _run(["task.method=equal_arms", "task.base=ei"], tmp_path / "n")
    assert set(pd.read_parquet(null.artifacts / "tables" / "decisions.parquet").chosen) <= {"ei@0", "ei@1"}


def test_belief_is_refused_off_the_prior(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="belief"):
        _run(["task.method=belief", "task.arms=[ei]", "task.base=ei"], tmp_path)

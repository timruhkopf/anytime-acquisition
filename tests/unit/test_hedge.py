"""GP-Hedge (bo/runner.py::run_hedge): exact null, Hedge's update rule, CRN, preference for the arm that
nominates low posterior means, and the pipeline with the PFN and the GP."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from anytime_acq.bo.engine import CandidateConfig, Engine
from anytime_acq.bo.runner import Budget, hedge_eta, run_hedge, run_static
from anytime_acq.bo.tasks import draw_tasks, task_seed
from anytime_acq.tasks.bo import bo
from anytime_acq.tracking.sink import Run

BUDGET = Budget(n_init=3, T=6, segment=3)


@pytest.fixture(scope="module")
def setup():
    pfn, _ = load("pfn_smoke")
    tasks = draw_tasks(2, [task_seed(1, 2, i) for i in range(4)], n_dense=1024, refine_steps=20)
    return Engine(pfn, candidates=CandidateConfig(n_sobol=128, n_local=8)), tasks


def test_hedge_over_identical_arms_is_the_static_arm(setup) -> None:
    """The null: identical arms nominate the same point, so the choice cannot matter."""
    eng, tasks = setup
    static = run_static(eng, tasks, "ei", BUDGET, "cpu")
    out = run_hedge(eng, tasks, ["ei", "ei", "ei"], BUDGET, 5.0, "cpu")
    torch.testing.assert_close(out["X"], static["X"])
    torch.testing.assert_close(out["F"], static["F"])
    torch.testing.assert_close(out["gains"][..., 0], out["gains"][..., 1])


def test_probabilities_follow_hedge_and_runs_are_reproducible(setup) -> None:
    eng, tasks = setup
    eta = 3.0
    out = run_hedge(eng, tasks, ["ei", "ucb0.9", "pi0.01"], BUDGET, eta, "cpu")
    p, g = out["probs"], out["gains"]
    torch.testing.assert_close(p[:, 0], torch.full_like(p[:, 0], 1 / 3))  # g_0 = 0
    torch.testing.assert_close(p[:, 1:], torch.softmax(eta * g[:, :-1].double(), -1).float())
    assert bool((g.diff(dim=1) <= 0).all())  # rewards are -mu on [0, 1]
    again = run_hedge(eng, tasks, ["ei", "ucb0.9", "pi0.01"], BUDGET, eta, "cpu")
    assert torch.equal(out["choice"], again["choice"]) and torch.equal(out["X"], again["X"])


def test_eta_zero_is_uniform_and_large_eta_follows_the_best_gain(setup) -> None:
    """`mean` always nominates the lowest posterior mean, so it earns the highest reward; `std` doesn't."""
    eng, tasks = setup
    flat = run_hedge(eng, tasks, ["std", "mean"], BUDGET, 0.0, "cpu")
    assert bool((flat["probs"] == 0.5).all())
    greedy = run_hedge(eng, tasks, ["std", "mean"], BUDGET, 1e4, "cpu")
    assert bool((greedy["gains"][:, -1, 1] > greedy["gains"][:, -1, 0]).all())
    assert bool((greedy["choice"][:, 1:] == 1).all())


def test_auto_eta() -> None:
    assert hedge_eta("auto", 3, 20) == pytest.approx(math.sqrt(8 * math.log(3) / 20))
    assert hedge_eta(0.5, 3, 20) == 0.5


TINY = [
    "task.device=cpu",
    "task.n_tasks=3",
    "task.T_per_d=2",
    "task.candidates.n_sobol=64",
    "task.candidates.n_local=4",
    "task.task_draw.n_dense=256",
    "task.task_draw.refine_steps=5",
    "task.d=2",
    "task.method=hedge",
    "task.arms=[ei,ucb0.9]",
]


def _tables(overrides: list[str], tmp_path: Path) -> dict[str, pd.DataFrame]:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=static_tune", *TINY, *overrides])
    run = Run(tmp_path)
    bo(cfg, run)
    return {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}


@pytest.mark.parametrize(
    "surrogate",
    [
        ["task.pfn=pfn_smoke"],
        [
            "task.benchmark=bbob",
            "task.bbob.fids=[1,15]",
            "task.bbob.reps=1",
            "task.split=0",
            "task.surrogate=gp",
            "task.gp_fit_iters=20",
        ],
    ],
    ids=["pfn", "gp"],
)
def test_pipeline_logs_the_hedge_table(surrogate: list[str], tmp_path: Path) -> None:
    tab = _tables(["task.hedge_eta=auto", *surrogate], tmp_path)
    h, best = tab["hedge"], tab["best"]
    B = best.task.nunique()
    assert set(best.arm) == {"hedge"} and len(h) == B * 4  # tasks x T
    assert set(h.chosen) <= {"ei", "ucb0.9"}
    np.testing.assert_allclose(h[["p/ei", "p/ucb0.9"]].sum(axis=1), 1.0, atol=1e-6)

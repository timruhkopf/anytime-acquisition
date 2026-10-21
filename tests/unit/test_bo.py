"""The BO machinery's validity properties: reproducible tasks, common random numbers, an exact controller
when all arms are equal, non-anticipating rollouts, a correct posterior sampler."""

import pytest
import torch
from pfn_repro.checkpoints import load

from anytime_acq.bo.engine import CandidateConfig, Engine
from anytime_acq.bo.runner import Budget, regret_curve, run_controller, run_static, sample_bar
from anytime_acq.bo.tasks import draw_tasks, task_seed

BUDGET = Budget(n_init=3, T=6, segment=3)


@pytest.fixture(scope="module")
def setup():
    pfn, _ = load("pfn_smoke")
    tasks = draw_tasks(2, [task_seed(1, 2, i) for i in range(4)], n_dense=1024, refine_steps=20)
    return Engine(pfn, candidates=CandidateConfig(n_sobol=128, n_local=8)), tasks


def test_tasks_are_reproducible_and_fstar_is_a_lower_bound() -> None:
    a = draw_tasks(2, [task_seed(1, 2, 5)], n_dense=512, refine_steps=10)
    b = draw_tasks(2, [task_seed(1, 2, 5)], n_dense=512, refine_steps=10)
    x = torch.rand(1, 300, 2)
    torch.testing.assert_close(a.f(x), b.f(x))
    assert bool((a.f(x) >= a.fstar[:, None] - 1e-6).all())
    assert task_seed(0, 2, 0) != task_seed(1, 2, 0)  # splits are disjoint seed ranges


def test_candidates_are_common_across_arms_and_fresh_across_replicates(setup) -> None:
    eng, tasks = setup
    X, Y = torch.rand(1, 3, 2).expand(2, -1, -1), torch.rand(1, 3).expand(2, -1)  # same history
    rows = torch.tensor([1, 1])
    same = eng.candidate_set(X, Y, rows, torch.tensor([0, 0]), 2, tasks.B)
    torch.testing.assert_close(same[0], same[1])  # the arm doesn't enter: CRN
    diff = eng.candidate_set(X, Y, rows, torch.tensor([0, 1]), 2, tasks.B)
    assert not torch.allclose(diff[0], diff[1])  # another replicate: fresh nuisance


def test_controller_over_identical_arms_is_the_static_arm(setup) -> None:
    """With identical arms the choice cannot matter: the evaluated trajectory must equal the static run
    exactly. This checks that rollouts copy the state and never touch the evaluated trajectory."""
    eng, tasks = setup
    static = run_static(eng, tasks, "ei", BUDGET, "cpu")
    ctrl = run_controller(eng, tasks, ["ei", "ei"], "ei", BUDGET, 2, "privileged", "cpu")
    torch.testing.assert_close(ctrl["X"], static["X"])
    torch.testing.assert_close(ctrl["F"], static["F"])


def test_rollouts_never_use_the_evaluated_trajectorys_nuisance(setup, monkeypatch) -> None:
    """Non-anticipation: every rollout lane runs on a replicate id > 0 (replicate 0 is the evaluation)."""
    from anytime_acq.bo import runner

    eng, tasks = setup
    seen = []
    real = runner.simulate

    def spy(engine, tasks_, X, Y, F, rows, reps, arm_at, t0, t1, mode):
        seen.append((len(rows), reps.min().item()))
        return real(engine, tasks_, X, Y, F, rows, reps, arm_at, t0, t1, mode)

    monkeypatch.setattr(runner, "simulate", spy)
    runner.run_controller(eng, tasks, ["ei", "pi"], "ei", BUDGET, 2, "belief", "cpu")
    rollout_calls = [m for n, m in seen if n > tasks.B]
    assert rollout_calls and min(rollout_calls) >= 1


def test_sample_bar_matches_the_distribution(setup) -> None:
    eng, _ = setup
    bd = eng.pfn.bar_dist
    logits = torch.zeros(20_000, bd.num_bars)
    logits[:, 3] = 4.0  # most mass in bin 3
    y = sample_bar(bd, logits, torch.rand(20_000, generator=torch.Generator().manual_seed(0)))
    assert bool(((y >= bd.borders[0]) & (y <= bd.borders[-1])).all())
    p3 = torch.softmax(logits[0], -1)[3].item()
    in3 = ((y >= bd.borders[3]) & (y < bd.borders[4])).float().mean().item()
    assert abs(in3 - p3) < 0.01


def test_regret_curve_is_monotone_and_nonnegative(setup) -> None:
    eng, tasks = setup
    out = run_static(eng, tasks, "ucb0.99", BUDGET, "cpu")
    r = regret_curve(out["F"], tasks.fstar, BUDGET.n_init)
    assert r.shape == (tasks.B, BUDGET.T + 1)
    assert bool((r >= 0).all()) and bool((r[:, 1:] <= r[:, :-1] + 1e-7).all())

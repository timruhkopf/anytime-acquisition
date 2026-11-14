"""Amortized policy improvement (experimental/api.py, tasks/api.py, analysis/api.py): the base pick is
EI's pick exactly, the query path reproduces the PFN, the head is anchored, the conservative rule falls back
to EI, labels are paired, and the pipeline runs end to end."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from anytime_acq.analysis.api import advantage_map, api_verdict, paired_delta
from anytime_acq.bo.engine import CandidateConfig, Engine
from anytime_acq.bo.runner import initial_design
from anytime_acq.bo.tasks import draw_tasks, task_seed
from anytime_acq.experimental.api import (
    N_CAND_FEATURES,
    N_GLOBAL_FEATURES,
    CandidateSpec,
    LabelSpec,
    ValueHead,
    advantages,
    base_picks,
    candidate_set,
    f_lanes,
    features,
    generate_labels,
    improved_pick,
    pi0,
    query_states,
    rollout,
    run_improved,
)
from anytime_acq.tasks.api import api
from anytime_acq.tracking.sink import Run

SPEC = CandidateSpec(n_global=8, n_top=2, n_local_per_point=3)


@pytest.fixture(scope="module")
def setup():
    pfn, _ = load("pfn_smoke")
    tasks = draw_tasks(2, [task_seed(4, 2, i) for i in range(3)], n_dense=512, refine_steps=5)
    eng = Engine(pfn, candidates=CandidateConfig(n_sobol=64, n_local=4), chunk=2)
    return eng, tasks


def _state(tasks, n: int = 4):
    X = initial_design(tasks, n)
    rows = torch.arange(tasks.B)
    return X, f_lanes(tasks, X.reshape(-1, 2), rows.repeat_interleave(n)).view(tasks.B, n), rows


def test_query_states_reproduce_the_pfn_logits(setup) -> None:
    eng, tasks = setup
    X, Y, _ = _state(tasks)
    nf = torch.full((tasks.B,), 2)
    xq = torch.rand(tasks.B, 7, 2)
    with torch.no_grad():
        kv = eng.pfn.context_kv(X, Y, n_features=nf)
        z = query_states(eng.pfn, kv, xq, nf)
        torch.testing.assert_close(eng.pfn.out_head(eng.pfn.out_ln(z)), eng.pfn.logits_from_context_kv(kv, xq, nf))


def test_base_pick_is_engine_ei_pick(setup) -> None:
    eng, tasks = setup
    X, Y, rows = _state(tasks)
    reps = torch.zeros(tasks.B, dtype=torch.long)
    x0 = base_picks(eng, X, Y, rows, reps, 1, tasks.B)[:, 0]
    torch.testing.assert_close(x0, eng.propose(X, Y, ["ei"] * tasks.B, rows, reps, 1, tasks.B)["x"])


def test_candidate_set_starts_with_x0_and_is_common_across_lanes(setup) -> None:
    eng, tasks = setup
    X, Y, rows = _state(tasks)
    reps = torch.zeros(tasks.B, dtype=torch.long)
    C, valid = candidate_set(eng, SPEC, X, Y, rows, reps, 2, tasks.B)
    assert C.shape == (tasks.B, SPEC.size, 2)
    assert bool(valid[:, 0].all())
    assert bool(((C >= 0) & (C <= 1)).all())
    torch.testing.assert_close(C[:, 0], base_picks(eng, X, Y, rows, reps, 2, tasks.B)[:, 0])
    C2, _ = candidate_set(eng, SPEC, X.repeat(2, 1, 1), Y.repeat(2, 1), rows.repeat(2), reps.repeat(2), 2, tasks.B)
    torch.testing.assert_close(C2[: tasks.B], C2[tasks.B :])  # same task, step, replicate: same candidates


def test_head_is_anchored_and_rule_falls_back_to_x0() -> None:
    torch.manual_seed(0)
    head = ValueHead(16, width=32, layers=1, heads=2)
    L, C = 3, 9
    valid = torch.ones(L, C, dtype=torch.bool)
    A, sig = head(torch.randn(L, C, 16), torch.randn(L, C, N_CAND_FEATURES), torch.randn(L, N_GLOBAL_FEATURES), valid)
    assert bool((A[:, 0] == 0).all())
    assert bool((sig > 0).all())
    A = torch.tensor([[0.0, 2.0, -1.0], [0.0, -0.5, -1.0], [0.0, 2.0, 1.0]])
    sig = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 3.0, 0.1]])
    valid = torch.ones(3, 3, dtype=torch.bool)
    j, dev = improved_pick(A, sig, valid, torch.tensor([0.0, 0.0, 1.0]))
    assert j.tolist() == [1, 0, 2]  # 2 > 0; nothing > 0; lcb 2-3 < 0 but 1-0.1 > 0
    assert dev.tolist() == [True, False, True]
    j, dev = improved_pick(A, sig, valid, torch.full((3,), float("inf")))
    assert j.tolist() == [0, 0, 0]
    assert not bool(dev.any())
    valid[0, 1] = False
    assert improved_pick(A, sig, valid, torch.zeros(3))[0][0] == 0  # invalid candidates are never chosen


def test_f_lanes_matches_per_task_evaluation(setup) -> None:
    _, tasks = setup
    rows = torch.tensor([2, 0, 2, 1, 2])
    x = torch.rand(5, 2)
    ref = torch.stack([tasks.f(x[i][None, None], rows[i : i + 1])[0, 0] for i in range(5)])
    torch.testing.assert_close(f_lanes(tasks, x, rows), ref)


def test_labels_are_paired_and_x0_follows_the_behaviour_rollout(setup) -> None:
    eng, tasks = setup
    n_init, T = 3, 4
    X0 = initial_design(tasks, n_init)
    branch_t = torch.tensor([0, 2, 3])
    L = generate_labels(eng, SPEC, LabelSpec(n_labeled=5, n_cont=2), tasks, n_init, T, X0, ["ei"] * 3, branch_t, 0)
    assert L["R"].shape == (3, 5, 2)
    for b, t in enumerate(branch_t.tolist()):  # behaviour = EI on replicate 0: x0 is its next point
        torch.testing.assert_close(L["C"][b, 0], L["X"][b, n_init + t])
    assert bool((L["lab"][:, 0] == 0).all())
    adv, se = advantages(L["R"], L["lab_valid"])
    assert bool((adv[:, 0] == 0).all())
    assert bool((se[:, 0] == 0).all())
    assert bool((L["R"] >= 0).all())


def test_improved_policy_with_infinite_c_is_ei(setup) -> None:
    eng, tasks = setup
    T = 3
    X0 = initial_design(tasks, 3)
    feats = features(
        eng,
        SPEC,
        X0,
        f_lanes(tasks, X0.reshape(-1, 2), torch.arange(3).repeat_interleave(3)).view(3, 3),
        *candidate_set(eng, SPEC, X0, tasks.f(X0), torch.arange(3), torch.zeros(3, dtype=torch.long), 0, 3),
        0,
        T,
    )
    head = ValueHead(feats["z"].shape[-1], width=32, layers=1, heads=2)
    out = run_improved(eng, SPEC, head, tasks, X0, T, [float("inf"), 0.0])
    rows = torch.arange(3)
    _, F = rollout(tasks, X0, tasks.f(X0), rows, torch.zeros(3, dtype=torch.long), 0, T, pi0(eng, 3))
    torch.testing.assert_close(out["F"][:, 0], F)
    assert not bool(out["deviated"][:, 0].any())


def test_analysis_rule_and_advantage_map() -> None:
    fin = pd.DataFrame({"task": [0, 1, 2, 0, 1, 2], "arm": ["a"] * 3 + ["ei"] * 3, "norm_regret": [1, 2, 3, 2, 3, 4.0]})
    d = paired_delta(fin, "a", "ei", n_boot=200)
    assert d["delta"] == pytest.approx(-1.0)
    assert d["rel"] == pytest.approx(-1 / 3)
    null = {"hi": 1.0}
    assert api_verdict(d, null) == "GO"
    assert api_verdict(d, {"hi": -0.1}) == "INVALID"
    assert api_verdict({"rel": 0.0, "hi": 0.1, "rel_lo": -0.01}, null) == "NO-GO"
    assert api_verdict({"rel": -0.04, "hi": 0.1, "rel_lo": -0.2}, null) == "INCONCLUSIVE"
    lab = pd.DataFrame(
        {
            "state": [0, 0, 0],
            "j": [0, 1, 2],
            "t": 1,
            "behaviour": "ei",
            "valid": [True, True, False],
            "R0": [5.0, 1.0, 0.0],
            "R1": [5.0, 1.0, 0.0],
            "R2": [5.0, 6.0, 0.0],
            "R3": [5.0, 6.0, 0.0],
        }
    )
    m = advantage_map(lab, 4)
    # j=1 selected on (R0,R1): adv 4, scored on (R2,R3): -1; selected on (R2,R3): j=0 (adv 0) -> 0
    assert m.best_crossfit.iloc[0] == pytest.approx(-0.5)
    assert m.best_in_sample.iloc[0] == pytest.approx(1.5)  # the invalid j=2 is excluded


TINY = [
    "task.device=cpu",
    "task.pfn=pfn_smoke",
    "task.T_per_d=2",
    "task.candidates.n_sobol=64",
    "task.candidates.n_local=4",
    "task.api_candidates.n_global=8",
    "task.api_candidates.n_local_per_point=3",
    "task.label.n_tasks=12",
    "task.label.chunk_tasks=6",
    "task.label.n_labeled=5",
    "task.label.n_cont=2",
    "task.label.val_frac=0.3",
    "task.label.task_draw.n_dense=256",
    "task.label.task_draw.refine_steps=3",
    "task.head.width=32",
    "task.head.layers=1",
    "task.head.heads=2",
    "task.train.epochs=2",
    "task.calib.n_tasks=3",
    "task.test.n_tasks=3",
    "task.task_draw.n_dense=256",
    "task.task_draw.refine_steps=3",
    "task.chunk=64",
]


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=api_round1", *TINY])
    run = Run(tmp_path)
    api(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert {"labels", "calib", "test", "best", "decisions", "fstar"} <= set(tables)
    c_star = run.read()["summary"]["calib/c_star"]
    assert set(tables["test"].arm) == {
        "ei",
        "api_c0",
        "api_c0.5",
        "api_c1",
        "api_c2",
        "api_c3",
        "api_c4",
        f"null_c{c_star:g}",
    }
    assert len(tables["labels"]) == 12 * 5
    assert np.isfinite(tables["test"].norm_regret).all()
    assert (run.artifacts / "heads.pt").exists()

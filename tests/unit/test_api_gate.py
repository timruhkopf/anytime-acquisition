"""The model-based gate: belief continuations are paired across candidates, x0 has advantage 0, the selection
rule, the rollout policy's EI lanes are pi_0, and the pipeline end to end on a tiny label set."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from anytime_acq.analysis.api import gate_verdict
from anytime_acq.bo.engine import CandidateConfig, Engine
from anytime_acq.bo.runner import initial_design
from anytime_acq.bo.tasks import draw_tasks, task_seed
from anytime_acq.experimental.api import CandidateSpec, f_lanes, pi0, rollout
from anytime_acq.experimental.api_gate import belief_values, model_advantage, run_model_based
from anytime_acq.tasks.api import api
from anytime_acq.tasks.api_gate import api_gate, selected_gain
from anytime_acq.tracking.sink import Run

from .test_api import TINY

SPEC = CandidateSpec(n_global=8, n_top=2, n_local_per_point=3)


@pytest.fixture(scope="module")
def setup():
    pfn, _ = load("pfn_smoke")
    tasks = draw_tasks(2, [task_seed(4, 2, i) for i in range(3)], n_dense=512, refine_steps=5)
    return Engine(pfn, candidates=CandidateConfig(n_sobol=64, n_local=4), chunk=4), tasks


def test_belief_values_are_paired_across_candidates(setup) -> None:
    eng, tasks = setup
    X = initial_design(tasks, 3)
    Y = f_lanes(tasks, X.reshape(-1, 2), torch.arange(3).repeat_interleave(3)).view(3, 3)
    xq = torch.rand(3, 1, 2).expand(-1, 3, -1).contiguous()  # three identical candidates
    V = belief_values(eng, X, Y, torch.arange(3), xq, 0, 3, 2, torch.full((3,), 100), 3)
    assert V.shape == (3, 3, 2)
    torch.testing.assert_close(V[:, 0], V[:, 1])  # same candidate, same nuisance: same value
    torch.testing.assert_close(V[:, 0], V[:, 2])
    adv, se = model_advantage(V)
    torch.testing.assert_close(adv, torch.zeros_like(adv))  # up to fp32 batch-position effects
    torch.testing.assert_close(se, torch.zeros_like(se))


def test_selection_rule_scores_the_chosen_candidate() -> None:
    tab = pd.DataFrame(
        {
            "state": [0, 0, 0, 1, 1],
            "t": 0,
            "j": [0, 1, 2, 0, 1],
            "valid": [True, True, True, True, False],
            "adv_model": [0.0, 0.5, 2.0, 0.0, 9.0],
            "se_model": [0.0, 0.1, 3.0, 0.0, 0.0],
            "adv_true": [0.0, 1.0, -1.0, 0.0, 5.0],
        }
    )
    s0 = selected_gain(tab, 0.0).set_index("state")
    assert s0.loc[0, "j"] == 2  # the best model advantage
    assert s0.loc[0, "adv_true"] == -1.0
    assert s0.loc[1, "j"] == 0  # the invalid candidate is never chosen
    s1 = selected_gain(tab, 1.0).set_index("state")
    assert s1.loc[0, "j"] == 1  # 2 - 3 < 0.5 - 0.1


def test_model_based_policy_runs_and_its_ei_lanes_are_pi0(setup) -> None:
    eng, tasks = setup
    X0 = initial_design(tasks, 3)
    out = run_model_based(eng, SPEC, tasks, X0, 3, 4, 2, 0)
    assert out["F"].shape == (3, 2, 6)
    rows = torch.arange(3)
    _, F = rollout(tasks, X0, tasks.f(X0), rows, torch.zeros(3, dtype=torch.long), 0, 3, pi0(eng, 3))
    torch.testing.assert_close(out["F"][:, 0], F)


def test_gate_verdict() -> None:
    assert gate_verdict({"rel": -0.1, "hi": -0.01, "rel_lo": -0.2}) == "GO"
    assert gate_verdict({"rel": 0.0, "hi": 0.1, "rel_lo": -0.02}) == "NO-GO"
    assert gate_verdict({"rel": -0.04, "hi": 0.1, "rel_lo": -0.2}) == "INCONCLUSIVE"


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=api_round1", *TINY])
    src = Run(tmp_path / "src")
    api(cfg, src)
    rel = (src.artifacts / "labels.pt").relative_to(tmp_path)
    digest = hashlib.sha256((src.artifacts / "labels.pt").read_bytes()).hexdigest()
    gate_tiny = [o for o in TINY if o.split("=")[0].split(".")[1] not in ("label", "head", "train", "calib", "test")]
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=api_gate",
                *gate_tiny,
                f"paths.runs={tmp_path}",
                f"task.labels.path={rel}",
                f"task.labels.sha256={digest}",
                "task.states.n_states=6",
                "task.states.n_cont=2",
                "task.policy.n_tasks=2",
                "task.policy.n_labeled=4",
                "task.policy.n_cont=2",
            ],
        )
    run = Run(tmp_path / "gate")
    api_gate(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert {"gate_states", "test", "best", "decisions", "fstar"} <= set(tables)
    assert set(tables["test"].arm) == {"ei", "mb"}
    assert tables["gate_states"].state.nunique() == 6
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        bad = compose(
            config_name="config",
            overrides=[
                "experiment=api_gate",
                *gate_tiny,
                f"paths.runs={tmp_path}",
                f"task.labels.path={rel}",
                "task.labels.sha256=0",
            ],
        )
    with pytest.raises(ValueError, match="sha256"):
        api_gate(bad, Run(tmp_path / "bad"))

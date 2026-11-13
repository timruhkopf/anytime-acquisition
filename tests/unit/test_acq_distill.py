from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from pfn_repro.checkpoints import load

from anytime_acq.analysis.acq_distill import distill_verdict
from anytime_acq.bo.engine import Engine
from anytime_acq.experimental.acq_branch import (
    AcqBranch,
    BranchSpec,
    agreement,
    branch_scores,
    budget_features,
    extra_queries,
    rank_loss,
    teacher,
)
from anytime_acq.experimental.options import arm_pick
from anytime_acq.tasks.acq_distill import acq_distill
from anytime_acq.tasks.bo import _candidates
from anytime_acq.tracking.sink import Run

ARMS = ["ei", "pi0.01", "ucb0.995", "mes", "mean"]


def _engine() -> Engine:
    pfn = load("pfn_smoke")[0].eval()
    cand = _candidates(OmegaConf.create({"n_sobol": 64, "n_local": 4, "local_sigmas": [0.01, 0.2]}))
    return Engine(pfn, None, cand, chunk=4)


def test_teacher_argmax_is_the_arm_pick_and_agreement_of_identical_scores() -> None:
    eng = _engine()
    g = torch.Generator().manual_seed(0)
    X, Y = torch.rand(6, 7, 2, generator=g), torch.rand(6, 7, generator=g)
    xq, t, valid, _ = teacher(eng, ARMS, X, Y, extra_queries(X, Y, 16, g))
    assert xq.shape == (6, 64 + 8 + 16, 2) and t.shape == (6, 88, 5) and float(valid.float().mean()) > 0.9
    P0 = 72
    for a, arm in enumerate(ARMS):  # on the seed-0 candidates the teacher's best is the arm's own pick
        best = t[:, :P0, a].masked_fill(~valid[:, :P0], float("-inf")).max(1).values
        pick = arm_pick(eng, arm, X, Y, 0, 0)
        j = (xq[:, :P0] - pick[:, None]).abs().sum(-1).argmin(1)
        assert torch.allclose(t[torch.arange(6), j, a], best, rtol=1e-6, atol=0)
    ag = agreement(t, t, valid)
    assert bool((ag["top1"] == 1).all()) and float(ag["rank_pct"].max()) == 0.0
    assert float(ag["spearman"].min()) > 0.999
    worst = agreement(-t, t, valid)
    assert float(worst["rank_pct"].median()) > 0.5 and float(worst["spearman"].median()) < -0.9  # ties aside


def test_branch_learns_a_ranking() -> None:
    eng = _engine()
    pfn = eng.pfn
    assert pfn is not None
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(0)
    br = AcqBranch(64, 4, 6, 2, BranchSpec(width=32, heads=2))
    X, Y = torch.rand(8, 6, 1, generator=g), torch.rand(8, 6, generator=g)
    xq, t, valid, _ = teacher(eng, ["mean", "ucb0.995"], X, Y, extra_queries(X, Y, 32, g))
    b = budget_features(2, 10, 6, 8, "cpu")
    opt = torch.optim.Adam(br.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss, _ = rank_loss(branch_scores(br, eng, X, Y, xq, b), t, valid, g)
        first = first or float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.6 * first
    assert float(agreement(branch_scores(br, eng, X, Y, xq, b).detach(), t, valid)["spearman"].mean()) > 0.6
    assert all(not p.requires_grad or p.grad is None for p in pfn.parameters())  # the PFN gets no gradient


def test_verdict() -> None:
    def s(good: float, diff: float) -> pd.DataFrame:
        return pd.DataFrame({"arm": ["a", "b"], "cand0:good_pick": [0.99, good], "cl:diff": [0.0, diff]})

    assert distill_verdict(s(0.95, 0.02)) == "GO"
    assert distill_verdict(s(0.95, 0.3)) == "INCONCLUSIVE"
    assert distill_verdict(s(0.7, 0.0)) == "INCONCLUSIVE"
    assert distill_verdict(s(0.3, 0.0)) == "NO-GO"


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=acq_distill",
                "task.d=1",
                "task.device=cpu",
                "task.pfn=pfn_smoke",
                "task.shape.d1=[3,4]",
                "task.candidates.n_sobol=32",
                "task.candidates.n_local=2",
                "task.branch.width=32",
                "task.branch.heads=2",
                "task.train.tasks.n_tasks=6",
                "task.train.steps=4",
                "task.train.batch=4",
                "task.train.n_extra=8",
                "task.train.n_queries=24",
                "task.train.log_every=2",
                "task.eval.tasks.n_tasks=5",
                "task.eval.n_dense=16",
                "task.eval.n_examples=2",
                "task.eval.n_closed_loop=3",
                "task.task_chunk=4",
                "task.task_draw.n_dense=64",
                "task.task_draw.refine_steps=2",
                "task.n_boot=20",
                "task.chunk=8",
            ],
        )
    run = Run(tmp_path)
    acq_distill(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert {"train", "agree", "closed", "curves", "examples", "examples_ctx", "summary"} <= set(tables)
    st, cl, ex = tables["agree"], tables["closed"], tables["examples"]
    assert set(st.query_set) == {"cand0", "cand1", "dense"} and set(st.arm) == set(ARMS)
    assert len(st) == 6 * 4 * 3 * 5 and int(st.n_states.iloc[0]) == 5  # runs (5 arms + random) x steps x sets x arms
    assert len(cl) == 3 * 5 and bool((cl.regret_teacher >= 0).all()) and bool((cl.regret_student >= 0).all())
    assert {"teacher:ei", "student:mes", "f", "pfn_mean", "x0"} <= set(ex.columns)
    assert ex.example.nunique() == 2 * 2 * len({1, 4 // 3, 8 // 3})
    assert (run.artifacts / "branch.pt").exists()
    assert set(tables["curves"].step) == set(range(5))

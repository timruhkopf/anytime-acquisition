"""option_headroom: deterministic arms, the null mask, descent, cross-fitting removes selection luck, the rule,
and the pipeline end to end."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from anytime_acq.analysis.options import crossfit_per_state, option_verdict, static_per_state, with_eps
from anytime_acq.bo.engine import CandidateConfig, Engine
from anytime_acq.bo.tasks import draw_tasks
from anytime_acq.experimental.options import DescentSpec, arm_pick, descend, seeded_candidates
from anytime_acq.tasks.option_headroom import option_headroom
from anytime_acq.tracking.sink import Run


def _engine() -> Engine:
    return Engine(load("pfn_smoke")[0].eval(), None, CandidateConfig(64, 4, (0.01, 0.1)), base_seed=0, chunk=4)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return torch.rand(6, 5, 2, generator=g), torch.rand(6, 5, generator=g)


def test_arms_are_deterministic_per_seed() -> None:
    eng = _engine()
    X, Y = _data()
    assert torch.equal(seeded_candidates(eng, X, Y, 0), seeded_candidates(eng, X, Y, 0))
    assert not torch.equal(seeded_candidates(eng, X, Y, 0), seeded_candidates(eng, X, Y, 1))
    for arm in ("ei", "pi0.01", "ucb0.995", "mes", "mean", "ei@0"):
        assert torch.equal(arm_pick(eng, arm, X, Y, 0, 3), arm_pick(eng, arm, X, Y, 0, 3))
    # the null differs from EI somewhere (half of the candidates are masked)
    picks = [arm_pick(eng, f"ei@{j}", X, Y, 0, 3) for j in range(4)]
    assert any(not torch.equal(p, arm_pick(eng, "ei", X, Y, 0, 3)) for p in picks)


def test_descent_lowers_f() -> None:
    tasks = draw_tasks(2, [3, 4], n_dense=64, n_refine=1, refine_steps=1)
    x = torch.rand(10, 2, generator=torch.Generator().manual_seed(0))
    rows = torch.tensor([0, 1] * 5)
    f0 = tasks.f(x[:, None], rows)[:, 0]
    f1 = descend(tasks, x, rows, DescentSpec(steps=30, lr=0.01))
    assert bool((f1 <= f0 + 1e-7).all())


def _M(adv0: np.ndarray, adv1: np.ndarray) -> pd.DataFrame:
    rows = []
    for rep, A in ((0, adv0), (1, adv1)):
        for s in range(A.shape[0]):
            for o in range(A.shape[1]):
                rows.append({"state": s, "rep": rep, "option": f"o{o}_k3", "arm": f"a{o}", "k": 3, "adv": A[s, o]})
    return pd.DataFrame(rows)


def test_crossfit_removes_selection_luck() -> None:
    g = np.random.default_rng(0)
    pure_luck = crossfit_per_state(_M(g.normal(size=(400, 5)), g.normal(size=(400, 5))), 3, "arms")
    assert abs(pure_luck.H.mean()) < 0.1 and pure_luck.oracle.mean() > 0.8  # the in-sample best is biased
    signal = np.zeros((400, 5))
    signal[:, 2] = 1.0
    real = crossfit_per_state(_M(signal + 0.1 * g.normal(size=(400, 5)), signal), 3, "arms")
    assert real.H.mean() > 0.9


def test_static_part_and_eps() -> None:
    g = np.random.default_rng(1)
    signal = np.zeros((300, 4))
    signal[:, 1] = 1.0  # one option is better everywhere: all of the headroom is static
    M = _M(signal + 0.05 * g.normal(size=(300, 4)), signal + 0.05 * g.normal(size=(300, 4)))
    st = static_per_state(M, 3)
    H = crossfit_per_state(M, 3, "arms").H
    assert abs(float(st.mean()) - 1.0) < 0.05 and abs(float((H - st).mean())) < 0.05
    B = pd.DataFrame({"state": [0, 0], "rep": [0, 0], "option": ["base", "o"], "regret_bins": [0.0, 1.0]})
    assert with_eps(B, 1e-6).adv.iloc[1] == np.log10(1e-6) - np.log10(1e-3 + 1e-6)


def test_verdict() -> None:
    h = lambda H, lo, hi: pd.DataFrame([{"family": "arms", "k": 3, "H": H, "lo": lo, "hi": hi}])  # noqa: E731
    assert option_verdict(h(0.05, 0.01, 0.09), 3, 0.022) == "GO"
    assert option_verdict(h(0.005, -0.01, 0.015), 3, 0.022) == "NO-GO"
    assert option_verdict(h(0.02, -0.01, 0.05), 3, 0.022) == "INCONCLUSIVE"


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=option_headroom",
                "task.device=cpu",
                "task.pfn=pfn_smoke",
                "task.T_per_d=2",
                "task.candidates.n_sobol=32",
                "task.candidates.n_local=2",
                "task.ks=[1,2]",
                "task.primary_k=1",
                "task.part_a.n_tasks=3",
                "task.part_b.n_tasks=4",
                "task.part_b.t_max=2",
                "task.task_draw.n_dense=64",
                "task.task_draw.refine_steps=2",
                "task.descent.steps=5",
                "task.descent.check_steps=5",
                "task.n_boot=20",
                "task.chunk=8",
            ],
        )
    run = Run(tmp_path)
    option_headroom(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert {"part_a", "part_b", "headroom", "arms_b", "descent_check"} <= set(tables)
    A, B = tables["part_a"], tables["part_b"]
    assert len(A) == 3 * 5 * 2 * 5  # tasks x arms x seeds x (T + 1)
    n_opts = 1 + 4 * 2 + 2 * 2
    assert len(B) == 4 * n_opts * 2
    assert bool((B.loc[B.option == "base", "adv"] == 0).all())
    # options that never disagree with EI and are EI afterwards have advantage 0 exactly
    assert bool((B.loc[~B.disagree & (B.k == 1), "adv"].abs() < 1e-9).all())
    assert set(tables["headroom"].family) == {"arms", "nulls", "arms-nulls", "static", "arms-static"}

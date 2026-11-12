"""ei_opt: the fixed candidate set depends only on the data, refinement never lowers EI and is deterministic, the
decision rule, and the pipeline end to end on a tiny label set."""

import hashlib
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from anytime_acq.analysis.ei_opt import ei_opt_verdict
from anytime_acq.bo.engine import CandidateConfig, Engine
from anytime_acq.experimental.ei_opt import RefineSpec, fixed_candidates, pick
from anytime_acq.tasks.api import api
from anytime_acq.tasks.ei_opt import ei_opt
from anytime_acq.tracking.sink import Run

from .test_api import TINY


def _engine() -> Engine:
    return Engine(load("pfn_smoke")[0].eval(), None, CandidateConfig(64, 4, (0.01, 0.1)), base_seed=0, chunk=4)


def _data(L: int = 6, n: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return torch.rand(L, n, 2, generator=g), torch.rand(L, n, generator=g)


def test_fixed_candidates_depend_only_on_the_data() -> None:
    eng = _engine()
    X, Y = _data()
    a, b = fixed_candidates(eng, X, Y), fixed_candidates(eng, X, Y)
    assert torch.equal(a, b) and a.shape == (6, 64 + 8, 2)
    assert torch.equal(fixed_candidates(eng, X[:2], Y[:2]), a[:2])


def test_refinement_never_lowers_ei_and_is_deterministic() -> None:
    eng = _engine()
    X, Y = _data()
    rows, spec = torch.arange(6), RefineSpec(k=3, steps=10, lr=0.01)
    for base in ("sampled", "fixed"):
        reps = torch.zeros(6, dtype=torch.long)
        _, e0 = pick(eng, base, X, Y, rows, reps, 1, 6, spec)
        x1, e1 = pick(eng, base + "+refine", X, Y, rows, reps, 1, 6, spec)
        assert bool((e1 >= e0 - 1e-7).all()) and bool(((x1 >= 0) & (x1 <= 1)).all())
    xa, _ = pick(eng, "fixed+refine", X, Y, rows, torch.zeros(6, dtype=torch.long), 1, 6, spec)
    xb, _ = pick(eng, "fixed+refine", X, Y, rows, torch.ones(6, dtype=torch.long), 3, 6, spec)
    assert torch.equal(xa, xb)


def _summary(det: float) -> pd.DataFrame:
    return pd.DataFrame([{"variant": "fixed+refine", "share_tasks_deterministic": det}])


def _paired(lo: float, hi: float) -> pd.DataFrame:
    return pd.DataFrame([{"a": "fixed+refine", "metric": "norm_regret", "rel_lo": lo, "rel_hi": hi}])


def test_verdict() -> None:
    assert ei_opt_verdict(_summary(1.0), _paired(-0.1, 0.02)) == "GO"
    assert ei_opt_verdict(_summary(1.0), _paired(-0.1, 0.2)) == "INCONCLUSIVE"
    assert ei_opt_verdict(_summary(1.0), _paired(0.06, 0.2)) == "NO-GO"
    assert ei_opt_verdict(_summary(0.9), _paired(-0.1, 0.02)) == "NO-GO"


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=api_round1", *TINY])
    src = Run(tmp_path / "src")
    api(cfg, src)
    rel = (src.artifacts / "labels.pt").relative_to(tmp_path)
    digest = hashlib.sha256((src.artifacts / "labels.pt").read_bytes()).hexdigest()
    keep = ("device", "pfn", "T_per_d", "candidates")
    tiny = [o for o in TINY if o.split("=")[0].split(".")[1] in keep]
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=ei_opt",
                *tiny,
                f"paths.runs={tmp_path}",
                f"task.labels.path={rel}",
                f"task.labels.sha256={digest}",
                "task.states.n_states=8",
                "task.states.task_draw.n_dense=64",
                "task.rollouts.n_tasks=3",
                "task.rollouts.task_draw.n_dense=64",
                "task.rollouts.task_draw.refine_steps=2",
                "task.refine.k=2",
                "task.refine.steps=5",
                "task.n_reps=2",
                "task.group=4",
                "task.chunk=4",
            ],
        )
    run = Run(tmp_path / "eo")
    ei_opt(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert {"states", "final", "spread", "summary", "paired"} <= set(tables)
    assert len(tables["final"]) == 3 * 4 * 2 and len(tables["states"]) == 8 * 4 * 2
    st = tables["states"]
    assert bool((st.ei_ratio <= 1 + 1e-6).all())
    s = tables["summary"].set_index("variant")
    assert s.loc["fixed+refine", "share_tasks_deterministic"] == 1.0
    assert s.loc["fixed", "share_tasks_deterministic"] == 1.0

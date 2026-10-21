"""The bo pipeline end to end on the CPU with the smoke PFN: tables, shapes, and the null's bookkeeping."""

from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_module

from anytime_acq.tasks.bo import bo
from anytime_acq.tracking.sink import Run

TINY = [
    "task.device=cpu",
    "task.pfn=pfn_smoke",
    "task.n_tasks=3",
    "task.T_per_d=2",
    "task.segment_per_d=1",
    "task.n_rollouts=2",
    "task.null_k=2",
    "task.candidates.n_sobol=64",
    "task.candidates.n_local=4",
    "task.task_draw.n_dense=256",
    "task.task_draw.refine_steps=5",
    "task.d=2",
]


class Out:
    def __init__(self, run: Run) -> None:
        self.summary = run.read()["summary"]
        tables = run.artifacts / "tables"
        self.tables = {p.stem: pd.read_parquet(p) for p in tables.glob("*.parquet")}


def _run(overrides: list[str], tmp_path: Path) -> Out:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=static_tune", *TINY, *overrides])
    run = Run(tmp_path)
    bo(cfg, run)
    return Out(run)


def test_static_runs_every_arm_and_random_search(tmp_path: Path) -> None:
    run = _run(["task.arms=[ei,ucb0.9]", "task.null_reference=true", "task.base=ei"], tmp_path)
    best = run.tables["best"]
    assert set(best.arm) == {"ei", "ucb0.9", "ei@0", "random"}
    assert len(best) == 4 * 3 * (2 * 2 + 1)  # arms x tasks x (T + 1)
    assert (best.groupby(["arm", "task"]).best_f.diff().dropna() <= 0).all()  # best-so-far
    fstar = run.tables["fstar"].set_index("task").fstar
    assert (best.groupby("task").best_f.min() >= fstar - 1e-6).all()


@pytest.mark.parametrize("method", ["privileged", "belief", "equal_arms"])
def test_controllers_log_decisions(method: str, tmp_path: Path) -> None:
    run = _run([f"task.method={method}", "task.arms=[ei,ucb0.9]", "task.base=ei"], tmp_path)
    dec = run.tables["decisions"]
    assert len(dec) == 3 * 2  # tasks x decisions (T=4, segment=2)
    expected = {"ei@0", "ei@1"} if method == "equal_arms" else {"ei", "ucb0.9"}
    assert set(dec.chosen) <= expected
    assert set(run.tables["best"].arm) == {f"ctrl:{method}"}


def test_controller_needs_the_base_arm_first(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="base arm first"):
        _run(["task.method=privileged", "task.arms=[ucb0.9,ei]", "task.base=ei"], tmp_path)

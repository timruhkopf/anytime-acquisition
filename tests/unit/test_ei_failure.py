from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_module

from anytime_acq.analysis.ei_failure import NEVER, gate_choice, gate_value, spearman_ci
from anytime_acq.tasks.ei_failure import ei_failure
from anytime_acq.tracking.sink import Run


def test_gate_choice_and_value() -> None:
    rng = np.random.default_rng(0)
    N = 400
    score = rng.normal(size=(N, 2))
    A = np.zeros((N, 2, 2))
    A[:, 1, 0] = np.where(score[:, 1] > 0, 1.0, -1.0)  # option 0 at step 1 pays iff the score is high
    A[:, 0, 1] = 0.1  # a static schedule worth 0.1
    j, o, c = gate_choice(score, A)
    assert (j, o) == (1, 0) and abs(c) < 0.3
    v, m = gate_value(score, A, (j, o, c))
    assert v.mean() > 0.4 and 0.3 < m.mean() < 0.7
    # an uninformative score: the ungated static schedule (c = -inf), and never deviating if nothing pays
    j, o, c = gate_choice(rng.normal(size=(N, 2)), np.where(A == 0.1, 0.1, 0.0))
    assert (j, o, c) == (0, 1, -np.inf)
    assert gate_choice(score, -np.abs(A)) == NEVER
    assert gate_value(score, A, NEVER)[0].sum() == 0


def test_spearman_ci() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    a, b = y + 0.5 * rng.normal(size=300), y + 2.0 * rng.normal(size=300)
    rho, lo, hi = spearman_ci(a, y, n_boot=100)
    assert lo < rho < hi and lo > 0.7
    d, lo, hi = spearman_ci(a, y, b, n_boot=100)
    assert lo > 0 and lo < d < hi


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=ei_failure",
                "task.device=cpu",
                "task.pfn=pfn_smoke",
                "task.T_per_d=3",
                "task.candidates.n_sobol=32",
                "task.candidates.n_local=2",
                "task.ks=[1]",
                "task.steps=[0,2]",
                "task.tasks.n_tasks=12",
                "task.task_chunk=7",
                "task.task_draw.n_dense=64",
                "task.task_draw.refine_steps=2",
                "task.n_folds=3",
                "task.n_inner=2",
                "task.z_pca=4",
                "task.fit.epochs=5",
                "task.fit.n_ens=1",
                "task.n_boot=20",
                "task.n_boot_corr=10",
                "task.chunk=8",
            ],
        )
    run = Run(tmp_path)
    ei_failure(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert {"summary", "probe", "probe_diff", "strata", "gates", "per_task", "options"} <= set(tables)
    summ, probe, per = tables["summary"], tables["probe"], tables["per_task"]
    assert {"static", "gate/belief/mlp", "gate/pfn/ridge", "gate/priv:other_seed"} <= set(summ.variant)
    assert set(probe.score) >= {"data/ridge", "pfn/mlp", "priv:other_seed", "priv:current_regret"}
    assert len(per) == 12 and set(probe.step) == {0, 2}
    s = summ[(summ.variant == "gate/belief/mlp") & (summ.seed == 0)].set_index("contrast")
    d = per["gate/belief/mlp:v0"] - per["static:v0"]
    assert abs(float(s.loc["vs_static", "mean"]) - float(d.mean())) < 1e-9
    # a task that does not deviate has value 0
    assert bool((per.loc[~per["gate/belief/mlp:deviates"], "gate/belief/mlp:v0"] == 0).all())

"""The readout's invariants: a normalized density, a frozen PFN, consistent targets, an end-to-end run."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from anytime_acq.bo.engine import CandidateConfig, Engine
from anytime_acq.readout.data import ei_at, ei_targets, make_states
from anytime_acq.readout.expert import ActionExpert, log_prob, mode
from anytime_acq.tasks.ei_readout import ei_readout
from anytime_acq.tracking.sink import Run


@pytest.fixture(scope="module")
def pfn():
    return load("pfn_smoke")[0].eval()


def test_mixture_density_integrates_to_one() -> None:
    torch.manual_seed(0)
    mix = (torch.randn(1, 3).log_softmax(-1), torch.randn(1, 3, 1), 0.3 * torch.randn(1, 3, 1))
    x = torch.linspace(1e-4, 1 - 1e-4, 20001)[None, :, None]
    p = log_prob(mix, x).exp()[0]
    assert float(torch.trapezoid(p, x[0, :, 0])) == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("trunk", ["pfn_all", "pfn_last", "d_only"])
def test_expert_trains_only_its_own_weights(pfn, trunk: str) -> None:
    ex = ActionExpert(trunk, pfn, d_expert=16, n_heads=2, n_mix=4, n_probe=8 if trunk == "pfn_all" else 0)
    X, Y = torch.rand(3, 5, 2), torch.rand(3, 5)
    m = torch.tensor([[True] * 5, [True] * 3 + [False] * 2, [True] + [False] * 4])
    mix = ex(X, Y, m, torch.ones(3))
    log_prob(mix, torch.rand(3, 7, 2)).sum().backward()
    assert all(p.grad is None for p in pfn.parameters())  # frozen
    if trunk == "d_only":
        assert any(p.grad is not None for p in ex.encoder.parameters())
    assert mode(mix).shape == (3, 2)


def test_padding_is_invisible(pfn) -> None:
    """A padded state gives the same output as the unpadded one."""
    ex = ActionExpert("pfn_all", pfn, d_expert=16, n_heads=2, n_mix=4).eval()
    X, Y = torch.rand(1, 3, 2), torch.rand(1, 3)
    a = ex(X, Y, torch.ones(1, 3, dtype=torch.bool), torch.ones(1))
    Xp, Yp = torch.cat([X, torch.rand(1, 2, 2)], 1), torch.cat([Y, torch.rand(1, 2)], 1)
    b = ex(Xp, Yp, torch.tensor([[True] * 3 + [False] * 2]), torch.ones(1))
    for u, v in zip(a, b, strict=True):
        torch.testing.assert_close(u, v, atol=1e-5, rtol=1e-4)


def test_targets_are_normalized_and_peak_at_max_ei(pfn) -> None:
    eng = Engine(pfn, candidates=CandidateConfig(n_sobol=64, n_local=4))
    st = make_states(eng, 2, 9, 3, 2, 3, "cpu")
    tg = ei_targets(pfn, st, 0.03, 16, CandidateConfig(256, 8, (0.01, 0.05)), seed=0)
    torch.testing.assert_close(tg.w.sum(1), torch.ones(len(st)))
    assert bool((tg.mass_top <= 1 + 1e-6).all())
    torch.testing.assert_close(ei_at(pfn, st, tg.x[:, 0]), tg.max_ei, rtol=1e-4, atol=1e-7)
    assert int(st.mask.sum(1).min()) >= 1


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    tiny = [
        "task.device=cpu",
        "task.pfn=pfn_smoke",
        "task.d_expert=16",
        "task.n_heads=2",
        "task.n_mix=4",
        "task.data.n_tasks_train=4",
        "task.data.n_tasks_test=2",
        "task.data.T_per_d=1",
        "task.data.n_sobol=64",
        "task.data.n_local=4",
        "task.data.n_top=8",
        "task.steps=3",
        "task.batch=4",
        "task.val_every=2",
        "task.val_size=4",
        "task.closed_loop.n_tasks=2",
        "task.closed_loop.T_per_d=1",
        "task.closed_loop.candidates.n_sobol=32",
        "task.closed_loop.candidates.n_local=2",
        f"paths.data={tmp_path}",
    ]
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=ei_readout", *tiny])
    run = Run(tmp_path / "run")
    ei_readout(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert set(tables) == {"ei_ratio", "closed_loop", "cheap"}
    assert set(tables["closed_loop"].method) == {"ei", "expert_mode", "expert_sample"}
    assert len(list((tmp_path / "readout").glob("*.pt"))) == 1  # cached once

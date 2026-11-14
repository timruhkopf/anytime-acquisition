"""The 0006 value decoder: bar distribution, query independence (no candidate-candidate attention), the
decision rule, the EI lanes are pi_0, and the pipeline end to end on a tiny label set."""

import hashlib
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from anytime_acq.bo.engine import CandidateConfig, Engine
from anytime_acq.bo.runner import initial_design
from anytime_acq.bo.tasks import draw_tasks, task_seed
from anytime_acq.experimental.api import CandidateSpec, pi0, rollout
from anytime_acq.experimental.decoder import Bars, ValueDecoder, decoder_pick, run_decoder
from anytime_acq.tasks.api import api
from anytime_acq.tasks.api_decoder import api_decoder
from anytime_acq.tracking.sink import Run

from .test_api import TINY

SPEC = CandidateSpec(n_global=8, n_top=2, n_local_per_point=3)


def _bars() -> Bars:
    a = torch.cat([torch.zeros(500), torch.randn(2000, generator=torch.Generator().manual_seed(0)) ** 3])
    return Bars.from_targets(a, 32, 1e-3)


def test_bars() -> None:
    bars = _bars()
    b = bars.borders
    assert (b.diff() > 0).all()
    assert ((b == -1e-3) | (b == 1e-3)).sum() == 2  # the zero bin
    assert bars.n_bins == 31
    logits = torch.randn(3, 5, bars.n_bins)
    # the density integrates to 1: density at each bin's midpoint times the bin width (piecewise uniform)
    mid = (b[1:] + b[:-1]) / 2
    dens = (-bars.nll(logits[:1, :1].expand(len(mid), 1, -1)[:, 0], mid)).exp()
    torch.testing.assert_close((dens * b.diff()).sum(), torch.tensor(1.0))
    # quantiles are monotone in the level, and the median of a one-bin distribution is the bin's midpoint
    q = torch.stack([bars.quantile(logits, lv) for lv in (0.1, 0.5, 0.9)])
    assert (q.diff(dim=0) >= 0).all()
    one = torch.full((1, bars.n_bins), -1e9)
    one[0, 7] = 0
    torch.testing.assert_close(bars.quantile(one, 0.5), (b[7] + b[8])[None] / 2)
    torch.testing.assert_close(bars.mean(one), (b[7] + b[8])[None] / 2)


def test_queries_are_independent() -> None:
    torch.manual_seed(0)
    dec = ValueDecoder(7, 5, 16, _bars(), width=32, blocks=2, heads=2).eval()
    q, ctx, glob, bud = torch.randn(2, 6, 7), torch.randn(2, 2, 9, 16), torch.randn(2, 5), torch.rand(2, 4)
    full = dec(q, ctx, glob, bud, q[:, 0])
    part = dec(q[:, [0, 3]], ctx, glob, bud, q[:, 0])
    torch.testing.assert_close(full[:, [0, 3]], part)  # a score does not depend on the other candidates


def test_decoder_pick() -> None:
    bars = _bars()
    nb = bars.n_bins
    mid = (bars.borders[1:] + bars.borders[:-1]) / 2
    logits = torch.full((3, 3, nb), -1e9)
    pos, zero = int((mid > 0.5).nonzero()[0]), int((mid.abs() < 1e-3).nonzero()[0])
    logits[:, :, zero] = 0  # every candidate at 0
    logits[:, 2, zero], logits[:, 2, pos] = -1e9, 0  # candidate 2 clearly better
    valid = torch.ones(3, 3, dtype=torch.bool)
    j, dev, _ = decoder_pick(bars, logits, valid, ["q0.1", "ei", "mean"])
    assert j.tolist() == [2, 0, 2]
    assert dev.tolist() == [True, False, True]
    valid[:, 2] = False
    j, dev, _ = decoder_pick(bars, logits, valid, ["q0.1", "ei", "mean"])
    assert j.tolist() == [0, 0, 0]  # nothing above the zero bin


def test_ei_lanes_are_pi0() -> None:
    pfn, _ = load("pfn_smoke")
    tasks = draw_tasks(2, [task_seed(4, 2, i) for i in range(3)], n_dense=512, refine_steps=5)
    eng = Engine(pfn, candidates=CandidateConfig(n_sobol=64, n_local=4), chunk=4)
    torch.manual_seed(0)
    dm = pfn.blocks[0].attn.h * pfn.blocks[0].attn.dk
    from anytime_acq.experimental.api import N_GLOBAL_FEATURES, SOURCES

    dec = ValueDecoder(2 + 11 + len(SOURCES), N_GLOBAL_FEATURES, dm, _bars(), width=16, blocks=2, heads=2).eval()
    X0 = initial_design(tasks, 3)
    out = run_decoder(eng, SPEC, dec, tasks, X0, 3, ["mean", "ei"], [1, 2])
    assert out["F"].shape == (3, 2, 6)
    assert not out["deviated"][:, 1].any()
    _, F = rollout(tasks, X0, tasks.f(X0), torch.arange(3), torch.zeros(3, dtype=torch.long), 0, 3, pi0(eng, 3))
    torch.testing.assert_close(out["F"][:, 1], F)


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=api_round1", *TINY])
    src = Run(tmp_path / "src")
    api(cfg, src)
    rel = (src.artifacts / "labels.pt").relative_to(tmp_path)
    digest = hashlib.sha256((src.artifacts / "labels.pt").read_bytes()).hexdigest()
    keep = [o for o in TINY if o.split("=")[0].split(".")[1] not in ("head", "train")]
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=api_decoder",
                *keep,
                f"paths.runs={tmp_path}",
                f"task.labels.path={rel}",
                f"task.labels.sha256={digest}",
                "task.label.n_tasks=6",
                "task.label.task_offset=12",
                "task.decoder.read=[1,2]",
                "task.decoder.width=16",
                "task.decoder.blocks=2",
                "task.decoder.heads=2",
                "task.decoder.n_bins=16",
                "task.train.epochs=2",
                "task.group=5",
            ],
        )
    run = Run(tmp_path / "dec")
    api_decoder(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert {"val", "calib", "test", "best", "decisions", "fstar"} <= set(tables)
    assert set(tables["test"].arm) == {"ei", "dec_q0.1", "dec_q0.3", "dec_q0.5", "dec_mean"} | {
        a for a in tables["test"].arm if a.startswith("null_")
    }
    assert sum(a.startswith("null_") for a in set(tables["test"].arm)) == 1
    assert (run.artifacts / "labels.pt").exists()
    assert run.read()["summary"]["labels/n_states"] == 18.0

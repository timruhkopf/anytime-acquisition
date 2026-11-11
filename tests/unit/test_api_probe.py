"""Readout probes: the per-layer states reproduce the PFN's own forward pass, the probe metrics, the verdict
rule, and the pipeline end to end on a tiny label set."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from anytime_acq.analysis.api import probe_verdict
from anytime_acq.bo.engine import CandidateConfig, Engine
from anytime_acq.experimental.api import pfn_pass, query_states
from anytime_acq.experimental.probes import (
    auc,
    context_states,
    query_layer_states,
    ridge_fit_predict,
    within_spearman,
)
from anytime_acq.tasks.api import api
from anytime_acq.tasks.api_probe import api_probe
from anytime_acq.tracking.sink import Run

from .test_api import TINY


def test_layer_states_match_the_forward_pass() -> None:
    pfn, _ = load("pfn_smoke")
    pfn.eval()
    torch.manual_seed(0)
    X, Y, xq = torch.rand(3, 5, 2), torch.rand(3, 5), torch.rand(3, 4, 2)
    Z = context_states(pfn, X, Y)
    assert Z.shape[:3] == (3, len(pfn.blocks) + 1, 5)
    Q = query_layer_states(pfn, X, Y, xq)
    assert Q.shape[:3] == (3, 4, len(pfn.blocks) + 1)
    nf = torch.full((3,), 2, dtype=torch.long)
    kv = pfn.context_kv(X, Y, n_features=nf)
    torch.testing.assert_close(Q[:, :, -1], query_states(pfn, kv, xq, nf))
    eng = Engine(pfn, candidates=CandidateConfig(n_sobol=64, n_local=4), chunk=4)
    lg, _ = pfn_pass(eng, X, Y, xq, with_z=False)
    assert lg.shape[:2] == (3, 4)


def test_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    assert auc(np.array([0.1, 0.2, 0.3, 0.4]), y) == 1.0
    assert auc(np.array([0.4, 0.3, 0.2, 0.1]), y) == 0.0
    g = np.array([0, 0, 0, 1, 1, 1])
    rho = within_spearman(np.array([1, 2, 3, 3, 2, 1.0]), np.array([1, 2, 3, 1, 2, 3.0]), g)
    np.testing.assert_allclose(rho, [1.0, -1.0])
    X = torch.randn(400, 5)
    w = torch.tensor([1.0, -2, 0, 0, 0.5])
    p, _ = ridge_fit_predict(X[:300], X[:300] @ w, X[300:])
    torch.testing.assert_close(p, X[300:] @ w, atol=0.05, rtol=0.05)


def _probes(lo_all: float, lo_last: float) -> pd.DataFrame:
    rows = []
    for tg, m, chance in (("headroom_bin", "auc", 0.5), ("adv", "within_spearman", 0.0)):
        for inp, lo in (("all", lo_all), ("last", lo_last), ("hand", chance + 0.1)):
            rows.append({"target": tg, "input": inp, "model": "ridge", "metric": m, "lo": chance + lo})
    return pd.DataFrame(rows)


def _contrasts(lo: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"target": tg, "model": "ridge", "a": "all", "b": "hand", "lo": lo} for tg in ("headroom_bin", "adv")]
    )


def test_probe_verdict() -> None:
    assert probe_verdict(_probes(0.05, 0.0), _contrasts(0.01)) == "GO"
    assert probe_verdict(_probes(0.05, 0.0), _contrasts(-0.01)) == "INCONCLUSIVE"
    assert probe_verdict(_probes(-0.05, -0.05), _contrasts(-0.01)) == "NO-GO"


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=api_round1", *TINY])
    src = Run(tmp_path / "src")
    api(cfg, src)
    rel = (src.artifacts / "labels.pt").relative_to(tmp_path)
    digest = hashlib.sha256((src.artifacts / "labels.pt").read_bytes()).hexdigest()
    keep = ("device", "pfn", "T_per_d", "candidates", "api_candidates", "chunk")
    probe_tiny = [o for o in TINY if o.split("=")[0].split(".")[1] in keep]
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=api_probe",
                *probe_tiny,
                f"paths.runs={tmp_path}",
                f"task.labels.path={rel}",
                f"task.labels.sha256={digest}",
                "task.n_states=12",
                "task.kg_n_y=2",
                "task.test_frac=0.4",
                "task.n_boot=5",
                "task.group=5",
                "task.draw_chunk=6",
                "task.task_draw.n_dense=256",
                "task.task_draw.refine_steps=3",
            ],
        )
    run = Run(tmp_path / "probe")
    api_probe(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert {"probes", "contrasts", "reliability", "targets"} <= set(tables)
    pr = tables["probes"]
    assert set(pr.input) == {"hand", "last", "all"}
    assert {"headroom_bin", "adv", "kg", "log_gap", "xstar"} <= set(pr.target)
    assert {"auc", "r2", "spearman", "within_spearman"} <= set(pr.metric)
    assert len(tables["targets"]) == 12

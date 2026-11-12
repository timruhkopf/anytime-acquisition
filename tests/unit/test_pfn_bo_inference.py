"""PFN inference on BO contexts: scoring is exact for draws from the predictive, the decision rule, and the
pipeline end to end on a tiny label set."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_module
from pfn_repro.objectives.bar_distribution import BarDistribution, uniform_bin_borders

from anytime_acq.analysis.inference import inference_verdict
from anytime_acq.experimental.pfn_inference import ece, event_prob, ks_uniform, local_points, score_points
from anytime_acq.tasks.api import api
from anytime_acq.tasks.pfn_bo_inference import gap_stratum, pfn_bo_inference
from anytime_acq.tracking.sink import Run

from .test_api import TINY


def test_scores_of_draws_from_the_predictive_are_calibrated() -> None:
    borders = uniform_bin_borders(50)
    bd = BarDistribution(borders)
    widths = borders[1:] - borders[:-1]
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 50, generator=g).expand(20000, -1)
    p = torch.softmax(logits[0], -1)
    b = torch.multinomial(p, 20000, replacement=True, generator=g)
    y = borders[b] + torch.rand(20000, generator=g) * widths[b]
    ld, pit = score_points(bd, logits, y, torch.rand(20000, generator=g))
    assert torch.allclose(ld, -bd(logits, y), atol=1e-5)
    assert ks_uniform(pit.numpy()) < 0.02
    thr = torch.full((20000,), 0.37)
    pe = event_prob(bd, logits[:, None], thr[:, None])[:, 0]
    assert abs(float(pe.mean()) - float((y < 0.37).float().mean())) < 0.02
    assert ece(pe.numpy(), (y < 0.37).numpy().astype(float)) < 0.02


def test_local_points_stay_in_the_box_and_gap_strata() -> None:
    x = local_points(torch.tensor([[0.0, 1.0]]), torch.randn(1, 30, 2) * 10, (0.01, 0.2))
    assert x.shape == (1, 30, 2) and bool(((x >= 0) & (x <= 1)).all())
    assert list(gap_stratum(np.array([0.0, 0.99, 1.0, 50.0, 1e4]))) == ["<1", "<1", "1-10", "10-100", ">100"]


def _tables(cov90: float, cov_ci: float, ece_diff: float, ece_lo: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, crows = [], []
    for pt in ("all", "ei_pick"):
        for des in ("bo", "random"):
            for m, v in (("cov90", cov90), ("cov50", 0.5)):
                rows.append({"design": des, "ptype": pt, "metric": m, "value": v, "lo": v - cov_ci, "hi": v + cov_ci})
            rows.append({"design": des, "ptype": pt, "metric": "ece_k0", "value": 0.01, "lo": 0.005, "hi": 0.015})
        crows.append({"ptype": pt, "metric": "ece_k0", "diff": ece_diff, "lo": ece_lo, "hi": ece_diff + 0.005})
    return pd.DataFrame(rows), pd.DataFrame(crows)


def test_inference_verdict() -> None:
    assert inference_verdict(*_tables(0.80, 0.01, 0.0, -0.01), [0]) == "GO"  # undercoverage
    assert inference_verdict(*_tables(0.90, 0.01, 0.05, 0.03), [0]) == "GO"  # event miscalibration
    assert inference_verdict(*_tables(0.90, 0.01, 0.0, -0.01), [0]) == "NO-GO"
    assert inference_verdict(*_tables(0.90, 0.05, 0.0, -0.01), [0]) == "INCONCLUSIVE"  # too wide


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=api_round1", *TINY])
    src = Run(tmp_path / "src")
    api(cfg, src)
    rel = (src.artifacts / "labels.pt").relative_to(tmp_path)
    digest = hashlib.sha256((src.artifacts / "labels.pt").read_bytes()).hexdigest()
    keep = ("device", "pfn", "T_per_d", "candidates", "chunk")
    tiny = [o for o in TINY if o.split("=")[0].split(".")[1] in keep]
    with initialize_config_module(config_module="anytime_acq.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=pfn_bo_inference",
                *tiny,
                f"paths.runs={tmp_path}",
                f"task.labels.path={rel}",
                f"task.labels.sha256={digest}",
                "task.n_states=12",
                "task.points.n_uniform=3",
                "task.points.n_local=3",
                "task.n_boot=5",
                "task.group=5",
                "task.draw_chunk=6",
                "task.task_draw.n_dense=64",
            ],
        )
    run = Run(tmp_path / "pbi")
    pfn_bo_inference(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert {"points", "cands", "states", "calibration", "contrasts", "strata"} <= set(tables)
    P = tables["points"]
    assert len(P) == 12 * 2 * 7 and set(P.design) == {"bo", "random"}
    assert bool(P.pit.between(0, 1).all())
    # both designs are scored at the same points: the truth agrees row by row
    bo, rd = P[P.design == "bo"].reset_index(drop=True), P[P.design == "random"].reset_index(drop=True)
    assert np.allclose(bo.f, rd.f) and (bo.state == rd.state).all()
    assert len(tables["states"]) == 12
    assert {"reliability", "headroom", "spearman_log_ei", "frac_adv_below_bin"} <= set(tables["strata"].metric)

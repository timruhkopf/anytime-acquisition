"""Predictive quality of the frozen PFN on BO contexts, and the label/headroom structure by resolution
(research_log/decisions/0007 §4, tests 1 and 2; pre-registration: notebooks/10_2026-11-12_pfn_bo_inference.ipynb).

One member = one d. At the labeled states of a round-1 label set (pinned by sha256):
- test 2 (E2): the PFN predictive given (a) the state's BO context and (b) a random-design context of the same
  task and the same n, scored against the noise-free f at held-out points (uniform, local around the BO
  incumbent, EI's pick at the state). Per point: log density of f(x)'s bin, randomized PIT, and event
  probabilities P(f(x) < incumbent - k bins) with their outcomes, k in `event_bins`, the incumbent being the
  context's own.
- test 1 (E3/E4): per labeled candidate, the label and PFN scores (log EI, PI) at the BO context; per state, the
  gap to f* in PFN bins.

Artifact `points.parquet` (one row per state x design x point). Tables: `pit_hist`, `reliability`, `by_t`,
`by_stratum` (aggregates of the points), `cands` (one row per labeled candidate), `states`,
`calibration` (by design and point type, bootstrap CIs over states), `contrasts` (BO - random, paired over
states), `strata` (test 1, by gap stratum).
"""

import time
from typing import cast

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load
from pfn_repro.objectives.bar_distribution import BarDistribution

from ..analysis.inference import inference_verdict
from ..bo import nuisance
from ..bo.engine import Engine
from ..bo.tasks import _select, draw_tasks
from ..experimental.api import base_picks, pfn_pass
from ..experimental.pfn_inference import ece, event_prob, ks_uniform, local_points, score_points
from ..experimental.probes import within_spearman
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .api import _dict
from .api_gate import load_labels
from .api_probe import crossfit_headroom
from .bo import _candidates

DESIGNS = ("bo", "random")
GAP_EDGES = (0.0, 1.0, 10.0, 100.0, float("inf"))  # in PFN bins


def gap_stratum(gap_bins: np.ndarray) -> np.ndarray:
    labels = np.array(["<1", "1-10", "10-100", ">100"])
    return labels[np.clip(np.searchsorted(GAP_EDGES, gap_bins, side="right") - 1, 0, 3)]


@torch.no_grad()
def collect(cfg: DictConfig, engine: Engine, L: dict, S: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = cfg.task
    dev = t.device
    d, n_init, T = int(t.d), int(t.n_init_per_d * t.d), int(t.T_per_d * t.d)
    assert engine.pfn is not None
    bd = cast(BarDistribution, engine.pfn.bar_dist)
    w = float(cast(torch.Tensor, bd.bucket_widths).mean().item())
    Ku, Kl = int(t.points.n_uniform), int(t.points.n_local)
    ks = [int(k) for k in t.event_bins]
    base = int(cfg.seed)
    pts, cands, sts = [], [], []
    sigmas = tuple(float(v) for v in t.points.local_sigmas)
    fcheck = 0.0
    for c0 in range(0, S, int(t.draw_chunk)):
        ii = torch.arange(c0, min(c0 + int(t.draw_chunk), S))
        seeds = L["seed"][ii].tolist()
        with torch.enable_grad():  # draw_tasks refines f* by gradient descent
            tasks = draw_tasks(d, seeds, **_dict(t.task_draw)).to(dev)
        bt_all = L["branch_t"][ii]
        for tb in bt_all.unique().tolist():
            loc = (bt_all == tb).nonzero().squeeze(1)  # rows within the chunk (= task rows of `tasks`)
            for s in range(0, len(loc), int(t.group)):
                r = loc[s : s + int(t.group)]
                g = ii[r]  # state indices
                G, n = len(g), n_init + tb
                rows_t = r.to(dev)
                gs = g.to(dev)
                X, F = L["X"][g, :n].to(dev), L["F"][g, :n].to(dev)
                # random design: same task, same n (nested across n: the first n of a fixed per-state sequence)
                Xr = nuisance.stream(
                    base, "pbi_design", 0, torch.zeros(G, dtype=torch.long), gs, (n_init + T, d), S, False, dev
                )[:, :n]
                Fr = tasks.f(Xr, rows_t)
                fcheck = max(fcheck, float((tasks.f(X, rows_t) - F).abs().max()))
                # held-out points per context, each chosen from that context's own data only (ignorable):
                # uniform (shared), local around the context's incumbent, the context's EI pick
                zeros = torch.zeros(G, dtype=torch.long)
                U = nuisance.stream(base, "pbi_query", 0, zeros, gs, (Ku, d), S, False, dev)
                z = nuisance.stream(base, "pbi_query", 1, zeros, gs, (Kl, d), S, True, dev)
                ptype = np.array(["uniform"] * Ku + ["local"] * Kl + ["ei_pick"])
                ar = torch.arange(G, device=dev)
                for di, (Xc, Fc) in enumerate(((X, F), (Xr, Fr))):
                    Lp = local_points(Xc[ar, Fc.argmin(1)], z, sigmas)
                    if di == 0:  # the state's EI pick, as labeled in round 1
                        x0 = L["C"][g, :1].to(dev)
                    else:
                        x0 = base_picks(engine, Xc, Fc, gs, zeros.to(dev), tb, S)[:, :1]
                    Q = torch.cat([U, Lp, x0], 1)  # [G, Qn, d]
                    fq = tasks.f(Q, rows_t)
                    # the PFN's target: y = f with the instance's own preactivation and output noise
                    pr = _select(tasks.prior, rows_t)
                    pr.generator = torch.Generator(device=dev).manual_seed(base * 7919 + c0 * 101 + tb * 13 + s + di)
                    yq = pr.evaluate(Q, noise=True)
                    lg, _ = pfn_pass(engine, Xc, Fc, Q, with_z=False)
                    u = nuisance.stream(base, "pbi_pit", di, zeros, gs, (Q.shape[1],), S, False, dev)
                    ld, pit = score_points(bd, lg, yq, u)
                    ld_f, pit_f = score_points(bd, lg, fq, u)
                    inc = Fc.min(1).values
                    rec = {
                        "state": np.repeat(g.numpy(), Q.shape[1]),
                        "design": DESIGNS[di],
                        "ptype": np.tile(ptype, G),
                        "t": tb,
                        "f": fq.flatten().cpu().numpy(),
                        "y": yq.flatten().cpu().numpy(),
                        "inc": np.repeat(inc.cpu().numpy(), Q.shape[1]),
                        "log_density": ld.flatten().cpu().numpy(),
                        "pit": pit.flatten().cpu().numpy(),
                        "log_density_f": ld_f.flatten().cpu().numpy(),
                        "pit_f": pit_f.flatten().cpu().numpy(),
                        "mean": bd.mean(lg).flatten().cpu().numpy(),
                    }
                    for k in ks:
                        thr = (inc - k * w)[:, None].expand(-1, Q.shape[1])
                        rec[f"p_k{k}"] = event_prob(bd, lg, thr).flatten().cpu().numpy()
                        rec[f"o_k{k}"] = (yq < thr).flatten().cpu().numpy()
                        rec[f"of_k{k}"] = (fq < thr).flatten().cpu().numpy()
                    pts.append(pd.DataFrame(rec))
                    del lg
                # test 1: PFN scores of the labeled candidates at the BO context
                lab = L["lab"][g].to(dev)
                xq = L["C"][g].to(dev).gather(1, lab[..., None].expand(-1, -1, d))
                lg, _ = pfn_pass(engine, X, F, xq, with_z=False)
                inc = F.min(1).values
                ei = bd.ei(lg, inc[:, None]).clamp(min=0)
                J = lab.shape[1]
                cands.append(
                    pd.DataFrame(
                        {
                            "state": np.repeat(g.numpy(), J),
                            "j": np.tile(np.arange(J), G),
                            "valid": L["lab_valid"][g].flatten().numpy(),
                            "adv": L["adv"][g].flatten().numpy(),
                            "log_ei": torch.log(ei + 1e-12).flatten().cpu().numpy(),
                            "pi": bd.pi(lg, inc[:, None].expand(-1, J)).flatten().cpu().numpy(),
                        }
                    )
                )
                del lg
                fs, fm = L["fstar"][g], L["fmed"][g]
                gap = (inc.cpu() - fs).clamp(min=0)
                sts.append(
                    pd.DataFrame(
                        {
                            "state": g.numpy(),
                            "seed": L["seed"][g].numpy(),
                            "t": tb,
                            "behaviour": L["behaviour"][g].numpy(),
                            "inc_bo": inc.cpu().numpy(),
                            "inc_random": Fr.min(1).values.cpu().numpy(),
                            "fstar": fs.numpy(),
                            "range": (fm - fs).numpy(),
                            "gap": gap.numpy(),
                            "gap_bins": (gap / w).numpy(),
                        }
                    )
                )
        del tasks
    P = pd.concat(pts, ignore_index=True)
    C = pd.concat(cands, ignore_index=True)
    St = pd.concat(sts, ignore_index=True).sort_values("state").reset_index(drop=True)
    St["stratum"] = gap_stratum(St.gap_bins.to_numpy())
    St.attrs["f_check"] = fcheck
    return P, C, St


def _point_stats(D: pd.DataFrame, ks: list[int]) -> dict[str, float]:
    pit = D.pit.to_numpy()
    out = {
        "log_density": float(D.log_density.mean()),
        "ks": ks_uniform(pit),
        "cov50": float((np.abs(pit - 0.5) <= 0.25).mean()),
        "cov90": float((np.abs(pit - 0.5) <= 0.45).mean()),
    }
    for k in ks:
        p, o = D[f"p_k{k}"].to_numpy(), D[f"o_k{k}"].to_numpy().astype(float)
        out[f"ece_k{k}"] = ece(p, o)
        out[f"citl_k{k}"] = float(p.mean() - o.mean())
        out[f"rate_k{k}"] = float(o.mean())
    return out


def calibration_tables(P: pd.DataFrame, ks: list[int], B: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`calibration`: per design x point type (and "all"): n, mean log density, KS(PIT), coverage of the central
    50/90% intervals (from the PIT), ECE, calibration-in-the-large (mean predicted - observed) and base rate of
    each event. `contrasts`: BO - random of the same statistics, paired over states (both designs share the states
    and the points). Percentile bootstrap over states (one state per task)."""
    rows, crows = [], []
    for pt in ("all", "uniform", "local", "ei_pick"):
        Dp = P if pt == "all" else P.loc[P.ptype == pt]
        Db = cast(pd.DataFrame, Dp.loc[Dp.design == "bo"]).reset_index(drop=True)
        Dr = cast(pd.DataFrame, Dp.loc[Dp.design == "random"]).reset_index(drop=True)
        ib: dict = dict(Db.groupby("state").indices)
        ir: dict = dict(Dr.groupby("state").indices)
        st = np.array(sorted(ib))
        pb, pr = _point_stats(Db, ks), _point_stats(Dr, ks)
        rng = np.random.default_rng(0)
        boots: dict[str, list] = {m: [] for m in pb}
        bootr: dict[str, list] = {m: [] for m in pb}
        for _ in range(B):
            pick = rng.choice(st, len(st))
            sb = _point_stats(Db.iloc[np.concatenate([ib[s] for s in pick])], ks)
            sr = _point_stats(Dr.iloc[np.concatenate([ir[s] for s in pick])], ks)
            for m in pb:
                boots[m].append(sb[m])
                bootr[m].append(sr[m])
        for m in pb:
            for des, point, bs in (("bo", pb, boots[m]), ("random", pr, bootr[m])):
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                n = len(Db) if des == "bo" else len(Dr)
                rows.append({"design": des, "ptype": pt, "metric": m, "value": point[m], "lo": lo, "hi": hi, "n": n})
            dl, dh = np.nanpercentile(np.array(boots[m]) - np.array(bootr[m]), [2.5, 97.5])
            crows.append({"ptype": pt, "metric": m, "diff": pb[m] - pr[m], "lo": dl, "hi": dh})
    return pd.DataFrame(rows), pd.DataFrame(crows)


def strata_table(C: pd.DataFrame, St: pd.DataFrame, L: dict, S: int, B: int) -> pd.DataFrame:
    """Test 1: per gap stratum (and "all"): states, label split-half reliability, cross-fitted headroom,
    the share of non-base advantages below one PFN bin (in ECDF units), within-state Spearman of log EI and PI
    with the label. Bootstrap CIs over states."""
    R, lv = L["R"][:S], L["lab_valid"][:S]
    head = crossfit_headroom(R, lv).numpy()
    h = R.shape[-1] // 2
    a1 = (R[:, :1, :h] - R[:, :, :h]).mean(-1).numpy()
    a2 = (R[:, :1, h:] - R[:, :, h:]).mean(-1).numpy()
    cm = lv.numpy().copy()
    cm[:, 0] = False
    rng_ = St.set_index("state")["range"].reindex(range(S)).to_numpy()
    adv_u = np.abs(L["adv"][:S].numpy()) / 100 * rng_[:, None]
    Cn = cast(pd.DataFrame, C.loc[C.valid & (C.j > 0)])
    rows = []
    strata = ["all", "<1", "1-10", "10-100", ">100"]
    for sname in strata:
        sel = St.state.to_numpy() if sname == "all" else St.loc[St.stratum == sname, "state"].to_numpy()
        if len(sel) == 0:
            continue
        m = np.zeros(S, bool)
        m[sel] = True
        cmask = cm & m[:, None]
        grp = np.repeat(np.arange(S), R.shape[1]).reshape(S, -1)[cmask]
        rel = within_spearman(a1[cmask], a2[cmask], grp)
        C_s = cast(pd.DataFrame, Cn.loc[Cn.state.isin(sel)])
        rho_ei = within_spearman(C_s.log_ei.to_numpy(), C_s.adv.to_numpy(), C_s.state.to_numpy())
        rho_pi = within_spearman(C_s.pi.to_numpy(), C_s.adv.to_numpy(), C_s.state.to_numpy())
        hs = head[sel]
        frac = (adv_u[cmask] < 1e-3).mean()
        rng = np.random.default_rng(0)

        def ci(x: np.ndarray, rng: np.random.Generator = rng) -> tuple[float, float]:
            x = x[np.isfinite(x)]
            if len(x) == 0:
                return float("nan"), float("nan")
            v = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
            return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

        for metric, x in (
            ("reliability", rel),
            ("headroom", hs),
            ("spearman_log_ei", rho_ei),
            ("spearman_pi", rho_pi),
        ):
            lo, hi = ci(np.asarray(x, float))
            rows.append(
                {
                    "stratum": sname,
                    "metric": metric,
                    "value": float(np.nanmean(x)),
                    "lo": lo,
                    "hi": hi,
                    "n_states": len(sel),
                }
            )
        rows.append(
            {
                "stratum": sname,
                "metric": "frac_adv_below_bin",
                "value": float(frac),
                "lo": np.nan,
                "hi": np.nan,
                "n_states": len(sel),
            }
        )
        rows.append(
            {
                "stratum": sname,
                "metric": "headroom_sum_share",
                "value": float(hs.sum() / head.sum()),
                "lo": np.nan,
                "hi": np.nan,
                "n_states": len(sel),
            }
        )
    return pd.DataFrame(rows)


def aggregate_points(P: pd.DataFrame, St: pd.DataFrame, ks: list[int]) -> dict[str, pd.DataFrame]:
    """Small tables of the per-point scores for the notebook: PIT histograms (`pit_hist`), reliability-diagram
    deciles (`reliability`), calibration by step t (`by_t`, a function of the context) and, descriptive only, by gap
    stratum (`by_stratum`, conditioned on f*)."""
    P = P.merge(St[["state", "stratum"]], on="state")
    P["t_bin"] = np.array(["0-4", "5-9", "10-14", "15-19"])[np.clip(P.t.to_numpy() // 5, 0, 3)]
    hist, rel = [], []
    for des in DESIGNS:
        for pt in ("all", "uniform", "local", "ei_pick"):
            D = cast(pd.DataFrame, P.loc[(P.design == des) & ((P.ptype == pt) | (pt == "all"))])
            for target, col in (("y", "pit"), ("f", "pit_f")):
                dens, _ = np.histogram(D[col], bins=20, range=(0, 1), density=True)
                hist.append(
                    pd.DataFrame({"design": des, "ptype": pt, "target": target, "bin": np.arange(20), "density": dens})
                )
            for k in ks:
                order = np.argsort(D[f"p_k{k}"].to_numpy(), kind="stable")
                dec = np.empty(len(D), dtype=int)
                for i, part in enumerate(np.array_split(order, 10)):
                    dec[part] = i
                g = D.assign(decile=dec).groupby("decile")
                g = g.agg(p=(f"p_k{k}", "mean"), o=(f"o_k{k}", "mean"), n=(f"p_k{k}", "size")).reset_index()
                rel.append(g.assign(design=des, ptype=pt, k=k))

    def cov(x: pd.Series, half: float) -> float:
        return float((np.abs(x - 0.5) <= half).mean())

    by_t = (
        P.groupby(["design", "ptype", "t_bin"])
        .agg(
            cov90=("pit", lambda x: cov(x, 0.45)),
            cov50=("pit", lambda x: cov(x, 0.25)),
            p_k0=("p_k0", "mean"),
            o_k0=("o_k0", "mean"),
            n=("pit", "size"),
        )
        .reset_index()
    )
    by_t["citl_k0"] = by_t.p_k0 - by_t.o_k0
    by_stratum = (
        P.groupby(["stratum", "design", "ptype"])
        .agg(
            log_density=("log_density", "mean"),
            log_density_f=("log_density_f", "mean"),
            cov90=("pit", lambda x: cov(x, 0.45)),
            n=("pit", "size"),
        )
        .reset_index()
    )
    return {
        "pit_hist": pd.concat(hist, ignore_index=True),
        "reliability": pd.concat(rel, ignore_index=True),
        "by_t": by_t,
        "by_stratum": by_stratum,
    }


def pfn_bo_inference(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(t.tf32)
    torch.backends.cudnn.allow_tf32 = bool(t.tf32)
    pfn = load(t.pfn)[0].to(t.device).eval()
    engine = Engine(pfn, None, _candidates(t.candidates), base_seed=cfg.seed, chunk=int(t.chunk))
    L = load_labels(cfg)
    S = min(int(t.n_states), len(L["seed"]))
    t0 = time.time()
    P, C, St = collect(cfg, engine, L, S)
    run.log_summary(
        **{
            "time/collect_s": time.time() - t0,
            "n_states": float(S),
            "n_points": float(len(P)),
            "check/max_abs_f_minus_label_F": float(St.attrs["f_check"]),
        }
    )
    t0 = time.time()
    ks = [int(k) for k in t.event_bins]
    cal, con = calibration_tables(P, ks, int(t.n_boot))
    strata = strata_table(C, St, L, S, int(t.n_boot))
    run.log_summary(**{"time/analysis_s": time.time() - t0})
    P.to_parquet(run.artifacts / "points.parquet")  # per point: an artifact (too large for a result table)
    for name, df in (
        *aggregate_points(P, St, ks).items(),
        ("cands", C),
        ("states", St),
        ("calibration", cal),
        ("contrasts", con),
        ("strata", strata),
    ):
        run.log_table(name, df)
    for _, r in cal.loc[cal.ptype == "all"].iterrows():
        run.log_summary(**{f"cal/{r['design']}/{r['metric']}": float(r["value"])})
    verdict = inference_verdict(cal, con, ks)
    run.log_summary(**{"verdict_code": {"GO": 1.0, "NO-GO": -1.0, "INCONCLUSIVE": 0.0}[verdict]})
    return float(cal.loc[(cal.design == "bo") & (cal.ptype == "all") & (cal.metric == "cov90"), "value"].iloc[0])

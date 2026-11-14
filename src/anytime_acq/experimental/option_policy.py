"""A one-deviation option policy learned from exact option labels (tasks/option_policy.py; pre-registration:
notebooks/13_2026-11-12_option_policy.ipynb; reasoning: research_log/decisions/0008 §4 step 3).

The policy follows the base arm and, at a decision step, may commit once to an option "arm a for k steps (or for
the rest of the run), then the base to T". It reads only what is known at decision time:
- `data_features`: statistics of D = (X, Y) and the budget;
- `belief_features`: summaries of the frozen PFN's predictive on the deterministic seed-s candidate set, and of
  what each acquisition rule would pick there;
- `pfn_state`: the PFN's pooled context-token states (last layer).
Privileged information (the true f, f*) enters only through the labels the value model Q(state, option) is
regressed on (`fit_predict`).
"""

from dataclasses import dataclass

import torch
from pfn_repro.acquisitions.standard import arm_scores, fstar_quantiles, parse_arm
from torch import Tensor, nn

from ..bo.engine import Engine
from .api import _argmax, pfn_pass
from .options import _bd, seeded_candidates
from .probes import pooled, ridge_fit_predict

FEATURE_ARMS = ("ei", "pi0.01", "ucb0.995", "mes", "mean")
TINY = 1e-7


def _log(x: Tensor, floor: float = TINY) -> Tensor:
    return torch.log10(x.clamp(min=0) + floor)


def data_features(X: Tensor, Y: Tensor, t: int, T: int, n_init: int) -> tuple[Tensor, list[str]]:
    """[L, p]: statistics of the data and the budget only (no model). X [L, n, d], Y [L, n], n = n_init + t.
    Feature 0 is t / T."""
    L, n, _ = X.shape
    assert n == n_init + t and n >= 3
    r = torch.arange(L, device=X.device)
    order = Y.argsort(1)
    ys = Y.gather(1, order)
    inc = ys[:, 0]
    xi = X[r, order[:, 0]]
    D = (X - xi[:, None]).norm(dim=-1)
    Dm = D.clone()
    Dm[r, order[:, 0]] = float("inf")
    nn_d, jn = Dm.min(1)
    rm = Y.cummin(1).values
    zeros = torch.zeros(L, device=X.device)
    if t > 0:
        imp = rm[:, n_init - 1 : -1] - rm[:, n_init:]  # [L, t]: improvement at each BO step
        hit = imp > 0
        last = torch.where(hit.any(1), t - 1 - hit.flip(1).float().argmax(1), torch.full_like(r, -1))
        since = torch.where(last >= 0, t - 1 - last, torch.full_like(r, t)).float() / T
        last_imp = torch.where(last >= 0, _log(imp[r, last.clamp(min=0)]), torch.full((L,), -7.0, device=X.device))
        n_imp = hit.float().mean(1)
        m = min(3, t)
        recent_local = (D[:, n - m :] < 0.05).float().mean(1)
        recent_dist = D[:, n - m :].mean(1)
    else:
        since, last_imp, n_imp, recent_local, recent_dist = zeros, zeros - 7.0, zeros, zeros, zeros
    feats = {
        "t_frac": zeros + t / T,
        "y_inc": inc,
        "y_median": ys[:, n // 2],
        "y_std": Y.std(1),
        "log_gap2": _log(ys[:, 1] - inc),
        "log_gap3": _log(ys[:, 2] - inc),
        "since_improved": since,
        "log_last_improvement": last_imp,
        "share_improving_steps": n_imp,
        "log_total_improvement": _log(rm[:, n_init - 1] - rm[:, -1]),
        "log_nn_dist": _log(nn_d, 1e-4),
        "log_nn_slope": _log((Y[r, jn] - inc) / (nn_d + 1e-4)),
        "share_within_0.05": (D < 0.05).float().mean(1),
        "share_within_0.2": (D < 0.2).float().mean(1),
        "dist_best_second": (X[r, order[:, 1]] - xi).norm(dim=-1),
        "boundary_dist": torch.minimum(xi, 1 - xi).min(1).values,
        "recent_local": recent_local,
        "recent_dist": recent_dist,
    }
    return torch.stack(list(feats.values()), 1), list(feats)


@torch.no_grad()
def belief_features(engine: Engine, X: Tensor, Y: Tensor, s: int) -> tuple[Tensor, list[str]]:
    """[L, p]: summaries of the PFN's predictive on the seed-s candidate set and of each arm's pick there (the
    pick `options.py::arm_pick` would make). Depends on the data and s only."""
    bd = _bd(engine)
    ns, nl = engine.candidates.n_sobol, engine.candidates.n_local
    out, names = [], []
    for a in range(0, len(X), engine.chunk):
        Xa, Ya = X[a : a + engine.chunk], Y[a : a + engine.chunk]
        r = torch.arange(len(Xa), device=X.device)
        C = seeded_candidates(engine, Xa, Ya, s)
        lg, _ = pfn_pass(engine, Xa, Ya, C, with_z=False)
        inc = Ya.min(1).values
        xi = Xa[r, Ya.argmin(1)]
        mu = bd.mean(lg)
        sd = bd.variance(lg).clamp(min=0).sqrt()
        pi = bd.pi(lg, inc[:, None])
        ei = bd.ei(lg, inc[:, None])
        valid = torch.cdist(C, Xa).min(-1).values > 1e-7
        fs = fstar_quantiles(bd, lg, inc)
        eiv = ei.masked_fill(~valid, 0.0)
        f = {
            "log_max_ei": _log(eiv.max(1).values, 1e-12),
            "max_pi": pi.masked_fill(~valid, 0.0).max(1).values,
            "min_mean_minus_inc": mu.min(1).values - inc,
            "sd_global": sd[:, :ns].mean(1),
            "sd_local": sd[:, ns : ns + nl].mean(1),
            "share_pi_above_0.05": (pi[:, :ns] > 0.05).float().mean(1),
            "share_mean_below_inc": (mu[:, :ns] < inc[:, None]).float().mean(1),
            "log_ei_local_over_global": _log(eiv[:, ns:].max(1).values, 1e-12) - _log(eiv[:, :ns].max(1).values, 1e-12),
            "log_expected_fstar_gap": _log(inc - fs.mean(1)),
            "log_fstar_spread": _log(fs.std(1)),
        }
        picks = {}
        for arm in FEATURE_ARMS:
            sc = arm_scores(arm, bd, lg, inc, engine.ucb_quantile, mes_fs=fs if parse_arm(arm)[0] == "mes" else None)
            j = _argmax(sc, mu, valid)
            picks[arm] = j
            f[f"{arm}:dist_inc"] = (C[r, j] - xi).norm(dim=-1)
            f[f"{arm}:mean_minus_inc"] = mu[r, j] - inc
            f[f"{arm}:sd"] = sd[r, j]
            f[f"{arm}:pi"] = pi[r, j]
            f[f"{arm}:log_ei"] = _log(ei[r, j], 1e-12)
            f[f"{arm}:is_local"] = (j >= ns).float()
            if arm != "ei":
                f[f"{arm}:dist_ei_pick"] = (C[r, j] - C[r, picks["ei"]]).norm(dim=-1)
        P = torch.stack(list(picks.values()), 1)
        f["n_distinct_picks"] = (P.sort(1).values.diff(dim=1) != 0).sum(1).float() + 1
        out.append(torch.stack(list(f.values()), 1))
        names = list(f)
        del lg
    return torch.cat(out), names


@torch.no_grad()
def pfn_state(engine: Engine, X: Tensor, Y: Tensor) -> Tensor:
    """[L, 3 d_model]: the last layer's pooled context states (mean, max, the incumbent's token)."""
    assert engine.pfn is not None
    return torch.cat(
        [
            pooled(engine.pfn, X[a : a + engine.chunk], Y[a : a + engine.chunk])[:, -1]
            for a in range(0, len(X), engine.chunk)
        ]
    ).float()


def pca(Z: Tensor, k: int) -> Tensor:
    """[N, k]: the first k principal components of Z (unsupervised; fit on all rows)."""
    Zc = (Z - Z.mean(0)).double()
    _, _, V = torch.pca_lowrank(Zc, q=min(k, Zc.shape[1], Zc.shape[0] - 1), center=False, niter=4)
    return (Zc @ V).float()


# --------------------------------------------------------------------------------------------------
# the value model Q(state, option)
# --------------------------------------------------------------------------------------------------


@dataclass
class FitSpec:
    width: int = 128
    epochs: int = 150
    patience: int = 15
    batch: int = 1024
    lr: float = 2e-3
    weight_decay: float = 1e-3
    dropout: float = 0.1
    n_ens: int = 2


def _mlp_once(Xtr: Tensor, Ytr: Tensor, gtr: Tensor, Xte: Tensor, seed: int, spec: FitSpec) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    ug = gtr.unique().cpu()
    val_g = ug[torch.randperm(len(ug), generator=gen)][: max(1, len(ug) // 6)].to(gtr.device)
    is_val = torch.isin(gtr, val_g)
    tr = (~is_val).nonzero().squeeze(1)
    torch.manual_seed(seed)
    w = spec.width
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], w),
        nn.GELU(),
        nn.Dropout(spec.dropout),
        nn.Linear(w, w),
        nn.GELU(),
        nn.Linear(w, Ytr.shape[1]),
    ).to(Xtr.device)
    opt = torch.optim.AdamW(net.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    best, state, bad = float("inf"), None, 0
    for _ in range(spec.epochs):
        net.train()
        for b in tr[torch.randperm(len(tr), generator=gen).to(tr.device)].split(spec.batch):
            opt.zero_grad()
            ((net(Xtr[b]) - Ytr[b]) ** 2).mean().backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(Xtr[is_val]) - Ytr[is_val]) ** 2).mean())
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= spec.patience:
                break
    assert state is not None
    net.load_state_dict(state)
    with torch.no_grad():
        return net(Xte)


def fit_predict(
    Xtr: Tensor, Ytr: Tensor, gtr: Tensor, Xte: Tensor, learner: str, seed: int, spec: FitSpec | None = None
) -> Tensor:
    """Predictions [Nte, O] of the option advantages from features, by squared error (the policy needs the
    conditional mean). Rows of one task share a group in `gtr` (validation is by task). learner: mlp | ridge."""
    spec = spec or FitSpec()
    m, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    A, B = (((x - m) / sd).clamp(-8, 8).float() for x in (Xtr, Xte))
    if learner == "ridge":
        return ridge_fit_predict(A, Ytr, B, groups=gtr)[0]
    if learner != "mlp":
        raise ValueError(learner)
    ym, ys = Ytr.mean(0), Ytr.std(0).clamp(min=1e-6)
    Yz = ((Ytr - ym) / ys).float()
    with torch.enable_grad():
        preds = [_mlp_once(A, Yz, gtr, B, seed * 1000 + e, spec) for e in range(spec.n_ens)]
    return torch.stack(preds).mean(0) * ys + ym


def crossfit_q(
    X: Tensor, Y: Tensor, task: Tensor, fold: Tensor, learner: str, n_inner: int, seed: int, spec: FitSpec | None = None
) -> tuple[Tensor, Tensor]:
    """Nested cross-fitting over tasks. X [R, p], Y [R, O], task [R], fold [R] (a task's rows share its fold).
    -> (Q_out [R, O]: each row predicted by a model that never saw its fold;
        Q_in [K, R, O]: for outer fold k, out-of-inner-fold predictions of the rows *not* in fold k, from models
        that saw neither fold k nor the row's inner fold (NaN on fold k's rows). Thresholds for fold k are chosen
        on Q_in[k], so nothing about fold k's tasks enters its policy."""
    K = int(fold.max()) + 1
    Q_out = torch.full_like(Y, float("nan"))
    Q_in = torch.full((K, *Y.shape), float("nan"), device=Y.device)
    for k in range(K):
        tr = (fold != k).nonzero().squeeze(1)
        te = (fold == k).nonzero().squeeze(1)
        Q_out[te] = fit_predict(X[tr], Y[tr], task[tr], X[te], learner, seed + k, spec)
        inner = (fold[tr] - (fold[tr] > k).long()) % n_inner  # the K - 1 training folds, merged into n_inner groups
        for i in range(n_inner):
            a, b = tr[inner != i], tr[inner == i]
            Q_in[k, b] = fit_predict(X[a], Y[a], task[a], X[b], learner, seed + 100 * (k + 1) + i, spec)
    return Q_out, Q_in

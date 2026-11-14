"""Readout probes (research_log/decisions/0006 §1): what can be read from the frozen PFN's state at a BO context?

Representations (frozen `pfn_large`, no training of the PFN):
- `context_states`: the context tokens' residual-stream states after the embedding and after every block
  (layer 0 = embedding, L = last), exactly as `PFN.context_kv_with_grad` advances them;
- `pooled`: per layer, the mean and the max over context tokens and the incumbent's token -> [S, L+1, 3 d_model];
- `query_layer_states`: a candidate's query-token state after the embedding and every block (the test-token path
  of `logits_from_context_kv`) -> [G, J, L+1, d_model].

Targets (ground truth from the task, or computed by the PFN for KG):
- state level: log10 of the remaining gap (incumbent − f*) in % of the task's gap; x* (location, d values);
  the number of relevant inputs (prior's relevant_mask); roughness (log mean |grad f| over Sobol points); and
  the clairvoyant headroom of the state (asinh of the cross-fitted best advantage of the labels, and whether
  it is > 0.5; both heavy-tailed);
- candidate level: the ground-truth advantage label; KG by re-encoding (`kg_reencode`).

Probes: ridge regression (linear, regularization chosen on a validation part of the training tasks) and a
one-hidden-layer MLP, each on: hand features (the round-1 head's), the last layer, all layers.
"""

import math

import numpy as np
import torch
import torch.nn.functional as Fn
from pfn_repro.surrogates.pfn import _pad_and_rescale_features
from torch import Tensor, nn

from ..bo.engine import Engine
from ..bo.tasks import TaskSet
from .api import pfn_pass


@torch.no_grad()
def context_states(pfn: nn.Module, X: Tensor, Y: Tensor) -> Tensor:
    """[S, L+1, n, d_model]: context tokens after the embedding and after each block."""
    nf = torch.full((len(X),), X.shape[-1], device=X.device, dtype=torch.long)
    x = _pad_and_rescale_features(X, pfn.max_x_dim, nf)
    tok = pfn.train_embed(torch.cat([x, Y.unsqueeze(-1)], dim=-1))
    out = [tok]
    for blk in pfn.blocks:
        tok, _ = blk(tok, tok[:, :0], None, None)
        out.append(tok)
    return torch.stack(out, 1)


@torch.no_grad()
def pooled(pfn: nn.Module, X: Tensor, Y: Tensor) -> Tensor:
    """[S, L+1, 3 d_model]: mean, max over context tokens, and the incumbent's (argmin y) token, per layer."""
    Z = context_states(pfn, X, Y)
    inc = Z[torch.arange(len(X), device=X.device), :, Y.argmin(1)]  # [S, L+1, dm]
    return torch.cat([Z.mean(2), Z.max(2).values, inc], -1)


@torch.no_grad()
def query_layer_states(pfn: nn.Module, X: Tensor, Y: Tensor, xq: Tensor) -> Tensor:
    """[G, J, L+1, d_model]: the query-token states of candidates xq [G, J, d] after the embedding and each block."""
    nf = torch.full((len(X),), X.shape[-1], device=X.device, dtype=torch.long)
    kv = pfn.context_kv(X, Y, n_features=nf)
    xs = _pad_and_rescale_features(xq, pfn.max_x_dim, nf)
    tok = pfn.test_x_embed(xs) + pfn.test_placeholder.view(1, 1, -1)
    B, Q, Dm = tok.shape
    out = [tok]
    for blk, (K, V) in zip(pfn.blocks, kv, strict=True):
        a = blk.attn
        N = K.shape[1]
        q = a.q_proj(blk.ln1(tok)).view(B, Q, a.h, a.dk).transpose(1, 2)
        k = K.view(B, N, a.h, a.dk).transpose(1, 2)
        v = V.view(B, N, a.h, a.dk).transpose(1, 2)
        att = ((q @ k.transpose(-2, -1)) / (a.dk**0.5)).softmax(-1)
        tok = tok + a.out_proj((att @ v).transpose(1, 2).reshape(B, Q, Dm))
        tok = tok + blk.ff(blk.ln2(tok))
        out.append(tok)
    return torch.stack(out, 2)


@torch.no_grad()
def kg_reencode(engine: Engine, X: Tensor, Y: Tensor, xq: Tensor, C: Tensor, n_y: int = 8) -> Tensor:
    """One-step knowledge gradient of candidates xq [G, J, d] by re-encoding (minimization): with the domain
    D = C ∪ evaluated points ∪ {xq}, KG(x) = min_D mu_D(x') − E_y[min_D mu_{D+(x,y)}(x')], y at n_y stratified
    quantiles of the PFN's predictive at x. -> [G, J]."""
    assert engine.pfn is not None
    bd = engine.pfn.bar_dist
    G, J, d = xq.shape
    dom = torch.cat([C, X, xq], 1)  # [G, P, d]
    lg_dom, _ = pfn_pass(engine, X, Y, dom, with_z=False)
    old = bd.mean(lg_dom).min(1).values  # [G]
    lg_q, _ = pfn_pass(engine, X, Y, xq, with_z=False)  # [G, J, nb]
    ys = torch.stack([bd.icdf(lg_q, (k + 0.5) / n_y) for k in range(n_y)], -1)  # [G, J, n_y]
    L = G * J * n_y
    Xl = torch.cat([X.repeat_interleave(J * n_y, 0), xq.repeat_interleave(n_y, 1).reshape(L, 1, d)], 1)
    Yl = torch.cat([Y.repeat_interleave(J * n_y, 0), ys.reshape(L, 1)], 1)
    new = torch.empty(L, device=X.device)
    for s in range(0, L, engine.chunk):
        sl = slice(s, s + engine.chunk)
        lg, _ = pfn_pass(engine, Xl[sl], Yl[sl], dom.repeat_interleave(J * n_y, 0)[sl], with_z=False)
        new[sl] = bd.mean(lg).min(1).values
    return old[:, None] - new.view(G, J, n_y).mean(-1)


def roughness(tasks: TaskSet, n: int = 1024, chunk: int = 128) -> Tensor:
    """[B]: log mean gradient norm of f over a scrambled-Sobol design (true task); in chunks of points (memory)."""
    x_all = torch.quasirandom.SobolEngine(tasks.d, scramble=True, seed=31).draw(n).to(tasks.fstar.device)
    tot = torch.zeros(tasks.B, device=tasks.fstar.device)
    for s in range(0, n, chunk):
        x = x_all[None, s : s + chunk].expand(tasks.B, -1, -1).clone().requires_grad_(True)
        tasks.f(x).sum().backward()
        assert x.grad is not None
        tot += x.grad.norm(dim=-1).sum(1).detach()
    return torch.log(tot / n + 1e-9)


# --------------------------------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------------------------------


def _standardize(tr: Tensor, *others: Tensor) -> list[Tensor]:
    m, s = tr.mean(0), tr.std(0).clamp(min=1e-6)
    return [(x - m) / s for x in (tr, *others)]


def ridge_fit_predict(
    Xtr: Tensor,
    ytr: Tensor,
    Xte: Tensor,
    alphas: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4),
    val_frac: float = 0.2,
    groups: Tensor | None = None,
    chunk: int = 4096,
) -> tuple[Tensor, float]:
    """Ridge with alpha chosen on a validation part of the training rows (by group if given). ytr [N] or [N, k].
    Standardized inputs; float64 normal equations accumulated over row chunks (no float64 copy of X), or the
    kernel form when there are more features than rows. -> predictions on Xte, chosen alpha."""
    m, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    y = ytr.double()
    ym = y.mean(0)
    yc = y - ym
    n, p = Xtr.shape
    g = torch.arange(n) if groups is None else groups.cpu()
    ug = g.unique()
    perm = ug[torch.randperm(len(ug), generator=torch.Generator().manual_seed(0))]
    val_g = perm[: max(1, int(len(ug) * val_frac))]
    is_val = torch.isin(g, val_g).to(Xtr.device)
    tr_i, va_i = (~is_val).nonzero().squeeze(1), is_val.nonzero().squeeze(1)

    def z(X: Tensor, idx: Tensor) -> Tensor:
        return ((X[idx] - m) / sd).double()

    def predict(X: Tensor, w: Tensor) -> Tensor:
        idx = torch.arange(len(X), device=X.device)
        return torch.cat([z(X, idx[s : s + chunk]) @ w for s in range(0, len(X), chunk)]) + ym

    eye = lambda k: torch.eye(k, dtype=torch.float64, device=Xtr.device)  # noqa: E731
    best, best_a = math.inf, alphas[0]
    if p > len(tr_i):  # kernel form (few rows)
        A_tr, A_va = z(Xtr, tr_i), z(Xtr, va_i)
        K = A_tr @ A_tr.T
        for a in alphas:
            w = A_tr.T @ torch.linalg.solve(K + a * eye(len(A_tr)), yc[tr_i])
            err = float(((A_va @ w + ym - y[va_i]) ** 2).mean())
            if err < best:
                best, best_a = err, a
        A = z(Xtr, torch.arange(n, device=Xtr.device))
        w = A.T @ torch.linalg.solve(A @ A.T + best_a * eye(n), yc)
        return predict(Xte, w).float(), best_a

    def normal(idx: Tensor) -> tuple[Tensor, Tensor]:
        G = torch.zeros(p, p, dtype=torch.float64, device=Xtr.device)
        b = torch.zeros((p, *y.shape[1:]), dtype=torch.float64, device=Xtr.device)
        for s in range(0, len(idx), chunk):
            A = z(Xtr, idx[s : s + chunk])
            G += A.T @ A
            b += A.T @ yc[idx[s : s + chunk]]
        return G, b

    G1, b1 = normal(tr_i)
    G2, b2 = normal(va_i)
    for a in alphas:
        w = torch.linalg.solve(G1 + a * eye(p), b1)
        err = float(((predict(Xtr[va_i], w) - y[va_i]) ** 2).mean())
        if err < best:
            best, best_a = err, a
    w = torch.linalg.solve(G1 + G2 + best_a * eye(p), b1 + b2)
    return predict(Xte, w).float(), best_a


def mlp_fit_predict(
    Xtr: Tensor, ytr: Tensor, Xte: Tensor, kind: str, epochs: int = 200, seed: int = 0, groups: Tensor | None = None
) -> Tensor:
    """One hidden layer (256), AdamW, early stopping on a validation part of the training rows.
    kind: "reg" (MSE) or "bin" (logistic; returns logits). ytr [N] or [N, k]."""
    torch.manual_seed(seed)
    Xtr, Xte = _standardize(Xtr.float(), Xte.float())
    y = ytr.float() if ytr.dim() > 1 else ytr.float()[:, None]
    ym, ys = (
        (y.mean(0), y.std(0).clamp(min=1e-6))
        if kind == "reg"
        else (torch.zeros(1, device=y.device), torch.ones(1, device=y.device))
    )
    yz = (y - ym) / ys
    n = len(Xtr)
    g = torch.arange(n) if groups is None else groups.cpu()
    ug = g.unique()
    val_g = ug[torch.randperm(len(ug), generator=torch.Generator().manual_seed(seed))][: max(1, len(ug) // 5)]
    is_val = torch.isin(g, val_g).to(Xtr.device)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, y.shape[1])).to(
        Xtr.device
    )
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = (
        (lambda p, t: Fn.mse_loss(p, t)) if kind == "reg" else (lambda p, t: Fn.binary_cross_entropy_with_logits(p, t))
    )
    best, state, bad = math.inf, None, 0
    tr_idx = (~is_val).nonzero().squeeze(1)
    for _ in range(epochs):
        net.train()
        for b in tr_idx[torch.randperm(len(tr_idx), device=tr_idx.device)].split(256):
            opt.zero_grad()
            lossf(net(Xtr[b]), yz[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(Xtr[is_val]), yz[is_val]))
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    assert state is not None
    net.load_state_dict(state)
    with torch.no_grad():
        out = net(Xte) * ys + ym if kind == "reg" else net(Xte)
    return out.squeeze(-1) if ytr.dim() == 1 else out


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve (rank formulation; ties averaged)."""
    from scipy.stats import rankdata

    r = rankdata(score)
    pos = y.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def within_spearman(pred: np.ndarray, y: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Spearman correlation within each group (state); groups with constant y or pred give nan."""
    from scipy.stats import spearmanr

    out = []
    for g in np.unique(group):
        m = group == g
        if np.std(y[m]) == 0 or np.std(pred[m]) == 0:
            out.append(np.nan)
        else:
            out.append(spearmanr(pred[m], y[m]).statistic)
    return np.asarray(out, dtype=float)


def spearman(pred: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(pred, y).statistic) if np.std(pred) > 0 and np.std(y) > 0 else float("nan")

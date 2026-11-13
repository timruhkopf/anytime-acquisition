"""An acquisition branch on the frozen PFN (research_log/decisions/0009; tasks/acq_distill.py).

The PFN is not changed. A second, trainable stream runs beside it:
- *acquisition tokens*, one per observed point, start from the point's location only. In block b each token
  (1) cross-attends to the PFN's context-token states of layer `read[b]` (so it can pull decision-relevant
  information out of the PFN's train state, at its own location), (2) self-attends to the other acquisition
  tokens and a budget token (so global quantities can form), (3) passes a feed-forward layer;
- *query tokens*, one per candidate x, start from the candidate's location and the budget. In block b each
  cross-attends to the acquisition tokens of that block (the PFN's own train/test pattern, on the branch), then a
  feed-forward layer. Queries do not see each other: the score of x depends on the data and x only;
- *output*: one score per arm and query (`n_out` columns). Scores are only meaningful as a ranking over x.

`query_state=True` (ablation) also gives each query the PFN's final query-token state at x.

Distillation (`rank_loss`): for every arm, the branch's column is trained to rank candidates as the arm's
acquisition does: a listwise top-1 term (cross-entropy towards the arm's argmax), a pairwise logistic term on
random pairs, and one on (top-k of the arm, any other) pairs. All are invariant to the arms' scales.
"""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as Fn
from pfn_repro.acquisitions.standard import arm_scores, fstar_quantiles, parse_arm
from pfn_repro.surrogates.pfn import _pad_and_rescale_features
from torch import Tensor, nn

from ..bo.engine import Engine
from .api import _argmax, pfn_pass, query_states
from .options import _bd, seeded_candidates
from .probes import context_states

N_BUDGET = 3


def budget_features(t: int, T: int, n: int, B: int, device: torch.device | str) -> Tensor:
    """[B, 3]: share of the budget used, share remaining, context size / 50."""
    return torch.tensor([t / T, (T - t) / T, n / 50.0], device=device).expand(B, N_BUDGET)


@dataclass
class BranchSpec:
    width: int = 192
    heads: int = 4
    ff_mult: int = 2
    read: tuple[int, ...] | None = None  # PFN layers read by the blocks (1..L); None = every layer
    self_attn: bool = True
    query_state: bool = False
    dropout: float = 0.0


class _Block(nn.Module):
    def __init__(self, s: BranchSpec, d_pfn: int):
        super().__init__()
        w = s.width
        self.mem_ln, self.mem_proj = nn.LayerNorm(d_pfn), nn.Linear(d_pfn, w)
        self.ln_x, self.xattn = nn.LayerNorm(w), nn.MultiheadAttention(w, s.heads, s.dropout, batch_first=True)
        self.self_attn = s.self_attn
        if s.self_attn:
            self.ln_s, self.sattn = nn.LayerNorm(w), nn.MultiheadAttention(w, s.heads, s.dropout, batch_first=True)
        self.ln_f = nn.LayerNorm(w)
        self.ff = nn.Sequential(nn.Linear(w, s.ff_mult * w), nn.GELU(), nn.Linear(s.ff_mult * w, w))
        self.ln_k = nn.LayerNorm(w)
        self.ln_q, self.qattn = nn.LayerNorm(w), nn.MultiheadAttention(w, s.heads, s.dropout, batch_first=True)
        self.ln_qf = nn.LayerNorm(w)
        self.qff = nn.Sequential(nn.Linear(w, s.ff_mult * w), nn.GELU(), nn.Linear(s.ff_mult * w, w))

    def context(self, a: Tensor, z: Tensor) -> Tensor:
        m = self.mem_proj(self.mem_ln(z))
        h = self.ln_x(a)
        a = a + self.xattn(h, m, m, need_weights=False)[0]
        if self.self_attn:
            h = self.ln_s(a)
            a = a + self.sattn(h, h, h, need_weights=False)[0]
        return a + self.ff(self.ln_f(a))

    def query(self, q: Tensor, a: Tensor) -> Tensor:
        k = self.ln_k(a)
        q = q + self.qattn(self.ln_q(q), k, k, need_weights=False)[0]
        return q + self.qff(self.ln_qf(q))


class AcqBranch(nn.Module):
    def __init__(self, d_pfn: int, n_layers: int, max_x_dim: int, n_out: int, spec: BranchSpec | None = None):
        super().__init__()
        s = spec or BranchSpec()
        self.spec, self.max_x_dim = s, max_x_dim
        self.read = tuple(s.read) if s.read is not None else tuple(range(1, n_layers + 1))
        w = s.width
        self.loc = nn.Linear(max_x_dim, w)
        self.qloc = nn.Linear(max_x_dim, w)
        self.dim_emb = nn.Embedding(max_x_dim + 1, w)
        self.budget_tok = nn.Sequential(nn.Linear(N_BUDGET, w), nn.GELU(), nn.Linear(w, w))
        self.budget_q = nn.Sequential(nn.Linear(N_BUDGET, w), nn.GELU(), nn.Linear(w, w))
        self.zq = nn.Sequential(nn.LayerNorm(d_pfn), nn.Linear(d_pfn, w)) if s.query_state else None
        self.blocks = nn.ModuleList([_Block(s, d_pfn) for _ in self.read])
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Linear(w, n_out))

    def _x(self, x: Tensor) -> Tensor:
        nf = torch.full((len(x),), x.shape[-1], device=x.device, dtype=torch.long)
        return _pad_and_rescale_features(x, self.max_x_dim, nf)

    def encode(self, Z: Tensor, X: Tensor, budget: Tensor) -> list[Tensor]:
        """Z [B, L+1, n, d_pfn] (`probes.context_states`), X [B, n, d], budget [B, 3] -> the acquisition tokens
        after every block ([B, n + 1, width] each; the last token is the budget token)."""
        d = X.shape[-1]
        a = self.loc(self._x(X)) + self.dim_emb.weight[d]
        a = torch.cat([a, self.budget_tok(budget)[:, None]], 1)
        mem = []
        for blk, layer in zip(self.blocks, self.read, strict=True):
            a = cast(_Block, blk).context(a, Z[:, layer])
            mem.append(a)
        return mem

    def score(self, mem: list[Tensor], xq: Tensor, budget: Tensor, zq: Tensor | None = None) -> Tensor:
        """xq [B, Q, d] -> scores [B, Q, n_out]."""
        q = self.qloc(self._x(xq)) + self.dim_emb.weight[xq.shape[-1]] + self.budget_q(budget)[:, None]
        if self.zq is not None:
            assert zq is not None
            q = q + self.zq(zq)
        for blk, a in zip(self.blocks, mem, strict=True):
            q = cast(_Block, blk).query(q, a)
        return self.out(q)


@torch.no_grad()
def pfn_inputs(engine: Engine, X: Tensor, Y: Tensor, xq: Tensor, query_state: bool) -> tuple[Tensor, Tensor | None]:
    """The frozen PFN's context states [B, L+1, n, d_pfn] and, for the ablation, its query states [B, Q, d_pfn]."""
    pfn = engine.pfn
    assert pfn is not None
    Z = context_states(pfn, X, Y)
    zq = None
    if query_state:
        nf = torch.full((len(X),), X.shape[-1], device=X.device, dtype=torch.long)
        zq = query_states(pfn, pfn.context_kv(X, Y, n_features=nf), xq, nf).float()
    return Z.float(), zq


def branch_scores(
    branch: AcqBranch, engine: Engine, X: Tensor, Y: Tensor, xq: Tensor, budget: Tensor, chunk: int = 64
) -> Tensor:
    """[B, Q, n_out] in chunks of lanes (gradients flow into the branch only)."""
    out = []
    for a in range(0, len(X), chunk):
        s = slice(a, a + chunk)
        Z, zq = pfn_inputs(engine, X[s], Y[s], xq[s], branch.spec.query_state)
        out.append(branch.score(branch.encode(Z, X[s], budget[s]), xq[s], budget[s], zq))
    return torch.cat(out)


# --------------------------------------------------------------------------------------------------
# teachers: the arms' acquisition values
# --------------------------------------------------------------------------------------------------


@torch.no_grad()
def teacher(
    engine: Engine, arms: list[str], X: Tensor, Y: Tensor, extra: Tensor | None = None, s_ref: int = 0
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Queries = the seed-`s_ref` candidate set C0 of each state, followed by `extra` [B, E, d].
    -> (xq [B, P0 + E, d], scores [B, P0 + E, A] (higher = better), valid [B, P0 + E], mean [B, P0 + E]).
    MES's f* samples come from C0 for every query, so a query's score depends on the data and x only, and on C0
    the argmax is exactly `options.py::arm_pick(..., s=s_ref)`."""
    bd = _bd(engine)
    out: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for a in range(0, len(X), engine.chunk):
        Xa, Ya = X[a : a + engine.chunk], Y[a : a + engine.chunk]
        C0 = seeded_candidates(engine, Xa, Ya, s_ref)
        xq = C0 if extra is None else torch.cat([C0, extra[a : a + engine.chunk]], 1)
        lg, _ = pfn_pass(engine, Xa, Ya, xq, with_z=False)
        inc = Ya.min(1).values
        fs = fstar_quantiles(bd, lg[:, : C0.shape[1]], inc) if any(parse_arm(m)[0] == "mes" for m in arms) else None
        sc = torch.stack([arm_scores(m, bd, lg, inc, engine.ucb_quantile, mes_fs=fs) for m in arms], -1)
        out.append((xq, sc.float(), torch.cdist(xq, Xa).min(-1).values > 1e-7, bd.mean(lg).float()))
        del lg
    return (
        torch.cat([o[0] for o in out]),
        torch.cat([o[1] for o in out]),
        torch.cat([o[2] for o in out]),
        torch.cat([o[3] for o in out]),
    )


def extra_queries(X: Tensor, Y: Tensor, n: int, gen: torch.Generator) -> Tensor:
    """[B, n, d] fresh queries: half uniform, a quarter perturbations of the incumbent and a quarter of random
    observed points, with log-uniform scales in [1e-3, 0.3]. Drawn on the CPU from `gen`."""
    B, m, d = X.shape
    dev = X.device
    n_u = n // 2
    n_i = (n - n_u) // 2
    n_c = n - n_u - n_i
    u = torch.rand(B, n_u, d, generator=gen).to(dev)
    sig = (10 ** (-3 + 2.5 * torch.rand(B, n_i + n_c, 1, generator=gen))).to(dev)
    z = torch.randn(B, n_i + n_c, d, generator=gen).to(dev)
    r = torch.arange(B, device=dev)
    inc = X[r, Y.argmin(1)][:, None].expand(-1, n_i, -1)
    idx = torch.randint(0, m, (B, n_c), generator=gen).to(dev)
    pts = torch.gather(X, 1, idx[..., None].expand(-1, -1, d))
    return torch.cat([u, (torch.cat([inc, pts], 1) + sig * z).clamp(0, 1)], 1)


# --------------------------------------------------------------------------------------------------
# ranking loss and agreement
# --------------------------------------------------------------------------------------------------


def rank_loss(
    s: Tensor, t: Tensor, valid: Tensor, gen: torch.Generator, n_pairs: int = 4, top_k: int = 8
) -> tuple[Tensor, dict[str, float]]:
    """s, t [B, Q, A] (student, teacher; higher = better), valid [B, Q]. Mean over arms of: top-1 cross-entropy
    + pairwise logistic loss on random pairs + on (teacher top-k, any) pairs. Ties in the teacher are skipped."""
    B, Q, A = s.shape
    v = valid[..., None]
    tm = t.masked_fill(~v, float("-inf"))
    sm = s.masked_fill(~v, -1e9)
    top = tm.argmax(1)  # [B, A]
    l_top = Fn.cross_entropy(sm.transpose(1, 2).reshape(B * A, Q), top.reshape(-1))

    def pair(si: Tensor, ti: Tensor, vi: Tensor, sj: Tensor, tj: Tensor, vj: Tensor) -> Tensor:
        w = (vi & vj & (ti != tj)).float()
        loss = Fn.binary_cross_entropy_with_logits(si - sj, (ti > tj).float(), reduction="none")
        return (loss * w).sum() / w.sum().clamp(min=1.0)

    l_pair = s.new_zeros(())
    for _ in range(n_pairs):
        j = torch.randint(0, Q, (B, Q), generator=gen).to(s.device)
        ja = j[..., None].expand(-1, -1, A)
        l_pair = l_pair + pair(s, t, v, s.gather(1, ja), t.gather(1, ja), valid.gather(1, j)[..., None]) / n_pairs
    k = min(top_k, Q)
    ti = tm.topk(k, 1).indices  # [B, k, A]
    s_top, t_top = s.gather(1, ti), t.gather(1, ti)
    v_top = torch.isfinite(tm.gather(1, ti))
    l_topk = pair(s_top[:, :, None], t_top[:, :, None], v_top[:, :, None], s[:, None], t[:, None], v[:, None])
    loss = l_top + l_pair + l_topk
    return loss, {"top1_ce": float(l_top.detach()), "pair": float(l_pair.detach()), "topk_pair": float(l_topk.detach())}


def _ranks(x: Tensor) -> Tensor:
    return x.argsort(1).argsort(1).float()


def _corr(a: Tensor, b: Tensor, w: Tensor) -> Tensor:
    """Weighted Pearson correlation along dim 1; a, b, w [B, Q, A]."""
    n = w.sum(1, keepdim=True).clamp(min=1.0)
    am, bm = a - (a * w).sum(1, keepdim=True) / n, b - (b * w).sum(1, keepdim=True) / n
    num = (am * bm * w).sum(1)
    den = ((am**2 * w).sum(1) * (bm**2 * w).sum(1)).sqrt().clamp(min=1e-12)
    return num / den


@torch.no_grad()
def agreement(s: Tensor, t: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Per state and arm [B, A]: `top1` (same argmax), `rank_pct` (share of valid candidates the teacher ranks
    above the student's pick; 0 = the teacher's best), `gap` (teacher's best minus teacher's value of the student's
    pick, in units of best minus median), `spearman` over the valid candidates, `spearman_top` over the teacher's
    top quarter."""
    v = valid[..., None]
    tm = t.masked_fill(~v, float("-inf"))
    sm = s.masked_fill(~v, float("-inf"))
    pick, best = sm.argmax(1), tm.argmax(1)  # [B, A]
    t_pick = t.gather(1, pick[:, None])[:, 0]
    t_best = t.gather(1, best[:, None])[:, 0]
    n_valid = valid.sum(1, keepdim=True).float()
    rank_pct = ((tm > t_pick[:, None]) & v).sum(1).float() / n_valid
    t_nan = t.masked_fill(~v, float("nan"))
    med = t_nan.nanmedian(1).values
    gap = (t_best - t_pick) / (t_best - med).clamp(min=1e-30)
    w = v.float().expand_as(t)
    rt, rs = _ranks(tm), _ranks(sm)
    q75 = torch.nanquantile(t_nan, 0.75, dim=1, keepdim=True)
    w_top = w * (t >= q75).float()
    return {
        "top1": (pick == best).float(),
        "rank_pct": rank_pct,
        "gap": gap,
        "spearman": _corr(rt, rs, w),
        "spearman_top": _corr(
            _ranks(tm.masked_fill(w_top == 0, float("-inf"))), _ranks(sm.masked_fill(w_top == 0, float("-inf"))), w_top
        ),
    }


# --------------------------------------------------------------------------------------------------
# the branch as a BO policy
# --------------------------------------------------------------------------------------------------


def branch_policy(branch: AcqBranch, engine: Engine, cols: Tensor, T: int, s: int = 0):
    """A `Policy` for `experimental/api.py::rollout`: lane l picks the argmax of the branch's column cols[l] over
    the seed-s candidates (ties to the lower posterior mean, invalid candidates excluded, as the arms do)."""

    @torch.no_grad()
    def pol(X: Tensor, Y: Tensor, rows: Tensor, reps: Tensor, t: int) -> Tensor:
        out = []
        bd = _bd(engine)
        for a in range(0, len(X), engine.chunk):
            Xa, Ya = X[a : a + engine.chunk], Y[a : a + engine.chunk]
            C = seeded_candidates(engine, Xa, Ya, s)
            budget = budget_features(t, T, Xa.shape[1], len(Xa), X.device)
            sc = branch_scores(branch, engine, Xa, Ya, C, budget)
            sc = sc.gather(2, cols[a : a + engine.chunk, None, None].expand(-1, C.shape[1], 1))[..., 0]
            lg, _ = pfn_pass(engine, Xa, Ya, C, with_z=False)
            valid = torch.cdist(C, Xa).min(-1).values > 1e-7
            r = torch.arange(len(Xa), device=X.device)
            out.append(C[r, _argmax(sc, bd.mean(lg), valid)])
            del lg
        return torch.cat(out)

    return pol

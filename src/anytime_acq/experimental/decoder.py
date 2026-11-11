"""The 0006 value decoder (research_log/decisions/0006 §3), Phase A (noise-free `pfn_large`).

A PFN-style query decoder replaces round 1's feature MLP + FiLM + set transformer (experimental/api.py,
kept there as the ablation):
- *candidate tokens*: a linear embedding of set-independent features of the candidate x (x; the PFN's
  marginal at x relative to the incumbent and scaled by the std of the observed y; EI, PI; distances; source
  flag). No candidate–candidate attention: Q(s, x) depends on the state and x only;
- *memory* at decoder block b: the PFN's residual context states of layer `read[b]` (the decoder's own
  LayerNorm + projection, §1), plus three memory tokens: a state token (budget), a statistics token (round 1's
  global features) and an x0 token (features of EI's pick, the reference of the advantage);
- *budget conditioning*: adaptive LayerNorm in every block (scale and shift from t/T, (T − t)/T, log T, n);
- *output*: a bar distribution (`Bars`) over the single-seed paired advantage a_k(s, x) = R(x0, k) − R(x, k)
  (% of the task's gap; positive = better than EI), trained by NLL on every seed's target (§3 "Target and loss").

Decision rule (§3): deviate to the candidate with the largest score if that score is above `zero_eps` (the
upper edge of the bin around 0), else x0. The score is the predicted mean ("mean") or a quantile of the
predicted advantage ("q0.3" = 30% quantile); the rule is calibrated on held-out instances (split 0).

Deviation from 0006 §2: the candidate set C(s) and pi_0 stay as in round 1 (x0 = Engine.propose("ei"), so the
round-1 labels remain valid); gradient-refined acquisition argmaxes would change pi_0 and need new labels.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as Fn
from torch import Tensor, nn

from ..bo.engine import Engine
from ..bo.tasks import TaskSet
from .api import SOURCES, CandidateSpec, candidate_set, f_lanes, features, pfn_pass
from .probes import context_states

N_QUERY_BASE = 11  # + d (x) + len(SOURCES) (source flag)
N_BUDGET = 4


# --------------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------------


@torch.no_grad()
def query_features(engine: Engine, X: Tensor, Y: Tensor, xq: Tensor, src: Tensor) -> Tensor:
    """[G, J, d + 11 + n_sources]: features of candidates xq [G, J, d] that do not depend on the other candidates.
    src [G, J] source ids (experimental/api.py SOURCES). In chunks of `engine.chunk` lanes."""
    c = engine.chunk
    return torch.cat(
        [
            _query_features(engine, X[s : s + c], Y[s : s + c], xq[s : s + c], src[s : s + c])
            for s in range(0, len(X), c)
        ]
    )


def _query_features(engine: Engine, X: Tensor, Y: Tensor, xq: Tensor, src: Tensor) -> Tensor:
    assert engine.pfn is not None
    bd = engine.pfn.bar_dist
    G, J, d = xq.shape
    lg, _ = pfn_pass(engine, X, Y, xq, with_z=False)
    inc = Y.min(1).values[:, None]
    s = Y.std(1, keepdim=True).clamp(min=1e-4)
    mu, sd = bd.mean(lg), bd.variance(lg).clamp(min=0).sqrt()
    ei = bd.ei(lg, inc).clamp(min=0)
    pi = bd.pi(lg, inc)
    x_inc = X[torch.arange(G, device=X.device), Y.argmin(1)]
    f = torch.stack(
        [
            (mu - inc) / s,
            sd / s,
            (bd.icdf(lg, 0.05) - inc) / s,
            (bd.icdf(lg, 0.95) - inc) / s,
            torch.log(ei + 1e-12) - torch.log(s),
            pi,
            torch.cdist(xq, X).min(-1).values / math.sqrt(d),
            (xq - x_inc[:, None]).norm(dim=-1) / math.sqrt(d),
            mu,
            sd,
            inc.expand(-1, J),
        ],
        -1,
    )
    return torch.cat([xq, f, Fn.one_hot(src, len(SOURCES)).float()], -1)


def budget(t: int, T: int, n: int, S: int, device: torch.device | str) -> Tensor:
    """[S, 4]: t/T, (T − t)/T, log T, n/100."""
    v = torch.tensor([t / T, (T - t) / T, math.log(T), n / 100], device=device)
    return v[None].expand(S, -1)


@torch.no_grad()
def layer_states(engine: Engine, X: Tensor, Y: Tensor, read: Sequence[int]) -> Tensor:
    """[S, len(read), n, d_model]: the context tokens' residual states at the PFN layers `read` (0 = embedding)."""
    assert engine.pfn is not None
    out = []
    for s in range(0, len(X), engine.chunk):
        out.append(context_states(engine.pfn, X[s : s + engine.chunk], Y[s : s + engine.chunk])[:, list(read)])
    return torch.cat(out)


# --------------------------------------------------------------------------------------------------
# bar distribution over the advantage
# --------------------------------------------------------------------------------------------------


class Bars(nn.Module):
    """Piecewise-uniform density on `borders` [K+1]; one bin is [−zero_eps, zero_eps] (the point mass at 0)."""

    def __init__(self, borders: Tensor, zero_eps: float) -> None:
        super().__init__()
        self.register_buffer("borders", borders.float())
        self.zero_eps = zero_eps

    @staticmethod
    def from_targets(a: Tensor, n_bins: int, zero_eps: float) -> "Bars":
        """Inner borders at quantiles of the training targets a [N] (heavy tails, mass near 0), the zero bin,
        and outer borders beyond the observed range."""
        a = a.flatten().float()
        if len(a) > 1_000_000:
            a = a[torch.randperm(len(a), generator=torch.Generator().manual_seed(0))[:1_000_000]]
        lo, hi = float(a.min()), float(a.max())
        qs = torch.quantile(a.cpu(), torch.linspace(0.001, 0.999, n_bins - 3, dtype=torch.float32))
        qs = qs[(qs.abs() > 2 * zero_eps)]
        b = torch.cat(
            [qs, torch.tensor([-zero_eps, zero_eps, min(lo, -2 * zero_eps) - 1.0, max(hi, 2 * zero_eps) + 1.0])]
        )
        b = b.unique().sort().values
        keep = torch.cat([torch.tensor([True]), b.diff() > 1e-7])
        return Bars(b[keep], zero_eps)

    @property
    def n_bins(self) -> int:
        return len(self.borders) - 1

    def nll(self, logits: Tensor, y: Tensor) -> Tensor:
        b = self.borders
        y = y.clamp(float(b[0]), float(b[-1]))
        idx = torch.bucketize(y, b[1:-1].contiguous(), right=True)
        w = b.diff()
        return -logits.log_softmax(-1).gather(-1, idx[..., None])[..., 0] + w.log()[idx]

    def mean(self, logits: Tensor) -> Tensor:
        b = self.borders
        return (logits.softmax(-1) * (b[1:] + b[:-1]) / 2).sum(-1)

    def quantile(self, logits: Tensor, q: float) -> Tensor:
        b = self.borders
        p = logits.softmax(-1)
        cum = p.cumsum(-1)
        i = torch.searchsorted(cum, torch.full_like(cum[..., :1], q)).clamp(max=self.n_bins - 1)
        c_hi = cum.gather(-1, i)
        p_i = p.gather(-1, i).clamp(min=1e-12)
        frac = (1 - (c_hi - q) / p_i).clamp(0, 1)
        return (b[i] + frac * b.diff()[i])[..., 0]

    def score(self, logits: Tensor, rule: str) -> Tensor:
        return self.mean(logits) if rule == "mean" else self.quantile(logits, float(rule[1:]))


# --------------------------------------------------------------------------------------------------
# decoder
# --------------------------------------------------------------------------------------------------


class AdaLN(nn.Module):
    """LayerNorm with scale and shift from the conditioning vector (zero-initialized: starts as plain LN)."""

    def __init__(self, width: int, cond: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(width, elementwise_affine=False)
        self.mod = nn.Linear(cond, 2 * width)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        g, b = self.mod(c).chunk(2, -1)
        return self.ln(h) * (1 + g[:, None]) + b[:, None]


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, d_pfn: int) -> None:
        super().__init__()
        self.mem = nn.Sequential(nn.LayerNorm(d_pfn), nn.Linear(d_pfn, width))
        self.n1 = AdaLN(width, width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n2 = AdaLN(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, h: Tensor, ctx: Tensor, extra: Tensor, c: Tensor) -> Tensor:
        mem = torch.cat([self.mem(ctx), extra], 1)
        h = h + self.attn(self.n1(h, c), mem, mem, need_weights=False)[0]
        return h + self.ff(self.n2(h, c))


class ValueDecoder(nn.Module):
    """Candidate queries q [S, J, n_q] cross-attend to ctx [S, n_read, n, d_pfn] plus the state, statistics and
    x0 tokens; adaLN on the budget. -> logits [S, J, n_bins] of the advantage over x0."""

    def __init__(self, n_q: int, n_glob: int, d_pfn: int, bars: Bars, width: int = 256, blocks: int = 4,
                 heads: int = 4) -> None:  # fmt: skip
        super().__init__()
        self.bars = bars
        self.q_in = nn.Linear(n_q, width)
        self.cond = nn.Sequential(nn.Linear(N_BUDGET, width), nn.GELU(), nn.Linear(width, width))
        self.state_tok = nn.Linear(N_BUDGET, width)
        self.stat_tok = nn.Linear(n_glob, width)
        self.x0_tok = nn.Linear(n_q, width)
        self.blocks = nn.ModuleList([DecoderBlock(width, heads, d_pfn) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, bars.n_bins))
        for name, dim in (("q", n_q), ("g", n_glob)):
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))

    def set_normalization(self, q: Tensor, glob: Tensor) -> None:
        self.q_mean.copy_(q.mean(0))
        self.q_std.copy_(q.std(0).clamp(min=1e-3))
        self.g_mean.copy_(glob.mean(0))
        self.g_std.copy_(glob.std(0).clamp(min=1e-3))

    def forward(self, q: Tensor, ctx: Tensor, glob: Tensor, bud: Tensor, q0: Tensor) -> Tensor:
        qn = (q - self.q_mean) / self.q_std
        q0n = (q0 - self.q_mean) / self.q_std
        gn = (glob - self.g_mean) / self.g_std
        c = self.cond(bud)
        extra = torch.stack([self.state_tok(bud), self.stat_tok(gn), self.x0_tok(q0n)], 1)
        h = self.q_in(qn)
        for b, blk in enumerate(self.blocks):
            h = blk(h, ctx[:, b], extra, c)
        return self.out(h)


# --------------------------------------------------------------------------------------------------
# inputs of a state, and the policy
# --------------------------------------------------------------------------------------------------


@torch.no_grad()
def state_inputs(
    engine: Engine, spec: CandidateSpec, X: Tensor, Y: Tensor, C: Tensor, valid: Tensor, xq: Tensor, src: Tensor,
    t: int, T: int, read: Sequence[int],
) -> dict[str, Tensor]:  # fmt: skip
    """Decoder inputs for states (X, Y) [S, n, ·] and queried candidates xq [S, J, d] with source ids src [S, J]
    (xq[:, 0] = x0). The statistics token uses round 1's global features over the state's full candidate set
    (C, valid)."""
    glob = features(engine, spec, X, Y, C, valid, t, T)["glob"]
    q = query_features(engine, X, Y, xq, src)
    return {
        "q": q,
        "ctx": layer_states(engine, X, Y, read),
        "glob": glob,
        "bud": budget(t, T, X.shape[1], len(X), X.device),
        "q0": q[:, 0],
    }


@torch.no_grad()
def decoder_logits(dec: ValueDecoder, inp: dict[str, Tensor], chunk: int) -> Tensor:
    out = []
    for s in range(0, len(inp["q"]), chunk):
        sl = slice(s, s + chunk)
        out.append(dec(inp["q"][sl], inp["ctx"][sl], inp["glob"][sl], inp["bud"][sl], inp["q0"][sl]))
    return torch.cat(out)


def decoder_pick(bars: Bars, logits: Tensor, valid: Tensor, rules: Sequence[str]) -> tuple[Tensor, Tensor, Tensor]:
    """Per lane l, rule rules[l] ("ei" = never deviate, "mean", "q<level>"): the candidate with the largest score
    among valid ones if its score > zero_eps, else x0 (index 0). logits [L, C, nb]. -> index, deviated, score."""
    L = len(logits)
    score = torch.zeros(L, logits.shape[1], device=logits.device)
    for r in set(rules):
        if r == "ei":
            continue
        m = torch.tensor([x == r for x in rules], device=logits.device)
        score[m] = bars.score(logits[m], r)
    score = score.masked_fill(~valid, float("-inf"))
    score[:, 0] = 0.0
    best = score.max(1)
    never = torch.tensor([r == "ei" for r in rules], device=logits.device)
    deviated = (best.values > bars.zero_eps) & ~never
    j = torch.where(deviated, best.indices, torch.zeros_like(best.indices))
    return j, deviated, score.gather(1, j[:, None])[:, 0]


@torch.no_grad()
def run_decoder(
    engine: Engine, spec: CandidateSpec, dec: ValueDecoder, tasks: TaskSet, X0: Tensor, T: int,
    rules: Sequence[str], read: Sequence[int],
) -> dict[str, Tensor]:  # fmt: skip
    """Every task under the decoder policy with each rule (replicate 0, CRN across rules; "ei" = pi_0).
    -> F [B, K, n], deviated [B, K, T], chosen source [B, K, T], score of the chosen candidate [B, K, T]."""
    B, n_init, d = X0.shape
    K = len(rules)
    dev = X0.device
    rows = torch.arange(B, device=dev).repeat_interleave(K)
    reps = torch.zeros(B * K, dtype=torch.long, device=dev)
    lane_rules = list(rules) * B
    X = X0.repeat_interleave(K, 0)
    F = f_lanes(tasks, X.reshape(-1, d), rows.repeat_interleave(n_init)).view(B * K, n_init)
    src_ids = spec.source_ids().to(dev)
    devs, srcs, scores = [], [], []
    for t in range(T):
        C, valid = candidate_set(engine, spec, X, F, rows, reps, t, B)
        src = src_ids[None].expand(len(X), -1)
        logits = torch.cat(
            [
                decoder_logits(
                    dec,
                    state_inputs(engine, spec, X[sl], F[sl], C[sl], valid[sl], C[sl], src[sl], t, T, read),
                    len(X),
                )
                for sl in (slice(s, s + engine.chunk) for s in range(0, len(X), engine.chunk))
            ]
        )
        j, deviated, sc = decoder_pick(dec.bars, logits, valid, lane_rules)
        x = C[torch.arange(len(X), device=dev), j]
        X = torch.cat([X, x[:, None]], 1)
        F = torch.cat([F, f_lanes(tasks, x, rows)[:, None]], 1)
        devs.append(deviated)
        srcs.append(src_ids[j])
        scores.append(sc)
    return {
        "F": F.view(B, K, -1),
        "deviated": torch.stack(devs, 1).view(B, K, T),
        "source": torch.stack(srcs, 1).view(B, K, T),
        "score": torch.stack(scores, 1).view(B, K, T),
    }

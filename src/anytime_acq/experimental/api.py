"""Amortized policy improvement, round 1 (research_log/decisions/0003-amortized-policy-improvement.md).

Base policy pi_0: EI on the frozen PFN, on the engine's candidate generator (`bo/engine.py`). On the
noise-free prior (P0) latent EI and EI coincide. Here pi_0 is `base_picks(...)[:, 0]`, which reproduces
`Engine.propose(..., "ei")` (same candidates, same scores, same ties; tested). Every run of this module
uses it, so the base pick x0 inside the improved policy *is* pi_0's action.

At a state s = (D, t, T):
1. `candidate_set`: index 0 = x0; then the argmaxes of HEURISTICS on the engine's candidates; then
   `n_global` fresh Cranley–Patterson-shifted Sobol points; then `n_local_per_point` Gaussian
   perturbations of each of the `n_top` best evaluated points, scales cycling over `local_sigmas`,
   reflected at the box. All randomness comes from `bo/nuisance.py`, keyed by (step, replicate).
2. `features`: the PFN's final query-token state z(x) (the frozen weights, along the path of
   `logits_from_context_kv`; tested to reproduce its logits), bar-distribution summaries relative to the
   incumbent, distances, source, within-set ranks; and global budget / belief features.
3. `ValueHead`: candidate encoder -> FiLM(global) -> set transformer over candidates -> (g, sigma);
   A_hat(x) = g(x) - g(x0), so A_hat(x0) = 0 exactly.
4. `improved_pick`, the conservative rule: deviate to argmax_x [A_hat - c sigma] only if that maximum is
   > 0, else x0. With c = inf it is pi_0.

Labels (`generate_labels`, 0003 §5.2): one branch state per behaviour rollout (pi_0 for most instances, a
heuristic for the rest: off-policy coverage); x0 plus `n_labeled - 1` candidates (heuristic argmaxes,
the top by EI, then random); each queried, then continued with pi_0 to T under `n_cont` continuation
replicates shared across candidates (common random numbers). Label = mean_k [R(x0, k) - R(c, k)] with R
the final regret in % of the task's gap f_med - f* (positive = c is better), and its standard error.
Labels use the true task; the head only ever sees F_t-measurable inputs (0003 §1, privileged-information
rules).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as Fn
from pfn_repro.acquisitions.standard import arm_scores
from pfn_repro.surrogates.pfn import _pad_and_rescale_features
from torch import Tensor, nn

from ..bo import nuisance
from ..bo.engine import Engine
from ..bo.tasks import TaskSet

HEURISTICS = ("ucb0.9", "ucb0.99", "pi", "mean")
SOURCES = ("base", *HEURISTICS, "global", "local")
SOBOL_SEED = 23


@dataclass
class CandidateSpec:
    n_global: int = 64
    n_top: int = 4
    n_local_per_point: int = 16
    local_sigmas: tuple[float, ...] = (0.01, 0.03, 0.1)

    @property
    def size(self) -> int:
        return 1 + len(HEURISTICS) + self.n_global + self.n_top * self.n_local_per_point

    def source_ids(self) -> Tensor:
        """[C] index into SOURCES."""
        ids = [0, *range(1, 1 + len(HEURISTICS))]
        ids += [len(HEURISTICS) + 1] * self.n_global + [len(HEURISTICS) + 2] * (self.n_top * self.n_local_per_point)
        return torch.tensor(ids)


# --------------------------------------------------------------------------------------------------
# the frozen PFN's query path
# --------------------------------------------------------------------------------------------------


def query_states(pfn: nn.Module, kv: list[tuple[Tensor, Tensor]], x: Tensor, nf: Tensor) -> Tensor:
    """Final query-token states z [B, Q, d_model] (before `out_ln`), along the exact path of
    `PFN.logits_from_context_kv`: logits = out_head(out_ln(z)). Reads the frozen weights only."""
    xs = _pad_and_rescale_features(x, pfn.max_x_dim, nf)
    tok = pfn.test_x_embed(xs) + pfn.test_placeholder.view(1, 1, -1)
    B, Q, Dm = tok.shape
    for blk, (K, V) in zip(pfn.blocks, kv, strict=True):
        a = blk.attn
        N = K.shape[1]
        q = a.q_proj(blk.ln1(tok)).view(B, Q, a.h, a.dk).transpose(1, 2)
        k = K.view(B, N, a.h, a.dk).transpose(1, 2)
        v = V.view(B, N, a.h, a.dk).transpose(1, 2)
        att = ((q @ k.transpose(-2, -1)) / (a.dk**0.5)).softmax(-1)
        tok = tok + a.out_proj((att @ v).transpose(1, 2).reshape(B, Q, Dm))
        tok = tok + blk.ff(blk.ln2(tok))
    return tok


@torch.no_grad()
def pfn_pass(engine: Engine, X: Tensor, Y: Tensor, C: Tensor, with_z: bool) -> tuple[Tensor, Tensor | None]:
    """Logits [L, P, nb] (and z [L, P, d_model]) of candidates C [L, P, d] given contexts (X, Y), in
    chunks of `engine.chunk` lanes. Logits come from `logits_from_context_kv`, as in `Engine.score`."""
    pfn = engine.pfn
    assert pfn is not None
    lgs, zs = [], []
    for s in range(0, len(X), engine.chunk):
        Xa, Ya, Ca = X[s : s + engine.chunk], Y[s : s + engine.chunk], C[s : s + engine.chunk]
        nf = torch.full((len(Xa),), Xa.shape[-1], device=X.device, dtype=torch.long)
        kv = pfn.context_kv(Xa, Ya, n_features=nf)
        lgs.append(pfn.logits_from_context_kv(kv, Ca, n_features=nf).float())
        if with_z:
            zs.append(query_states(pfn, kv, Ca, nf).float())
    return torch.cat(lgs), (torch.cat(zs) if with_z else None)


def _argmax(score: Tensor, tie_key: Tensor, valid: Tensor) -> Tensor:
    """`Engine.propose`'s rule: best valid score, ties (relative 1e-7) to the lower posterior mean."""
    neg = torch.finfo(score.dtype).min
    sc = score.masked_fill(~valid, neg)
    best = sc.max(-1, keepdim=True).values
    tied = valid & (sc >= best - 1e-7 * best.abs().clamp(min=1e-12))
    return tie_key.masked_fill(~tied, float("inf")).argmin(-1)


@torch.no_grad()
def base_picks(engine: Engine, X: Tensor, Y: Tensor, rows: Tensor, reps: Tensor, step: int, n_tasks: int) -> Tensor:
    """[L, 1 + len(HEURISTICS), d]: the argmax of ei (= pi_0) and of each heuristic on the engine's
    candidate set at (step, replicate), from one PFN pass."""
    cand = engine.candidate_set(X, Y, rows, reps, step, n_tasks)
    assert engine.pfn is not None
    bd = engine.pfn.bar_dist
    out = []
    for s in range(0, len(X), engine.chunk):  # logits are [lanes, P, nb]: never for all lanes at once
        Xa, Ya, Ca = X[s : s + engine.chunk], Y[s : s + engine.chunk], cand[s : s + engine.chunk]
        lg, _ = pfn_pass(engine, Xa, Ya, Ca, with_z=False)
        inc = Ya.min(1).values
        tie = bd.mean(lg)
        valid = torch.cdist(Ca, Xa).min(-1).values > 1e-7
        r = torch.arange(len(Xa), device=X.device)
        arms = ("ei", *HEURISTICS)
        out.append(
            torch.stack([Ca[r, _argmax(arm_scores(a, bd, lg, inc, engine.ucb_quantile), tie, valid)] for a in arms], 1)
        )
        del lg
    return torch.cat(out)


def pi0(engine: Engine, n_tasks: int) -> Callable[[Tensor, Tensor, Tensor, Tensor, int], Tensor]:
    return lambda X, Y, rows, reps, t: base_picks(engine, X, Y, rows, reps, t, n_tasks)[:, 0]


def _reflect(x: Tensor) -> Tensor:
    x = x.abs()
    return (1 - (1 - x).abs()).clamp(0, 1)


@torch.no_grad()
def candidate_set(
    engine: Engine, spec: CandidateSpec, X: Tensor, Y: Tensor, rows: Tensor, reps: Tensor, step: int, n_tasks: int
) -> tuple[Tensor, Tensor]:
    """-> C [L, spec.size, d] (index 0 = pi_0's pick) and valid [L, spec.size] (not evaluated, not a
    duplicate of x0; index 0 always valid)."""
    L, n, d = X.shape
    dev = X.device
    picks = base_picks(engine, X, Y, rows, reps, step, n_tasks)
    shift = nuisance.stream(engine.base_seed, "api_global", step, reps, rows, (d,), n_tasks, False, dev)
    sob = torch.quasirandom.SobolEngine(d, scramble=True, seed=SOBOL_SEED).draw(spec.n_global).to(dev)
    glob = (sob[None] + shift[:, None]) % 1.0
    k = min(spec.n_top, n)
    top = X.gather(1, Y.topk(k, dim=1, largest=False).indices[..., None].expand(-1, -1, d))  # [L, k, d]
    if k < spec.n_top:  # fewer evaluated points than n_top: repeat the best
        top = torch.cat([top, top[:, :1].expand(-1, spec.n_top - k, -1)], 1)
    z = nuisance.stream(
        engine.base_seed, "api_local", step, reps, rows, (spec.n_top, spec.n_local_per_point, d), n_tasks, True, dev
    )
    sig = torch.tensor(spec.local_sigmas, device=dev).repeat(math.ceil(spec.n_local_per_point / len(spec.local_sigmas)))
    loc = _reflect(top[:, :, None] + sig[: spec.n_local_per_point, None] * z).reshape(L, -1, d)
    C = torch.cat([picks, glob, loc], 1)
    valid = torch.cdist(C, X).min(-1).values > 1e-7
    valid &= (C - C[:, :1]).norm(dim=-1) > 1e-7
    valid[:, 0] = True
    return C, valid


# --------------------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------------------

N_CAND_FEATURES = 13 + len(SOURCES)
N_GLOBAL_FEATURES = 11


def _rank(v: Tensor, valid: Tensor) -> Tensor:
    """Within-set rank in [0, 1] (0 = smallest) over valid candidates; invalid ones get 1."""
    big = torch.finfo(v.dtype).max
    order = v.masked_fill(~valid, big).argsort(-1).argsort(-1).float()
    return order / (valid.sum(-1, keepdim=True) - 1).clamp(min=1)


@torch.no_grad()
def features(
    engine: Engine, spec: CandidateSpec, X: Tensor, Y: Tensor, C: Tensor, valid: Tensor, t: int, T: int
) -> dict[str, Tensor]:
    """Per-candidate z [L, C, d_model] and cand [L, C, N_CAND_FEATURES]; glob [L, N_GLOBAL_FEATURES];
    ei [L, C]. Value features are relative to the incumbent and divided by s, the std of the posterior
    mean over the candidate set (0003 §4.2); the absolute incumbent level is a global feature (on P0 it is
    informative, and the PFN sees it as well). In chunks of `engine.chunk` lanes."""
    parts = [
        _features(
            engine,
            spec,
            X[s : s + engine.chunk],
            Y[s : s + engine.chunk],
            C[s : s + engine.chunk],
            valid[s : s + engine.chunk],
            t,
            T,
        )
        for s in range(0, len(X), engine.chunk)
    ]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def _features(
    engine: Engine, spec: CandidateSpec, X: Tensor, Y: Tensor, C: Tensor, valid: Tensor, t: int, T: int
) -> dict[str, Tensor]:
    assert engine.pfn is not None
    bd = engine.pfn.bar_dist
    L, _, d = C.shape
    lg, z = pfn_pass(engine, X, Y, C, with_z=True)
    assert z is not None
    inc = Y.min(1).values
    mu, sd = bd.mean(lg), bd.variance(lg).clamp(min=0).sqrt()
    q05, q95 = bd.icdf(lg, 0.05), bd.icdf(lg, 0.95)
    ei = bd.ei(lg, inc[:, None]).clamp(min=0)
    pi = bd.pi(lg, inc[:, None])
    vm = valid.float()
    s = (
        (
            ((mu - (mu * vm).sum(1, keepdim=True) / vm.sum(1, keepdim=True)) ** 2 * vm).sum(1, keepdim=True)
            / vm.sum(1, keepdim=True)
        )
        .sqrt()
        .clamp(min=1e-4)
    )
    rel = lambda v: (v - inc[:, None]) / s  # noqa: E731
    dist_data = torch.cdist(C, X).min(-1).values / math.sqrt(d)
    x_inc = X[torch.arange(L, device=X.device), Y.argmin(1)]
    dist_inc = (C - x_inc[:, None]).norm(dim=-1) / math.sqrt(d)
    log_ei = torch.log(ei + 1e-12) - torch.log(s)
    src = Fn.one_hot(spec.source_ids().to(C.device), len(SOURCES)).float()[None].expand(L, -1, -1)
    cand = torch.cat(
        [
            torch.stack(
                [
                    rel(mu),
                    sd / s,
                    rel(q05),
                    rel(q95),
                    log_ei,
                    pi,
                    _rank(mu, valid),
                    _rank(-sd, valid),
                    _rank(-ei, valid),
                    dist_data,
                    dist_inc,
                    mu,
                    sd,
                ],
                -1,
            ),
            src,
        ],
        -1,
    )
    n = X.shape[1]
    since = (n - 1 - Y.argmin(1)).float() / T
    fill = (dist_data * vm).max(1).values
    glob = torch.stack(
        [
            torch.full_like(inc, t / T),
            torch.full_like(inc, (T - t) / T),
            torch.full_like(inc, n / 100),
            inc,
            s[:, 0],
            (sd * vm).sum(1) / vm.sum(1),
            torch.log((ei * vm).max(1).values + 1e-12),
            (pi * vm).sum(1) / vm.sum(1),
            ((pi > 0.05).float() * vm).sum(1) / vm.sum(1),
            since,
            fill,
        ],
        -1,
    )
    return {"z": z, "cand": cand, "glob": glob, "ei": ei}


# --------------------------------------------------------------------------------------------------
# value head
# --------------------------------------------------------------------------------------------------


class ValueHead(nn.Module):
    """Candidate encoder (z and features) -> FiLM from the global vector -> set transformer -> (g, sigma).
    A_hat = g - g[:, 0] (the base pick, index 0, scores exactly 0); sigma > 0 is the predicted std of the
    label around A_hat (label units: % of the task's gap)."""

    def __init__(self, d_z: int, width: int = 256, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.z_proj = nn.Sequential(nn.LayerNorm(d_z), nn.Linear(d_z, width))
        self.c_mlp = nn.Sequential(nn.Linear(N_CAND_FEATURES, width), nn.GELU(), nn.Linear(width, width))
        self.g_mlp = nn.Sequential(nn.Linear(N_GLOBAL_FEATURES, width), nn.GELU(), nn.Linear(width, 2 * width))
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.set = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))
        for name, dim in (("c", N_CAND_FEATURES), ("g", N_GLOBAL_FEATURES)):
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))

    def set_normalization(self, cand: Tensor, glob: Tensor, valid: Tensor) -> None:
        """Standardize inputs with statistics of the training set (valid candidates only)."""
        c = cand[valid]
        self.c_mean.copy_(c.mean(0))
        self.c_std.copy_(c.std(0).clamp(min=1e-3))
        self.g_mean.copy_(glob.mean(0))
        self.g_std.copy_(glob.std(0).clamp(min=1e-3))

    def forward(self, z: Tensor, cand: Tensor, glob: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        c = (cand - self.c_mean) / self.c_std
        g = (glob - self.g_mean) / self.g_std
        h = Fn.gelu(self.z_proj(z) + self.c_mlp(c))
        gamma, beta = self.g_mlp(g).chunk(2, -1)
        h = h * (1 + gamma[:, None]) + beta[:, None]
        h = self.set(h, src_key_padding_mask=~valid)
        out = self.out(h)
        g_val, log_s = out[..., 0], out[..., 1]
        return g_val - g_val[:, :1], Fn.softplus(log_s) + 1e-3


def label_nll(A: Tensor, sigma: Tensor, a: Tensor, se: Tensor) -> Tensor:
    """Gaussian NLL of labels a with the label's own standard error added to the predicted variance
    (0003 §5.3); all [N]."""
    var = sigma**2 + se**2
    return 0.5 * ((a - A) ** 2 / var + var.log())


def improved_pick(A: Tensor, sigma: Tensor, valid: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
    """The conservative rule (0003 §3.3). A, sigma, valid [L, C]; c [L] (inf = never deviate).
    -> chosen index [L] (0 = the base pick) and whether the lane deviated [L]."""
    lcb = A - torch.where(torch.isinf(c), torch.zeros_like(c), c)[:, None] * sigma
    lcb = lcb.masked_fill(~valid, float("-inf"))
    lcb[:, 0] = 0.0
    best = lcb.max(1)
    deviate = (best.values > 0) & ~torch.isinf(c)
    return torch.where(deviate, best.indices, torch.zeros_like(best.indices)), deviate


# --------------------------------------------------------------------------------------------------
# rollouts on ground truth
# --------------------------------------------------------------------------------------------------


def f_lanes(tasks: TaskSet, x: Tensor, rows: Tensor) -> Tensor:
    """True f at x [L, d] for lanes with task rows[l]: one prior evaluation per distinct task (lanes
    packed), instead of a per-lane copy of the task's weights."""
    uniq, inv = rows.unique(return_inverse=True)
    counts = torch.bincount(inv, minlength=len(uniq))
    m = int(counts.max())
    order = inv.argsort(stable=True)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.empty_like(inv)
    slot[order] = torch.arange(len(inv), device=x.device) - starts[inv[order]]
    packed = x.new_zeros(len(uniq), m, x.shape[-1])
    packed[inv, slot] = x
    return tasks.f(packed, uniq)[inv, slot]


Policy = Callable[[Tensor, Tensor, Tensor, Tensor, int], Tensor]


def rollout(
    tasks: TaskSet, X: Tensor, F: Tensor, rows: Tensor, reps: Tensor, t0: int, t1: int, policy: Policy
) -> tuple[Tensor, Tensor]:
    """Steps t0..t1-1 on the true f (noise-free): policy(X, F, rows, reps, t) -> x [L, d]."""
    for t in range(t0, t1):
        x = policy(X, F, rows, reps, t)
        X = torch.cat([X, x[:, None]], 1)
        F = torch.cat([F, f_lanes(tasks, x, rows)[:, None]], 1)
    return X, F


@dataclass
class LabelSpec:
    n_labeled: int = 16
    n_cont: int = 4
    behaviour: dict[str, float] = field(default_factory=lambda: {"ei": 0.8, "ucb0.99": 0.1, "pi": 0.1})


def behaviour_policy(engine: Engine, arms: list[str], n_tasks: int) -> Policy:
    """Lane l follows arms[rows[l]]: ei (pi_0) or a heuristic, from one PFN pass."""
    col = {a: i for i, a in enumerate(("ei", *HEURISTICS))}

    def pol(X: Tensor, Y: Tensor, rows: Tensor, reps: Tensor, t: int) -> Tensor:
        picks = base_picks(engine, X, Y, rows, reps, t, n_tasks)
        idx = torch.tensor([col[arms[int(r)]] for r in rows], device=X.device)
        return picks[torch.arange(len(X), device=X.device), idx]

    return pol


def choose_labeled(valid: Tensor, ei: Tensor, n_labeled: int, u: Tensor) -> Tensor:
    """[L, n_labeled] candidate indices: 0 (x0), the valid heuristic argmaxes, then the top 4 by EI
    among the rest, then uniformly random valid candidates (u [L, C] uniforms). Invalid ones only when
    there are not enough valid candidates (then their labels are masked)."""
    _, C = valid.shape
    key = u.clone()  # random order in [0, 1)
    key[:, 1 : 1 + len(HEURISTICS)] = -2.0
    rest = torch.arange(C, device=valid.device) >= 1 + len(HEURISTICS)
    top_ei = ei.masked_fill(~(valid & rest[None]), float("-inf")).topk(4, dim=1).indices
    key.scatter_(1, top_ei, -1.0)
    key = key.masked_fill(~valid, 2.0)
    key[:, 0] = -3.0
    return key.argsort(1)[:, :n_labeled]


def normalized_regret(F: Tensor, fstar: Tensor, fmed: Tensor) -> Tensor:
    """Final regret of the best evaluated point (noise-free: the recommendation rule of 0003 §1) in % of
    the task's gap f_med - f*. F [L, n]; fstar, fmed [L]."""
    return 100 * (F.min(1).values - fstar).clamp(min=0) / (fmed - fstar).clamp(min=1e-9)


@torch.no_grad()
def task_median(tasks: TaskSet, n: int = 4096) -> Tensor:
    """[B]: median of f over a scrambled-Sobol design (the denominator of normalized regret)."""
    x = torch.quasirandom.SobolEngine(tasks.d, scramble=True, seed=29).draw(n).to(tasks.fstar.device)
    return tasks.f(x[None].expand(tasks.B, -1, -1)).median(1).values


@torch.no_grad()
def generate_labels(
    engine: Engine,
    spec: CandidateSpec,
    lspec: LabelSpec,
    tasks: TaskSet,
    n_init: int,
    T: int,
    X0: Tensor,
    behaviour: list[str],
    branch_t: Tensor,
    label_u_seed: int,
) -> dict[str, Tensor]:
    """Labels for every task of `tasks`: one branch state each (task b at step branch_t[b] of its
    behaviour rollout). -> per task: traj X [B, n_init + T, d] and F (the behaviour rollout), the
    candidate set C [B, C, d] and valid, labeled indices [B, n_labeled], R [B, n_labeled, n_cont],
    adv and se [B, n_labeled] (masked where not valid: `lab_valid`), fstar, fmed [B]."""
    B, dev = tasks.B, X0.device
    rows = torch.arange(B, device=dev)
    reps0 = torch.zeros(B, dtype=torch.long, device=dev)
    X, F = rollout(
        tasks,
        X0,
        f_lanes(tasks, X0.reshape(-1, X0.shape[-1]), rows.repeat_interleave(n_init)).view(B, -1),
        rows,
        reps0,
        0,
        T,
        behaviour_policy(engine, behaviour, B),
    )
    nC, nl, M = spec.size, lspec.n_labeled, lspec.n_cont
    out_C = torch.empty(B, nC, X.shape[-1], device=dev)
    out_valid = torch.empty(B, nC, dtype=torch.bool, device=dev)
    out_lab = torch.empty(B, nl, dtype=torch.long, device=dev)
    out_R = torch.empty(B, nl, M, device=dev)
    fmed = task_median(tasks)
    for t in branch_t.unique().tolist():
        g = (branch_t == t).nonzero().squeeze(1).to(dev)
        Xs, Fs = X[g, : n_init + t], F[g, : n_init + t]
        C, valid = candidate_set(engine, spec, Xs, Fs, g, reps0[g], t, B)
        ei = features_ei(engine, Xs, Fs, C)
        gen = torch.Generator().manual_seed(label_u_seed * 1_009 + t)
        u = torch.rand(B, nC, generator=gen)[g.cpu()].to(dev)
        lab = choose_labeled(valid, ei, nl, u)
        G = len(g)
        # lanes: (state g, labeled j, continuation k)
        xq = C.gather(1, lab[..., None].expand(-1, -1, C.shape[-1]))  # [G, nl, d]
        rows_l = g.repeat_interleave(nl * M)
        reps_l = (1 + torch.arange(M, device=dev)).repeat(G * nl)
        Xl = torch.cat([Xs.repeat_interleave(nl * M, 0), xq.repeat_interleave(M, 1).reshape(G * nl * M, 1, -1)], 1)
        fq = f_lanes(tasks, xq.reshape(G * nl, -1), g.repeat_interleave(nl))
        Fl = torch.cat([Fs.repeat_interleave(nl * M, 0), fq.repeat_interleave(M)[:, None]], 1)
        Xl, Fl = rollout(tasks, Xl, Fl, rows_l, reps_l, t + 1, T, pi0(engine, B))
        # one f* per task, lowered by every evaluated value of its rollouts (shared f*, standard S6)
        best = Fl.min(1).values.view(G, nl * M)
        fs = torch.minimum(torch.minimum(tasks.fstar[g], best.min(1).values), F[g].min(1).values)
        tasks.fstar[g] = fs
        out_R[g] = normalized_regret(Fl, fs.repeat_interleave(nl * M), fmed[rows_l]).view(G, nl, M)
        out_C[g], out_valid[g], out_lab[g] = C, valid, lab
    lab_valid = out_valid.gather(1, out_lab)
    return {
        "X": X,
        "F": F,
        "C": out_C,
        "valid": out_valid,
        "lab": out_lab,
        "lab_valid": lab_valid,
        "R": out_R,
        "fstar": tasks.fstar.clone(),
        "fmed": fmed,
        "branch_t": branch_t,
    }


@torch.no_grad()
def features_ei(engine: Engine, X: Tensor, Y: Tensor, C: Tensor) -> Tensor:
    """EI [L, C] of candidates (for the labeled-candidate stratification)."""
    assert engine.pfn is not None
    lg, _ = pfn_pass(engine, X, Y, C, with_z=False)
    return engine.pfn.bar_dist.ei(lg, Y.min(1).values[:, None])


def advantages(R: Tensor, lab_valid: Tensor) -> tuple[Tensor, Tensor]:
    """Paired advantage over x0 (index 0) and its standard error over continuations. R [B, nl, M]."""
    diff = R[:, :1] - R  # [B, nl, M], positive = candidate better
    M = R.shape[-1]
    adv = diff.mean(-1)
    se = diff.std(-1, unbiased=True) / math.sqrt(M) if M > 1 else torch.zeros_like(adv)
    return adv.masked_fill(~lab_valid, 0.0), se.masked_fill(~lab_valid, 0.0)


# --------------------------------------------------------------------------------------------------
# evaluation: pi_1 (conservative) against pi_0 on fresh tasks
# --------------------------------------------------------------------------------------------------


@torch.no_grad()
def run_improved(
    engine: Engine,
    spec: CandidateSpec,
    head: ValueHead,
    tasks: TaskSet,
    X0: Tensor,
    T: int,
    cs: list[float],
) -> dict[str, Tensor]:
    """Every task under pi_1 with each c in cs (c = inf is pi_0), replicate 0 (the evaluated trajectory;
    CRN across c). -> F [B, K, n_init + T], deviated [B, K, T], chosen source [B, K, T], A_hat of the chosen
    candidate [B, K, T]."""
    B, n_init, d = X0.shape
    K = len(cs)
    dev = X0.device
    rows = torch.arange(B, device=dev).repeat_interleave(K)
    reps = torch.zeros(B * K, dtype=torch.long, device=dev)
    c = torch.tensor(cs, device=dev).repeat(B)
    X = X0.repeat_interleave(K, 0)
    F = f_lanes(tasks, X.reshape(-1, d), rows.repeat_interleave(n_init)).view(B * K, n_init)
    src_ids = spec.source_ids().to(dev)
    devs, srcs, advs = [], [], []
    for t in range(T):
        C, valid = candidate_set(engine, spec, X, F, rows, reps, t, B)
        ft = features(engine, spec, X, F, C, valid, t, T)
        A, sig = head(ft["z"], ft["cand"], ft["glob"], valid)
        j, deviated = improved_pick(A, sig, valid, c)
        r = torch.arange(len(X), device=dev)
        x = C[r, j]
        X = torch.cat([X, x[:, None]], 1)
        F = torch.cat([F, f_lanes(tasks, x, rows)[:, None]], 1)
        devs.append(deviated)
        srcs.append(src_ids[j])
        advs.append(A[r, j])
    return {
        "F": F.view(B, K, -1),
        "deviated": torch.stack(devs, 1).view(B, K, T),
        "source": torch.stack(srcs, 1).view(B, K, T),
        "A_hat": torch.stack(advs, 1).view(B, K, T),
    }

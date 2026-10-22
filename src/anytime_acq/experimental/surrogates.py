"""Surrogates for BO off the PFN's prior: the PFN on rank-transformed y, and a batched GP.

Both see y only through its ranks within the lane's context (a copula transform), so the two surrogates
get the same information and neither depends on the benchmark's scale:
- `RankPFNEngine`: u = (rank + 1/2) / n in (0, 1). The PFN's training y is the prior's global ECDF, whose
  marginal over functions and x is Uniform(0, 1), so u matches the scale the PFN was trained on.
- `GPEngine`: z = Phi^-1(u), standard normal marginals (Gaussian copula).

The GP is BoTorch's default single-task GP, written out batched so that thousands of lanes fit at once:
constant mean, Matern-5/2 ARD kernel without outputscale, dimension-scaled LogNormal lengthscale prior
(loc = sqrt(2) + log(d) / 2, scale = sqrt(3); Hvarfner et al. 2024), LogNormal(-4, 1) noise prior,
MAP-II by Adam on every lane's own hyperparameters, refit from the prior's mode at every step.
Arms use the latent-f predictive N(mu, sigma^2), minimization, higher score = better, with the same names
and hyperparameters as `pfn_repro.acquisitions.standard` (`ei`, `pi<xi>`, `ucb<q>`, `mean`, `std`, `mes`).
"""

import math
from dataclasses import dataclass

import torch
from pfn_repro.acquisitions.standard import MES_SAMPLES, parse_arm

from ..bo.engine import Engine

SQRT2 = math.sqrt(2.0)
_N = torch.distributions.Normal(0.0, 1.0)


def rank_uniform(Y: torch.Tensor) -> torch.Tensor:
    """[L, n] -> (rank + 1/2) / n per row, ties at their average rank (float32)."""
    Y = Y.double()
    n = Y.shape[1]
    less = (Y[:, None, :] < Y[:, :, None]).sum(-1)
    equal = (Y[:, None, :] == Y[:, :, None]).sum(-1)
    return ((less + (equal - 1) / 2 + 0.5) / n).float()


@dataclass
class RankPFNEngine(Engine):
    """The PFN, with the context's y replaced by its ranks (u scale)."""

    def score(self, name, X, Y, C):
        return super().score(name, X, rank_uniform(Y), C)


def _matern52(A: torch.Tensor, B: torch.Tensor, ls: torch.Tensor) -> torch.Tensor:
    """A [M, n, d], B [M, m, d], ls [M, d] -> [M, n, m]."""
    r = torch.cdist(A / ls[:, None], B / ls[:, None]).clamp(min=1e-12)
    s = math.sqrt(5.0) * r
    return (1 + s + s * s / 3) * torch.exp(-s)


@dataclass
class GPFit:
    X: torch.Tensor  # [M, n, d]
    L: torch.Tensor  # [M, n, n] Cholesky of K + noise I
    alpha: torch.Tensor  # [M, n] (K + noise I)^-1 (y - c)
    ls: torch.Tensor  # [M, d]
    c: torch.Tensor  # [M]

    def predict(self, C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent f at C [M, P, d] -> mean, sd [M, P]."""
        Ks = _matern52(self.X, C, self.ls)  # [M, n, P]
        mu = self.c[:, None] + (Ks * self.alpha[..., None]).sum(1)
        v = torch.linalg.solve_triangular(self.L, Ks, upper=False)
        var = (1.0 - (v * v).sum(1)).clamp(min=1e-10)
        return mu, var.sqrt()


def fit_gp(X: torch.Tensor, z: torch.Tensor, iters: int = 100, lr: float = 0.1) -> GPFit:
    """MAP-II for every lane at once. X [M, n, d] in [0, 1], z [M, n] (standardized)."""
    M, n, d = X.shape
    X, z = X.double(), z.double()
    loc, scale = SQRT2 + 0.5 * math.log(d), math.sqrt(3.0)
    log_ls = torch.full((M, d), loc - scale**2, dtype=X.dtype, device=X.device)  # the prior's mode
    log_noise = torch.full((M,), -4.0 - 1.0, dtype=X.dtype, device=X.device)
    c = z.mean(1).clone()
    params = [log_ls.requires_grad_(), log_noise.requires_grad_(), c.requires_grad_()]
    opt = torch.optim.Adam(params, lr=lr)
    eye = torch.eye(n, dtype=X.dtype, device=X.device)

    def chol(log_ls: torch.Tensor, log_noise: torch.Tensor) -> torch.Tensor:
        noise = log_noise.exp().clamp(min=1e-6)[:, None, None]
        return torch.linalg.cholesky(_matern52(X, X, log_ls.exp()) + (noise + 1e-8) * eye)

    with torch.enable_grad():
        for _ in range(iters):
            opt.zero_grad()
            L = chol(log_ls, log_noise)
            r = (z - c[:, None])[..., None]
            a = torch.cholesky_solve(r, L)
            nll = 0.5 * (r * a).sum((1, 2)) + torch.diagonal(L, dim1=1, dim2=2).log().sum(1)
            prior = ((log_ls - loc) ** 2).sum(1) / (2 * scale**2) + (log_noise + 4.0) ** 2 / 2
            (nll + prior).sum().backward()
            opt.step()
    with torch.no_grad():
        L = chol(log_ls, log_noise)
        alpha = torch.cholesky_solve((z - c[:, None])[..., None], L)[..., 0]
        return GPFit(X, L, alpha, log_ls.exp(), c.detach())


def gp_mes(mu: torch.Tensor, sd: torch.Tensor, inc: torch.Tensor, n: int = MES_SAMPLES) -> torch.Tensor:
    """Max-value entropy search for minimization (Wang & Jegelka 2017) with f* quantiles of the pool's
    minimum (marginals independent, capped at the incumbent), as `fstar_quantiles` does for the PFN."""
    M = mu.shape[0]
    levels = (torch.arange(n, device=mu.device, dtype=mu.dtype) + 0.5) / n  # P(min <= s) = level
    lo = (mu - 6 * sd).min(1).values
    hi = inc.clone()
    a, b = lo[:, None].expand(M, n).clone(), hi[:, None].expand(M, n).clone()
    for _ in range(40):
        s = (a + b) / 2
        log_surv = torch.log1p(-_N.cdf((s[:, None, :] - mu[..., None]) / sd[..., None]).clamp(max=1 - 1e-9)).sum(1)
        below = (1 - log_surv.exp()) < levels
        a, b = torch.where(below, s, a), torch.where(below, b, s)
    fs = torch.minimum((a + b) / 2, inc[:, None])  # [M, n]
    g = ((mu[..., None] - fs[:, None, :]) / sd[..., None]).clamp(-8, 8)  # [M, P, n]
    cdf = _N.cdf(g).clamp(min=1e-12)
    pdf = _N.log_prob(g).exp()
    return (g * pdf / (2 * cdf) - cdf.log()).mean(-1)


def gp_scores(name: str, mu: torch.Tensor, sd: torch.Tensor, inc: torch.Tensor) -> torch.Tensor:
    """Higher = better; mu, sd [M, P], inc [M]."""
    base, hp = parse_arm(name)
    if base in ("ei", "pi"):
        imp = inc[:, None] - (hp or 0.0) - mu
        u = imp / sd
        if base == "pi":
            return _N.cdf(u)
        return sd * (u * _N.cdf(u) + _N.log_prob(u).exp())
    if base == "ucb":
        q = 0.95 if hp is None else hp
        return -(mu + _N.icdf(torch.tensor(1 - q, dtype=mu.dtype, device=mu.device)) * sd)
    if base == "mean":
        return -mu
    if base == "std":
        return sd
    if base == "mes":
        return gp_mes(mu, sd, inc)
    raise ValueError(f"arm {name!r} is not available for the GP")


@dataclass
class GPEngine(Engine):
    """Same candidates, CRN and null as the PFN engine; scores from a GP fitted to the lane's context."""

    fit_iters: int = 100

    def score(self, name, X, Y, C):
        z = _N.icdf(rank_uniform(Y).double())
        gp = fit_gp(X, z, iters=self.fit_iters)
        mu, sd = gp.predict(C.double())
        inc = z.min(1).values
        return gp_scores(name, mu, sd, inc).float(), mu.float(), None

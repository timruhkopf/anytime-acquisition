"""One BO step for many lanes at once: candidates, one PFN pass, every arm's argmax.

Candidate generator (standard S9): per (task, step, replicate) a fresh Cranley–Patterson-shifted
scrambled-Sobol set of `n_sobol` points, plus `n_local` Gaussian perturbations of the incumbent at each
scale in `local_sigmas`. All arms of a lane score the *same* candidates from the *same* PFN predictive
(one `context_kv` per lane, cached, then the candidates as test tokens). Points already evaluated are
excluded. Ties go to the lower posterior mean.

Arms (minimization; higher score = better; `pfn_repro.acquisitions.standard`):
  ei<ξ>, pi<ξ>, ucb<q>, mean, std, mes       from the PFN's bar distribution
  es                                         alpha-PFN mean (amortized Entropy Search, PES)
  <arm>@k                                    the arm restricted to random half-subset k of the candidates
                                             (k-specific per step and replicate): the equal-arms null
"""

from dataclasses import dataclass, field

import torch
from pfn_repro.acquisitions.standard import arm_scores, fstar_quantiles, parse_arm

from . import nuisance


@dataclass
class CandidateConfig:
    n_sobol: int = 1024
    n_local: int = 64
    local_sigmas: tuple[float, ...] = (0.01, 0.05, 0.2)


@dataclass
class Engine:
    pfn: torch.nn.Module
    alpha: torch.nn.Module | None = None
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    base_seed: int = 0
    ucb_quantile: float = 0.95
    chunk: int = 512

    def _sobol(self, d: int) -> torch.Tensor:
        if not hasattr(self, "_sobol_cache") or self._sobol_cache.shape[-1] != d:
            self._sobol_cache = torch.quasirandom.SobolEngine(d, scramble=True, seed=11).draw(self.candidates.n_sobol)
        return self._sobol_cache

    def candidate_set(
        self, X: torch.Tensor, Y: torch.Tensor, rows: torch.Tensor, reps: torch.Tensor, step: int, n_tasks: int
    ) -> torch.Tensor:
        """[L, P, d]: shifted Sobol (CRN per task, step, replicate) + incumbent perturbations."""
        L, _, d = X.shape
        dev = X.device
        shift = nuisance.stream(self.base_seed, "shift", step, reps, rows, (d,), n_tasks, False, dev)
        sob = (self._sobol(d).to(dev)[None] + shift[:, None]) % 1.0
        c = self.candidates
        n_loc = c.n_local * len(c.local_sigmas)
        z = nuisance.stream(self.base_seed, "perturb", step, reps, rows, (n_loc, d), n_tasks, True, dev)
        sig = torch.tensor(c.local_sigmas, device=dev).repeat_interleave(c.n_local)[None, :, None]
        inc = X[torch.arange(L, device=dev), Y.argmin(1)][:, None]
        return torch.cat([sob, (inc + sig * z).clamp(0, 1)], 1)

    @torch.no_grad()
    def propose(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        arms: list[str],
        rows: torch.Tensor,
        reps: torch.Tensor,
        step: int,
        n_tasks: int,
        return_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        """X [L, n, d], Y [L, n]; arms[l] is lane l's arm. -> {"x": [L, d], "logits": [L, nb] at x (optional)}."""
        L, _, d = X.shape
        dev = X.device
        cand = self.candidate_set(X, Y, rows, reps, step, n_tasks)
        P = cand.shape[1]
        out_x = torch.empty(L, d, device=dev)
        out_lg = torch.empty(L, self.pfn.bar_dist.num_bars, device=dev) if return_logits else None
        arm_idx = {a: [i for i, b in enumerate(arms) if b == a] for a in dict.fromkeys(arms)}
        for a, lanes_a in arm_idx.items():
            for s in range(0, len(lanes_a), self.chunk):
                li = torch.tensor(lanes_a[s : s + self.chunk], device=dev)
                Xa, Ya, Ca = X[li], Y[li], cand[li]
                nf = torch.full((len(li),), d, device=dev, dtype=torch.long)
                kv = self.pfn.context_kv(Xa, Ya, n_features=nf)
                lg = self.pfn.logits_from_context_kv(kv, Ca, n_features=nf).float()
                bd = self.pfn.bar_dist
                inc = Ya.min(1).values
                name = a.split("@")[0]
                base, _ = parse_arm(name)
                if base == "es":
                    assert self.alpha is not None, "arm 'es' needs the alpha-PFN"
                    kva = self.alpha.context_kv(Xa, Ya, n_features=nf)
                    score = self.alpha.bar_dist.mean(self.alpha.logits_from_context_kv(kva, Ca, n_features=nf).float())
                else:
                    fs = fstar_quantiles(bd, lg, inc) if base == "mes" else None
                    score = arm_scores(name, bd, lg, inc, self.ucb_quantile, mes_fs=fs)
                valid = torch.cdist(Ca, Xa).min(-1).values > 1e-7
                if "@" in a:  # equal-arms null: a k-specific random half of the candidates
                    k = int(a.split("@")[1])
                    u = nuisance.stream(
                        self.base_seed + 1_000 + k, "subset", step, reps[li], rows[li], (P,), n_tasks, False, dev
                    )
                    valid = valid & (u < 0.5)
                neg = torch.finfo(score.dtype).min
                sc = score.masked_fill(~valid, neg)
                best = sc.max(-1, keepdim=True).values
                tied = valid & (sc >= best - 1e-7 * best.abs().clamp(min=1e-12))
                j = bd.mean(lg).masked_fill(~tied, float("inf")).argmin(-1)
                r = torch.arange(len(li), device=dev)
                out_x[li] = Ca[r, j]
                if return_logits:
                    out_lg[li] = lg[r, j]
        return {"x": out_x, "logits": out_lg} if return_logits else {"x": out_x}

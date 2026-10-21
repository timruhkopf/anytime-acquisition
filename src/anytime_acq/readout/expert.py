"""PFN-VLA action expert: a small transformer that reads a PFN's per-layer context K/V and outputs a
distribution over the next query point (pi0-style joint attention; Black et al. 2024).

Expert tokens: one action token and one incumbent token (x_inc, y_inc). In expert layer l the tokens
attend jointly to themselves and to the context memory of source layer s(l), projected from the PFN's
(K_s, V_s) to the expert width:
- `pfn_all`:  s(l) = l, the frozen PFN, one expert layer per PFN layer;
- `pfn_last`: s(l) = L-1 for every l, the frozen PFN's last layer only;
- `d_only`:   s(l) = l of a PFN-shaped encoder with pfn_large's architecture, trained from scratch with
  the expert (the control: everything the frozen PFN has except its pretraining).
Optionally `n_probe` extra memory tokens: the frozen PFN's last hidden state at fixed Sobol probe points
(its prediction at query points), in every layer.

Head: a mixture of `n_mix` diagonal Gaussians in logit space u = logit(x), x in (0, 1)^d.
"""

import math

import torch
from pfn_repro.surrogates.pfn import PFN
from torch import nn

TRUNKS = ("pfn_all", "pfn_last", "d_only")
_X_EPS = 1e-4


class ExpertLayer(nn.Module):
    def __init__(self, de: int, d_mem: int, n_heads: int):
        super().__init__()
        self.h, self.dk = n_heads, de // n_heads
        self.ln = nn.LayerNorm(de)
        self.q, self.k, self.v, self.o = (nn.Linear(de, de) for _ in range(4))
        self.mem_k, self.mem_v = nn.Linear(d_mem, de), nn.Linear(d_mem, de)
        self.ff = nn.Sequential(nn.LayerNorm(de), nn.Linear(de, 4 * de), nn.GELU(), nn.Linear(4 * de, de))

    def forward(self, tok: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tok [B, T, de]; K, V [B, N, d_mem]; mask [B, N] (True = real memory token)."""
        B, T, de = tok.shape
        c = self.ln(tok)
        keys = torch.cat([self.mem_k(K), self.k(c)], 1)
        vals = torch.cat([self.mem_v(V), self.v(c)], 1)
        full = torch.cat([mask, mask.new_ones(B, T)], 1)

        def heads(z: torch.Tensor) -> torch.Tensor:
            return z.view(B, -1, self.h, self.dk).transpose(1, 2)

        s = heads(self.q(c)) @ heads(keys).transpose(-2, -1) / math.sqrt(self.dk)
        s = s.masked_fill(~full[:, None, None], torch.finfo(s.dtype).min)
        out = (s.softmax(-1) @ heads(vals)).transpose(1, 2).reshape(B, T, de)
        tok = tok + self.o(out)
        return tok + self.ff(tok)


class ActionExpert(nn.Module):
    def __init__(
        self,
        trunk: str,
        pfn: PFN,
        d_expert: int = 128,
        n_heads: int = 4,
        n_mix: int = 16,
        n_probe: int = 0,
        probe_seed: int = 5,
    ):
        super().__init__()
        if trunk not in TRUNKS:
            raise ValueError(f"trunk must be one of {TRUNKS}, got {trunk}")
        self.trunk, self.pfn, self.n_mix = trunk, pfn, n_mix  # pfn: frozen, not a submodule's parameters
        for p in pfn.parameters():
            p.requires_grad_(False)
        L, d_mem = len(pfn.blocks), pfn.train_embed.out_features
        self.max_x_dim = pfn.max_x_dim
        if trunk == "d_only":
            a = pfn.blocks[0]
            self.encoder = PFN(
                max_x_dim=pfn.max_x_dim,
                d_model=d_mem,
                n_heads=a.attn.h,
                n_layers=L,
                d_ff=a.ff[0].out_features,
                n_bins=pfn.bar_dist.num_bars,
            )
        self.n_probe = n_probe
        if n_probe:
            self.register_buffer(
                "probes", torch.quasirandom.SobolEngine(pfn.max_x_dim, scramble=True, seed=probe_seed).draw(n_probe)
            )
            self.probe_proj = nn.Linear(d_mem, d_mem)
        self.layers = nn.ModuleList(ExpertLayer(d_expert, d_mem, n_heads) for _ in range(L))
        self.action = nn.Parameter(torch.zeros(d_expert))
        self.inc_embed = nn.Linear(pfn.max_x_dim + 1, d_expert)
        self.budget_embed = nn.Linear(1, d_expert)
        self.out_ln = nn.LayerNorm(d_expert)
        self.head = nn.Linear(d_expert, n_mix * (1 + 2 * pfn.max_x_dim))

    def memory(
        self, X: torch.Tensor, Y: torch.Tensor, mask: torch.Tensor, nf: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Per expert layer (K, V, mask) of the source layer, plus probe tokens if any."""
        if self.trunk == "d_only":
            kv = self.encoder.context_kv_with_grad(X, Y, n_features=nf, train_key_padding_mask=mask)
        else:
            kv = self.pfn.context_kv(X, Y, n_features=nf, train_key_padding_mask=mask)
            if self.trunk == "pfn_last":
                kv = [kv[-1]] * len(kv)
        masks = [mask] * len(kv)
        if self.n_probe:
            h = self.probe_hidden(X, Y, mask, nf)
            pm = mask.new_ones(h.shape[:2])
            kv = [(torch.cat([K, h], 1), torch.cat([V, h], 1)) for K, V in kv]
            masks = [torch.cat([mask, pm], 1)] * len(kv)
        return [(K, V, m) for (K, V), m in zip(kv, masks, strict=True)]

    @torch.no_grad()
    def _pfn_probe_state(self, X, Y, mask, nf) -> torch.Tensor:
        B, _, d = X.shape
        P = self.probes[None, :, :d].expand(B, -1, -1)
        _, hidden = self.pfn(X, Y, P, n_features=nf, train_key_padding_mask=mask, return_hidden=True)
        return self.pfn.out_ln(hidden[-1][:, X.shape[1] :])  # the probes' final state, before the bar head

    def probe_hidden(self, X, Y, mask, nf) -> torch.Tensor:
        return self.probe_proj(self._pfn_probe_state(X, Y, mask, nf))

    def forward(
        self, X: torch.Tensor, Y: torch.Tensor, mask: torch.Tensor, budget_left: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """X [B, N, d] padded, Y [B, N], mask [B, N], budget_left [B] in [0, 1].
        -> mixture (log weights [B, M], means [B, M, d], log scales [B, M, d]) in logit space."""
        B, _, d = X.shape
        nf = torch.full((B,), d, device=X.device, dtype=torch.long)
        mem = self.memory(X, Y, mask, nf)
        Ym = Y.masked_fill(~mask, float("inf"))
        j = Ym.argmin(1)
        r = torch.arange(B, device=X.device)
        x_inc = torch.zeros(B, self.max_x_dim, device=X.device)
        x_inc[:, :d] = X[r, j]
        inc = self.inc_embed(torch.cat([x_inc, Y[r, j, None]], -1))
        tok = torch.stack([self.action.expand(B, -1) + self.budget_embed(budget_left[:, None]), inc], 1)
        for layer, (K, V, m) in zip(self.layers, mem, strict=True):
            tok = layer(tok, K, V, m)
        out = self.head(self.out_ln(tok[:, 0])).view(B, self.n_mix, 1 + 2 * self.max_x_dim)
        logw = out[..., 0].log_softmax(-1)
        mu = out[..., 1 : 1 + d]
        log_s = out[..., 1 + self.max_x_dim : 1 + self.max_x_dim + d].clamp(-7, 3)
        return logw, mu, log_s


def log_prob(mix: tuple[torch.Tensor, torch.Tensor, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """log q(x) for x [B, C, d] in (0, 1): mixture density in logit space plus the logit Jacobian."""
    logw, mu, log_s = mix
    x = x.clamp(_X_EPS, 1 - _X_EPS)
    u = torch.log(x) - torch.log1p(-x)
    z = (u[:, :, None] - mu[:, None]) / log_s.exp()[:, None]  # [B, C, M, d]
    comp = (-0.5 * z**2 - log_s[:, None] - 0.5 * math.log(2 * math.pi)).sum(-1)
    jac = -(torch.log(x) + torch.log1p(-x)).sum(-1)
    return torch.logsumexp(logw[:, None] + comp, -1) + jac


def mode(mix: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """The mean of the heaviest component, mapped to x: [B, d]."""
    logw, mu, _ = mix
    return torch.sigmoid(mu[torch.arange(len(mu)), logw.argmax(-1)])


def sample(
    mix: tuple[torch.Tensor, torch.Tensor, torch.Tensor], generator: torch.Generator | None = None
) -> torch.Tensor:
    logw, mu, log_s = mix
    r = torch.arange(len(mu))
    k = torch.multinomial(logw.exp(), 1, generator=generator)[:, 0]
    eps = torch.randn(mu.shape[0], mu.shape[-1], generator=generator, device=mu.device)
    return torch.sigmoid(mu[r, k] + log_s[r, k].exp() * eps)

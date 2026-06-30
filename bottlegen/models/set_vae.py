"""Set-latent VAE: surface → set of M latent vectors → σ̂.

Encoder: DiffusionNet produces per-vertex features; ``M`` learned query tokens
cross-attend (a permutation-invariant set pooling) to yield ``M`` latents, each
mapped to ``(μ, logσ²)``. Decoder (``SigmaDecoder``): per-degree query tokens
cross-attend to the latent set, then per-degree linears emit the packed σ̂.

DiffusionNet operators differ per mesh, so encoding loops over the batch; the
decoder is fully batched. Loss = Sobolev reconstruction on σ̂ + β·KL.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .diffusionnet import DiffusionNet
from .losses import sobolev_loss
from .pack import PackSpec

__all__ = ["SigmaDecoder", "SetVAE", "vae_loss"]


class SigmaDecoder(nn.Module):
    """Latent set ``[B, M, zdim]`` → packed σ̂ ``[B, dim]`` via per-degree query
    tokens cross-attending to the latents."""

    def __init__(self, zdim, M, L, d=256, heads=8, depth=2):
        super().__init__()
        self.spec = PackSpec(L)
        n_deg = len(self.spec.degs)
        self.lat_proj = nn.Linear(zdim, d)
        self.deg_q = nn.Parameter(torch.randn(n_deg, d) * 0.02)
        self.cross = nn.ModuleList([nn.MultiheadAttention(d, heads,
                                                          batch_first=True)
                                    for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            for _ in range(depth)])
        self.untok = nn.ModuleList([nn.Linear(d, dim) for dim in self.spec.dims])

    def forward(self, z):                                   # z: [B, M, zdim]
        B = z.shape[0]
        lat = self.lat_proj(z)                              # [B, M, d]
        h = self.deg_q.unsqueeze(0).expand(B, -1, -1)       # [B, n_deg, d]
        for attn, norm, mlp in zip(self.cross, self.norms, self.mlps):
            h = h + attn(norm(h), lat, lat, need_weights=False)[0]
            h = h + mlp(h)
        out = [self.untok[k](h[:, k]) for k in range(len(self.spec.degs))]
        return torch.cat(out, dim=-1)                       # [B, dim]


class SetVAE(nn.Module):
    def __init__(self, in_ch=6, C=128, M=32, zdim=16, L=16, n_blocks=4,
                 dec_d=256, heads=8, dec_depth=2):
        super().__init__()
        self.M, self.zdim, self.L = M, zdim, L
        self.dn = DiffusionNet(in_ch, C, n_blocks)
        self.q = nn.Parameter(torch.randn(M, C) * 0.02)
        self.enc_attn = nn.MultiheadAttention(C, heads, batch_first=True)
        self.enc_norm = nn.LayerNorm(C)
        self.to_mu = nn.Linear(C, zdim)
        self.to_lv = nn.Linear(C, zdim)
        self.dec = SigmaDecoder(zdim, M, L, d=dec_d, heads=heads, depth=dec_depth)

    def encode_one(self, ops):
        h = self.dn(ops).unsqueeze(0)                       # [1, V, C]
        q = self.q.unsqueeze(0)                             # [1, M, C]
        z = self.enc_attn(q, self.enc_norm(h), self.enc_norm(h),
                          need_weights=False)[0][0]         # [M, C]
        return self.to_mu(z), self.to_lv(z)                 # [M, zdim]

    def encode(self, ops_list):
        mus, lvs = zip(*[self.encode_one(o) for o in ops_list])
        return torch.stack(mus), torch.stack(lvs)           # [B, M, zdim]

    @staticmethod
    def reparam(mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    def decode(self, z):
        return self.dec(z)

    def forward(self, ops_list, sample=True):
        mu, lv = self.encode(ops_list)
        z = self.reparam(mu, lv) if sample else mu
        return self.decode(z), mu, lv

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def vae_loss(pred_sigma, target_sigma, mu, lv, weight, beta):
    """``(loss, recon, kl)`` — Sobolev σ̂ reconstruction + β·KL."""
    recon = sobolev_loss(pred_sigma, target_sigma, weight)
    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return recon + beta * kl, recon, kl

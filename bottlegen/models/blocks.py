"""DiT building blocks: sinusoidal timestep embedding + AdaLN-Zero block."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["timestep_embedding", "AdaLNBlock"]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    """Sinusoidal embedding of (normalized) timesteps ``t`` ∈ [0,1], ``[B]`` ->
    ``[B, dim]``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AdaLNBlock(nn.Module):
    """Transformer block with adaptive LayerNorm (AdaLN-Zero) conditioning.

    The conditioning MLP is zero-initialized so each block starts as the
    identity — the standard DiT-stable initialization.
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        sa, ba, ga, sm, bm, gm = self.ada(c).chunk(6, dim=-1)
        h = self.n1(x) * (1 + sa.unsqueeze(1)) + ba.unsqueeze(1)
        x = x + ga.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + sm.unsqueeze(1)) + bm.unsqueeze(1)
        x = x + gm.unsqueeze(1) * self.mlp(h)
        return x

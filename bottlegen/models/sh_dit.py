"""Degree-tokenized SWSH cold-diffusion DiT.

One token per degree ℓ ∈ [2, L]: a per-degree linear lifts the ``2(2ℓ+1)`` real
coefficients to width ``d``; ``depth`` AdaLN-Zero blocks attend across the
≤ ``L−1`` degree tokens conditioned on the (normalized) timestep; a per-degree
linear reads out the x0 prediction. Input and output use the shared packed
layout (:class:`bottlegen.models.pack.PackSpec`), so the network maps a packed
state ``[B, dim]`` to a packed x0 prediction ``[B, dim]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AdaLNBlock, timestep_embedding
from .pack import PackSpec

__all__ = ["SHDiT"]


class SHDiT(nn.Module):
    def __init__(self, L: int, d: int = 384, depth: int = 12, heads: int = 6,
                 t_embed: int = 256):
        super().__init__()
        self.L = L
        self.spec = PackSpec(L)
        self.degs = self.spec.degs
        self.t_embed = t_embed
        self.tok = nn.ModuleList([nn.Linear(dim, d) for dim in self.spec.dims])
        self.untok = nn.ModuleList([nn.Linear(d, dim) for dim in self.spec.dims])
        self.pos = nn.Parameter(torch.randn(len(self.degs), d) * 0.02)
        self.tproj = nn.Sequential(nn.Linear(t_embed, d), nn.SiLU(),
                                   nn.Linear(d, d))
        self.blocks = nn.ModuleList([AdaLNBlock(d, heads) for _ in range(depth)])

    def forward(self, x_packed: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``x_packed``: ``[B, dim]`` blurred state. ``t``: normalized ``[B]`` ∈
        [0,1]. Returns the packed x0 prediction ``[B, dim]``."""
        c = self.tproj(timestep_embedding(t, self.t_embed))
        parts = self.spec.split(x_packed)
        h = torch.stack([self.tok[k](parts[k]) for k in range(len(self.degs))],
                        dim=1) + self.pos
        for blk in self.blocks:
            h = blk(h, c)
        out = [self.untok[k](h[:, k]) for k in range(len(self.degs))]
        return self.spec.cat(out)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

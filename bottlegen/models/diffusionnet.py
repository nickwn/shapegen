"""DiffusionNet (Sharp et al., 2022): learned heat diffusion + spatial gradient
features on a surface, used as the SOTA encoder backbone.

Each block: (1) per-channel learned-time spectral heat diffusion using the LB
eigenbasis, (2) rotation-invariant tangent-gradient feature magnitudes, (3) an
MLP mixing the two. Operates on a single mesh ``x: [V, C]`` with its operators
``ops`` (a dict of torch tensors from :func:`ops_to_torch`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["DiffusionNetBlock", "DiffusionNet", "ops_to_torch"]


def ops_to_torch(ops_np, device=None, dtype=torch.float32, sparse_grad=False):
    """Convert NumPy mesh operators (:func:`bottlegen.geom.diffusion_ops.\
mesh_operators`) to torch tensors. Gradient operators are dense by default
    (fine for the meshes here); set ``sparse_grad`` for large meshes."""
    def t(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    def grad(g):
        if sparse_grad:
            g = g.tocoo()
            idx = torch.as_tensor(np.vstack([g.row, g.col]), dtype=torch.long,
                                  device=device)
            return torch.sparse_coo_tensor(idx, t(g.data), g.shape).coalesce()
        return t(g.toarray())

    return {
        "evals": t(ops_np["evals"]),
        "evecs": t(ops_np["evecs"]),
        "gradX": grad(ops_np["gradX"]),
        "gradY": grad(ops_np["gradY"]),
        "feats": t(np.concatenate([ops_np["points"], ops_np["normals"]], axis=1)),
    }


def _spmm(op, x):
    if op.is_sparse:
        return torch.sparse.mm(op, x)
    return op @ x


class DiffusionNetBlock(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.diffusion_time = nn.Parameter(torch.zeros(C))   # learned, per-channel
        self.mlp = nn.Sequential(nn.Linear(2 * C, C), nn.GELU(), nn.Linear(C, C))

    def forward(self, x, ops):
        evals, evecs = ops["evals"], ops["evecs"]            # [k], [V,k]
        t = torch.exp(self.diffusion_time)                   # [C] > 0
        coeff = evecs.t() @ x                                # [k, C]
        coeff = torch.exp(-evals[:, None] * t[None, :]) * coeff
        xs = evecs @ coeff                                   # [V, C] heat-diffused
        gx = _spmm(ops["gradX"], xs)
        gy = _spmm(ops["gradY"], xs)
        g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)            # [V, C] grad magnitude
        return x + self.mlp(torch.cat([xs, g], dim=-1))


class DiffusionNet(nn.Module):
    """Stack of DiffusionNet blocks with an input projection."""

    def __init__(self, in_ch: int, C: int, n_blocks: int = 4):
        super().__init__()
        self.inp = nn.Linear(in_ch, C)
        self.blocks = nn.ModuleList([DiffusionNetBlock(C) for _ in range(n_blocks)])

    def forward(self, ops):
        x = self.inp(ops["feats"])
        for blk in self.blocks:
            x = blk(x, ops)
        return x                                             # [V, C]

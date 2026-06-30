"""Real-packed state layout shared by the loader, the blur, and the DiT.

A state σ̂ (ℓ∈[2,L]) is packed into a single real vector of degree blocks; block
``ℓ`` is ``[Re(σ̂_{ℓ,−ℓ..ℓ}), Im(σ̂_{ℓ,−ℓ..ℓ})]`` of length ``2(2ℓ+1)``. This is
exactly :func:`bottlegen.flow.state.real_pack`, so the NumPy dataset, the torch
blur, and the network all speak the same layout.
"""

from __future__ import annotations

import numpy as np

from ..flow import schedule, state
from ..flow.schedule import _hermitian_symmetrize

__all__ = ["PackSpec", "kappa_packed", "blur_packed", "sobolev_packed",
           "perdeg_std_packed", "packed_to_sigma"]


def packed_to_sigma(vec: np.ndarray, L: int) -> np.ndarray:
    """Packed real vector -> complex σ̂ ``(L+1, 2L+1)``: unpack, Hermitian-
    symmetrize (real σ field), gauge-fix ℓ≤1. Ready for the Dirac decode."""
    C = state.real_unpack(np.asarray(vec, dtype=np.float64), L)
    C = _hermitian_symmetrize(C)
    return state.gauge_fix(C)


class PackSpec:
    """Degree-block boundaries for bandlimit ``L``."""

    def __init__(self, L: int):
        self.L = L
        self.degs = state.active_degrees(L)            # [2, .., L]
        self.dims = [2 * (2 * l + 1) for l in self.degs]
        self.bounds = np.concatenate([[0], np.cumsum(self.dims)]).astype(int)
        self.dim = int(self.bounds[-1])

    def split(self, x):
        """Split a packed tensor ``[B, dim]`` into per-degree tensors."""
        return [x[..., self.bounds[k]:self.bounds[k + 1]]
                for k in range(len(self.degs))]

    def cat(self, parts):
        import torch
        return torch.cat(list(parts), dim=-1)


def kappa_packed(L: int) -> np.ndarray:
    """Per-element blur rate ``κ_ℓ`` expanded over the packed layout (length
    ``PackSpec(L).dim``)."""
    sp = PackSpec(L)
    kap = schedule.kappa(L)
    return np.concatenate([np.full(sp.dims[k], kap[l], dtype=np.float64)
                           for k, l in enumerate(sp.degs)])


def perdeg_std_packed(perdeg_std: np.ndarray, L: int) -> np.ndarray:
    """Expand a per-degree RMS vector ``(L+1,)`` over the packed layout."""
    sp = PackSpec(L)
    return np.concatenate([np.full(sp.dims[k], perdeg_std[l])
                           for k, l in enumerate(sp.degs)])


def sobolev_packed(L: int, beta: float = 1.0,
                   perdeg_std: np.ndarray | None = None) -> np.ndarray:
    """Per-element loss weight: Sobolev ``(1+ℓ(ℓ+1))^β`` optionally whitened by
    ``1/perdeg_std_ℓ²`` so every degree contributes comparably."""
    sp = PackSpec(L)
    sob = state.sobolev_weights(L, beta)
    if perdeg_std is None:
        per = np.ones(L + 1)
    else:
        per = perdeg_std
    w = []
    for k, l in enumerate(sp.degs):
        w.append(np.full(sp.dims[k], sob[l] / max(per[l] ** 2, 1e-24)))
    return np.concatenate(w)


def blur_packed(x, t, kappa_packed_t):
    """Deterministic forward blur on packed states.

    ``x``: ``[B, dim]`` (real-packed). ``t``: physical times ``[B]``.
    ``kappa_packed_t``: the κ vector as a tensor on ``x``'s device. Returns
    ``x · exp(-κ t)`` (each degree block damped by ``e^{-κ_ℓ t}``).
    """
    import torch
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    t = t.reshape(-1, 1)
    return x * torch.exp(-kappa_packed_t.reshape(1, -1) * t)

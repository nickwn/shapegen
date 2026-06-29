"""Willmore energy and the conformal Willmore curvature flow.

``W = ∫H²dA ≥ 4π`` and ``W − 4π = ¼∫(κ₁−κ₂)² dA = ‖q‖²``. With ``q̂ = √κ_ℓ σ̂``
the spectral form is ``W − 4π = Σ_{ℓm} κ_ℓ |σ̂_{ℓm}|²``.

The linearized (spectral) Willmore L²-gradient flow is exactly the blur
``σ̂(t) = σ̂(0) e^{-κ_ℓ t}``: ``W(t)`` decreases monotonically to ``4π`` and the
Hopf differential ``q → 0`` (the surface rounds to a sphere). The full
Dirac-based flow for large deformations is the Model-B physics target and is
left for a later session; the spectral flow below is the faithful surrogate in
the near-sphere regime and is what the tests exercise.
"""

from __future__ import annotations

import numpy as np

from ..geom.curvature import mean_curvature
from ..geom.operators import vertex_areas
from .schedule import blur, kappa

__all__ = ["willmore_energy_spectral", "willmore_energy_mesh",
           "hopf_norm_sq_spectral", "willmore_flow"]


def hopf_norm_sq_spectral(sigma_hat: np.ndarray) -> float:
    """``‖q‖² = Σ_{ℓm} κ_ℓ |σ̂_{ℓm}|²`` (the Willmore excess ``W − 4π``)."""
    L = sigma_hat.shape[0] - 1
    kap = kappa(L)
    return float((kap[:, None] * np.abs(sigma_hat) ** 2).sum())


def willmore_energy_spectral(sigma_hat: np.ndarray) -> float:
    """``W = 4π + ‖q‖²`` from the state spectrum."""
    return 4.0 * np.pi + hopf_norm_sq_spectral(sigma_hat)


def willmore_energy_mesh(V, F) -> float:
    """Discrete ``W = ∫ H² dA`` (lumped per-vertex)."""
    H, _, _ = mean_curvature(V, F)
    A = vertex_areas(V, F)
    return float((H ** 2 * A).sum())


def willmore_flow(sigma_hat: np.ndarray, t_grid: np.ndarray):
    """Spectral Willmore flow trajectory over ``t_grid``.

    Returns ``(traj, W)`` with ``traj`` shape ``(T, L+1, 2L+1)`` the blurred
    states and ``W`` shape ``(T,)`` their Willmore energies (monotone ↓ to 4π).
    """
    L = sigma_hat.shape[0] - 1
    kap = kappa(L)
    traj = np.stack([blur(sigma_hat, float(t), kap) for t in t_grid], axis=0)
    W = np.array([willmore_energy_spectral(s) for s in traj])
    return traj, W

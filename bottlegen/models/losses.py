"""Training losses for the Simple model.

``sobolev_loss`` is a per-degree weighted MSE on the x0 (clean-state) prediction:
each packed element is weighted by ``(1+ℓ(ℓ+1))^β`` and optionally whitened by
``1/perdeg_std_ℓ²`` (see :func:`bottlegen.models.pack.sobolev_packed`).
"""

from __future__ import annotations

import numpy as np
import torch

from ..flow import state
from ..spectral import swsh
from .pack import PackSpec, sobolev_packed

__all__ = ["sobolev_loss", "make_weight",
           "gauss_residual", "codazzi_residual", "mean_curvature_multiplier",
           "codazzi_spectral_weight", "physics_residual_norms",
           "physics_weight_packed", "physics_loss"]


def make_weight(L: int, beta: float = 1.0, perdeg_std=None, device=None,
                dtype=torch.float32) -> torch.Tensor:
    """Packed loss-weight tensor on ``device``."""
    w = sobolev_packed(L, beta, perdeg_std)
    w = w / w.mean()                       # keep the loss O(1)
    return torch.as_tensor(w, dtype=dtype, device=device)


def sobolev_loss(pred: torch.Tensor, target: torch.Tensor,
                 weight: torch.Tensor) -> torch.Tensor:
    """Weighted MSE ``mean_b Σ_i w_i (pred − target)²_i / Σ_i w_i``."""
    se = (pred - target) ** 2
    return (se * weight.reshape(1, -1)).sum(-1).mean() / weight.sum()


# --------------------------------------------------------------------------- #
# Physics-prior residuals (structure equations)
# --------------------------------------------------------------------------- #
#
# Gauss   −Δ₀u = e^{2u}H² − e^{−2u}|q|²        (metric e^{2u}|dz|²)
# Codazzi  ∂_z̄ q = ½ e^{2u} ∂_z H
#
# In this project the *single* potential σ determines q = ð²σ, the mean-curvature
# response H, and the log conformal factor u, so the structure-equation residuals
# reduce to per-degree spectral operators on σ̂. The Codazzi residual in
# particular is a clean spectral weight that vanishes only as σ̂ → 0 (and grows
# ~ℓ³), so penalizing it acts as a geometrically-motivated high-ℓ regularizer —
# directly damping the excess high-ℓ content that destabilizes the decode.


def gauss_residual(u, H, q):
    """Gauss-equation residual field ``Δ₀u + e^{2u}H² − e^{−2u}|q|²`` (the plan's
    field-operator form; ``u``, ``H`` real spin-0 grids, ``q`` complex spin-2
    grid on ``gl_grid``). Zero for a consistent metric/curvature pair."""
    L = u.shape[0] - 1
    lap_u = np.real(swsh.synth(swsh.spin_laplacian(swsh.analy(u, 0, L), 0), 0, L))
    return lap_u + np.exp(2 * u) * H ** 2 - np.exp(-2 * u) * np.abs(q) ** 2


def codazzi_residual(u, H, q):
    """Codazzi-equation residual field ``ð̄q − ½ e^{2u} ðH`` (spin-1 grid)."""
    L = u.shape[0] - 1
    dbar_q = swsh.synth(swsh.edthbar(swsh.analy(q, 2, L), 2), 1, L)
    dz_H = swsh.synth(swsh.edth(swsh.analy(H.astype(complex), 0, L), 0), 1, L)
    return dbar_q - 0.5 * np.exp(2 * u) * dz_H


def mean_curvature_multiplier(L):
    """Per-degree mean-curvature response ``m_ℓ = (ℓ−1)(ℓ+2)/(ℓ(ℓ+1))`` (0 for
    ℓ≤1)."""
    l = np.arange(L + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (l - 1) * (l + 2) / (l * (l + 1))
    m[:2] = 0.0
    return m


def codazzi_spectral_weight(L):
    """Per-degree magnitude ``|c_C(ℓ)|`` of the linearized Codazzi residual
    ``ð̄q − ½ ðH`` with ``q̂=√κ_ℓ σ̂`` and ``Ĥ=m_ℓ σ̂``: grows ~ℓ³, zero for ℓ≤1."""
    l = np.arange(L + 1, dtype=float)
    kap = swsh.hopf_scaling(L)                     # √κ_ℓ
    m = mean_curvature_multiplier(L)
    with np.errstate(divide="ignore", invalid="ignore"):
        edthbar2 = np.sqrt((l + 2) * (l - 1))      # |ð̄ factor at spin 2|
        edth0 = np.sqrt(l * (l + 1))               # ð factor at spin 0
    c = -edthbar2 * kap - 0.5 * edth0 * m
    c[:2] = 0.0
    return np.abs(c)


def physics_residual_norms(sigma_hat):
    """``{R_C, R_G}``: area-weighted L2 norms of the (background-subtracted)
    Codazzi and Gauss residuals derived from ``σ̂``. Both vanish at ``σ̂ = 0``."""
    L = sigma_hat.shape[0] - 1
    # Codazzi: clean spectral residual c_C(ℓ)·σ̂
    cC = codazzi_spectral_weight(L)
    RC = float(np.sqrt(np.sum((cC[:, None] * np.abs(sigma_hat)) ** 2)))
    # Gauss: nonlinear, evaluated on the grid (background-subtracted)
    q = swsh.synth(swsh.q_from_sigma(sigma_hat), 2, L)
    Hc = swsh.empty_coeffs(L)
    Hc[0, 0] = np.sqrt(4 * np.pi)                  # H₀ = 1 (constant)
    m = mean_curvature_multiplier(L)
    Hc += m[:, None] * sigma_hat
    H = np.real(swsh.synth(Hc, 0, L))
    u = np.real(swsh.synth(_u_coeffs(sigma_hat), 0, L))
    from ..spectral.grids import gl_grid, solid_angle_weights
    g = gl_grid(L)
    wts = np.broadcast_to(solid_angle_weights(g), (g.ntheta, g.nphi))
    RGf = gauss_residual(u, H, q)
    RG0 = gauss_residual(np.zeros_like(u), np.ones_like(H), np.zeros_like(q))
    RG = float(np.sqrt(np.sum(wts * (RGf - RG0) ** 2)))
    return {"R_C": RC, "R_G": RG}


def _u_coeffs(sigma_hat):
    """Linearized log conformal factor coefficients ``û = (2/(ℓ(ℓ+1))) ρ̇`` with
    ``ρ̇ = (ℓ(ℓ+1)/2)·m_ℓ·σ̂`` -> ``û = m_ℓ σ̂`` (first-order, ℓ≥2)."""
    L = sigma_hat.shape[0] - 1
    return mean_curvature_multiplier(L)[:, None] * sigma_hat


# --- trainable spectral physics penalty (packed real σ̂) -------------------- #

def physics_weight_packed(L, eps=1e-6):
    """Codazzi spectral weight expanded over the packed layout and normalized so
    the largest degree is O(1) (keeps the penalty well-scaled)."""
    sp = PackSpec(L)
    cC = codazzi_spectral_weight(L)
    cC = cC / (cC.max() + eps)
    return np.concatenate([np.full(sp.dims[k], cC[l])
                           for k, l in enumerate(sp.degs)])


def physics_loss(pred_packed, phys_weight):
    """Differentiable Codazzi penalty ``mean_b Σ_i w_i² pred_i²`` on packed σ̂ —
    a geometrically-weighted high-ℓ regularizer (vanishes at σ̂=0)."""
    return ((pred_packed ** 2) * (phys_weight.reshape(1, -1) ** 2)).sum(-1).mean()

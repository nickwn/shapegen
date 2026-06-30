"""Field-aligned cross field and stripe parameterization from σ̂ / geometry.

The smoothest cross field (``mult=4`` RoSy) is the lowest eigenvector of the
connection Laplacian; away from umbilics it aligns with the principal-curvature
directions encoded by the Hopf differential ``q``. Singularities are forced on a
closed genus-0 surface — their indices sum to ``χ = 2`` (no defect-free quad
grid exists). A *stripe* field (``mult=1`` connection Laplacian carrying a
frequency 1-form) gives an evenly-spaced, alias-free line pattern aligned to the
cross field; two orthogonal stripe families give the quad texture.

Caveats from the plan are respected: we use the cross field's per-edge matching
directly (no comb into a directed field), and render stripes as solid bands, not
GL lines.
"""

from __future__ import annotations

import numpy as np

from ..geom.curvature import hopf_q, principal_curvatures
from ..geom.operators import vertex_areas, vertex_normals
from ..geom.transport import (connection_laplacian, frames_from_normals,
                              smallest_eig, transport_edges)

__all__ = ["cross_field", "cross_directions", "singularity_indices",
           "principal_alignment_error", "stripe_field", "stripe_phase"]


def _setup(V, F, normals=None):
    if normals is None:
        normals = vertex_normals(V, F)
    E1, E2, n = frames_from_normals(V, F, normals)
    edges, w, rho = transport_edges(V, F, E1, E2)
    return E1, E2, n, edges, w, rho


def cross_field(V, F, normals=None, mult=4, guide_weight=0.0):
    """``mult``-RoSy field. Returns ``(psi, frames, edge_data, lam)`` with
    ``psi`` the per-vertex complex field (``z = e^{i·mult·θ}``).

    With ``guide_weight = 0`` this is the *smoothest* field (smallest eigenvector
    of the connection Laplacian) — gauge-ambiguous up to a global rotation, so
    use it for singularity structure, not absolute orientation. With
    ``guide_weight > 0`` the field is pulled toward the principal-curvature
    directions (the Hopf differential), weighted by the anisotropy ``|κ₁−κ₂|``,
    by solving ``(L + γ·M·W) z = γ·M·W·t`` — smooth where umbilic, aligned where
    curved.
    """
    E1, E2, n, edges, w, rho = _setup(V, F, normals)
    M = vertex_areas(V, F)
    L, Mmat = connection_laplacian(len(V), edges, w, rho, mult, M)
    if guide_weight <= 0:
        lam, psi = smallest_eig(L, Mmat)
        return psi, (E1, E2), (edges, w, rho), lam
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    t, wgt = principal_guide(V, F, E1, E2, mult)
    gw = guide_weight * M * wgt
    A = (L + sp.diags(gw)).tocsc()
    psi = spla.spsolve(A, gw * t)
    psi = psi / np.maximum(np.abs(psi), 1e-30)        # unit field
    lam = float(np.real(np.conj(psi) @ (L @ psi)) / len(V))
    return psi, (E1, E2), (edges, w, rho), lam


def principal_guide(V, F, E1, E2, mult=4):
    """Target RoSy field ``t = e^{i·mult·θ_p}`` from the principal-curvature
    direction (``θ_p = −½ arg q``) and per-vertex anisotropy weight
    ``|κ₁−κ₂|`` (normalized). Umbilic vertices get ~zero weight."""
    q = hopf_q(V, F, E1, E2, vertex_normals(V, F))
    tp = -0.5 * np.angle(q)
    t = np.exp(1j * mult * tp)
    k1, k2 = principal_curvatures(V, F)
    wgt = np.abs(k1 - k2)
    wgt = wgt / (wgt.max() + 1e-30)
    return t, wgt


def cross_directions(psi, E1, E2, mult=4):
    """One representative 3D direction per vertex (the cross field is invariant
    to rotation by ``2π/mult``)."""
    theta = np.angle(psi) / mult
    return np.cos(theta)[:, None] * E1 + np.sin(theta)[:, None] * E2


def singularity_indices(V, F, psi, E1, E2, mult=4):
    """Per-face singularity indices of the RoSy field by geometric winding;
    ``Σ index = χ`` (Poincaré–Hopf). The cross direction is projected into each
    face plane and its rotation (mod ``2π/mult``) is accumulated around the
    triangle. Indices are multiples of ``1/mult`` and localized at singularities.
    """
    from ..geom.mesh import face_normals
    cd = cross_directions(psi, E1, E2, mult)
    fn = face_normals(V, F)
    period = 2.0 * np.pi / mult

    def wrap_period(a):
        return (a + period / 2) % period - period / 2

    idx = np.zeros(len(F))
    for f, (i, j, k) in enumerate(F):
        n = fn[f]
        e1 = E1[i] - (E1[i] @ n) * n
        e1 = e1 / max(np.linalg.norm(e1), 1e-30)
        e2 = np.cross(n, e1)
        ang = []
        for vid in (i, j, k):
            d = cd[vid] - (cd[vid] @ n) * n
            ang.append(np.arctan2(d @ e2, d @ e1))
        tot = (wrap_period(ang[1] - ang[0]) + wrap_period(ang[2] - ang[1])
               + wrap_period(ang[0] - ang[2]))
        idx[f] = tot / (2.0 * np.pi)
    return idx


def principal_alignment_error(V, F, psi, E1, E2, mult=4):
    """Mean misalignment (deg, mod 90°) between the cross field and the
    principal-curvature directions, weighted toward anisotropic (non-umbilic)
    vertices where the principal frame is well defined."""
    cd = cross_directions(psi, E1, E2, mult)
    q = hopf_q(V, F, E1, E2, vertex_normals(V, F))
    # principal direction angle in (E1,E2): θ_p = −½ arg(q) (see conformal.py)
    tp = -0.5 * np.angle(q)
    pd = np.cos(tp)[:, None] * E1 + np.sin(tp)[:, None] * E2
    # angle between lines (mod 90°) via the doubled-angle inner product
    ca = (cd * E1).sum(1)
    sa = (cd * E2).sum(1)
    cb = (pd * E1).sum(1)
    sb = (pd * E2).sum(1)
    ang_c = np.arctan2(sa, ca)
    ang_p = np.arctan2(sb, cb)
    d = np.abs(wrap90(ang_c - ang_p))
    k1, k2 = principal_curvatures(V, F)
    wgt = np.abs(k1 - k2)
    wgt = wgt / (wgt.sum() + 1e-30)
    return float(np.degrees((d * wgt).sum()))


def wrap90(a):
    """Wrap an angle difference to (−π/4, π/4] (lines are 90°-periodic for a
    cross field)."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def stripe_field(V, F, cross_dir, nu, normals=None):
    """Stripe (``mult=1``) field aligned to ``cross_dir`` with frequency ``nu``.

    Solves the connection Laplacian carrying a frequency 1-form
    ``ω_ij = ν · ⟨½(X_i+X_j), e_ij⟩``; the zero set of ``Im(ψ)`` is the stripe
    pattern. Returns ``(psi, lam)``.
    """
    E1, E2, n, edges, w, rho = _setup(V, F, normals)
    i, j = edges[:, 0], edges[:, 1]
    e = V[j] - V[i]
    omega = nu * (0.5 * (cross_dir[i] + cross_dir[j]) * e).sum(1)
    # transport for mult=1 plus the frequency phase
    import scipy.sparse as sp
    r = np.exp(1j * (rho + omega))
    I = np.concatenate([i, j])
    J = np.concatenate([j, i])
    vals = np.concatenate([-w * r, -w * np.conj(r)])
    L = sp.coo_matrix((vals, (I, J)), shape=(len(V), len(V))).tocsr()
    diag = np.zeros(len(V))
    np.add.at(diag, i, w)
    np.add.at(diag, j, w)
    L = (L + sp.diags(diag)).tocsr()
    lam, psi = smallest_eig(L, sp.diags(vertex_areas(V, F)))
    return psi, lam


def stripe_phase(psi):
    """Per-vertex stripe intensity ``cos(arg ψ)`` ∈ [−1, 1] for rendering solid
    bands (not GL lines)."""
    return np.cos(np.angle(psi))

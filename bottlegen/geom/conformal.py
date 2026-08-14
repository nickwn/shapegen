"""Spherical conformal parameterisation + Möbius gauge fix.

Produces the base sphere the ρ codec decodes from: a bijective embedding of a
genus-0 mesh onto S², on the mesh's own connectivity, with the ℓ=1 Möbius
freedom removed.

  1. :func:`spherical_conformal_map` — conformalized mean-curvature flow (cMCF,
     Kazhdan–Solomon–Ben-Chen): the *original* cotan Laplacian ``L0`` is held
     fixed while the mass ``M`` evolves, flowing any genus-0 mesh to a round
     sphere with low quasi-conformal distortion.
  2. :func:`mobius_normalize` — drive the conformal (area-weighted) barycenter of
     the sphere parameterisation to the origin via ball-automorphism inversions.
     This is the Möbius-registration centering: it kills the 3-dimensional
     Möbius "boost" freedom, leaving only rotation. (Scale is fixed by the unit
     sphere projection; **rotation is not fixed by anything** — see
     ``SIMPLIFICATION_PLAN.md`` §7.)

``scripts/willmore_flow.py`` offers an alternative to step 1 via the spin
transform, which on a bottle at subdiv 3 gives slightly *lower* distortion
(0.017 vs 0.019). Either works: because the encoder optimises decoded *shape*
error rather than the correspondence, the parameterisation's conformal quality is
largely compensated — the round trip lands well below the qc figure.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from .mesh import Mesh
from .operators import cotan_laplacian, vertex_areas, vertex_normals

__all__ = ["spherical_conformal_map", "mobius_normalize", "ball_automorphism"]


# --------------------------------------------------------------------------- #
# Conformalized mean-curvature flow (mesh -> sphere)
# --------------------------------------------------------------------------- #

def spherical_conformal_map(mesh: Mesh, dt: float = 5e-2, iters: int = 200,
                            tol: float = 1e-6) -> np.ndarray:
    """Flow ``mesh`` to the unit sphere by cMCF and return the sphere vertex
    positions (unit vectors, same connectivity).

    Backward Euler ``(M_k + dt·L0) f_{k+1} = M_k f_k`` with ``L0`` the *fixed*
    original cotan stiffness and ``M_k`` the evolving lumped mass; recentre and
    rescale each step. Converges (after projection) to a conformal sphere param.
    """
    V = mesh.vertices.astype(np.float64).copy()
    F = mesh.faces
    L0, _ = cotan_laplacian(V, F)
    L0 = L0.tocsc()
    for _ in range(iters):
        _, Mk = cotan_laplacian(V, F)
        A = (Mk + dt * L0).tocsc()
        solve = spla.factorized(A)
        rhs = Mk @ V
        Vn = np.stack([solve(rhs[:, c]) for c in range(3)], axis=1)
        Vn -= Vn.mean(0)
        radii = np.linalg.norm(Vn, axis=1)
        Vn /= max(radii.mean(), 1e-12)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------- #
# Ball automorphisms + Möbius normalization
# --------------------------------------------------------------------------- #

def ball_automorphism(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Conformal automorphism of the unit ball/sphere sending ``a`` to ``0``:

        φ_a(x) = [(1−|a|²)(x−a) − |x−a|² a] / [1 − 2⟨x,a⟩ + |x|²|a|²].

    For ``|x|=1`` the image lies on ``S²``; the result is re-normalized to guard
    against round-off.
    """
    a = np.asarray(a, float)
    a2 = float(a @ a)
    xa = x - a[None, :]
    xa2 = (xa * xa).sum(1, keepdims=True)
    num = (1.0 - a2) * xa - xa2 * a[None, :]
    den = (1.0 - 2.0 * (x @ a) + (x * x).sum(1) * a2)[:, None]
    y = num / den
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def mobius_normalize(mesh: Mesh, sph: np.ndarray, iters: int = 100,
                     tol: float = 1e-9) -> np.ndarray:
    """Drive the conformal barycenter of ``sph`` to the origin (kills ℓ=1).

    The barycenter is weighted by the *original* surface areas (the conformal
    density), so the fixed point is canonical under ball automorphisms of the
    parameterization.
    """
    w = vertex_areas(mesh.vertices, mesh.faces)
    w = w / w.sum()
    x = sph.copy()
    for _ in range(iters):
        c = (w[:, None] * x).sum(0)
        if np.linalg.norm(c) < tol:
            break
        # damp the step for stability when c is large
        step = c / max(1.0, 2.0 * np.linalg.norm(c))
        x = ball_automorphism(x, step)
    return x


# --------------------------------------------------------------------------- #
# Conformal factor and Hopf differential on the sphere
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# mesh -> σ̂
# --------------------------------------------------------------------------- #

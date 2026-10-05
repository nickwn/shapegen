"""Conformal-map utilities on the sphere: the Möbius gauge fix and the
edge-length conformal factor.

* :func:`mobius_normalize` drives the conformal (source-area-weighted)
  barycenter of a sphere parameterisation to the origin via ball automorphisms,
  killing the 3-parameter Möbius boost freedom and leaving only rotation.
* :func:`u_edge_lsq` fits the log conformal factor of a sphere → surface map
  from edge lengths.

The map itself is :func:`shapegen.geom.eigen_sphere.conformal_sphere` (it
replaced conformalized mean-curvature flow on 2026-09-27).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import Mesh
from .operators import vertex_areas

__all__ = [ "mobius_normalize", "ball_automorphism",
           "u_edge_lsq"]


# --------------------------------------------------------------------------- #
# Conformalized mean-curvature flow (mesh -> sphere)
# --------------------------------------------------------------------------- #

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


def u_edge_lsq(S: np.ndarray, B: np.ndarray, E: np.ndarray):
    """Log conformal factor ``u`` of the map ``S → B``, by edge least squares.

    Ported from ``rho_codec_pkg`` (rt/param/ufactor.py). For a conformal map every
    edge scales by the local factor, so

        (u_i + u_j)/2 = log( |B_j − B_i| / |S_j − S_i| )

    is one equation per edge — overdetermined, and solved by LSQR. Uniquely
    solvable because triangles are odd cycles, which kills the alternating
    null space a bipartite graph would have.

    This is far better than the lumped-area estimate ``e^u = √(a_B/a_S)``: it is
    exact to ~7e-14 on genuinely conformal input, where the area ratio carries the
    full O(h²) error of the mass lumping. ``ρ = H·e^u − 1`` inherits that error
    directly, so it is a first-order term for the direct encoder.

    Returns ``(u, residual_norm)``. The residual is a useful diagnostic on its own:
    it measures how far the correspondence is from conformal, so a large value
    means the map has shear the spin transform cannot express.
    """
    l0 = np.linalg.norm(S[E[:, 1]] - S[E[:, 0]], axis=1)
    l1 = np.linalg.norm(B[E[:, 1]] - B[E[:, 0]], axis=1)
    m, N = len(E), len(S)
    A = sp.coo_matrix((np.full(2 * m, 0.5),
                       (np.repeat(np.arange(m), 2), E.ravel())),
                      shape=(m, N)).tocsr()
    out = spla.lsqr(A, np.log(np.maximum(l1, 1e-300) / np.maximum(l0, 1e-300)),
                    atol=1e-14, btol=1e-14, iter_lim=8000)
    return out[0], float(out[3])



"""Conformal-map utilities on the sphere: the Möbius gauge fix.

:func:`mobius_normalize` drives the conformal (source-area-weighted) barycenter
of a sphere parameterisation to the origin via ball automorphisms, killing the
3-parameter Möbius boost freedom and leaving only rotation.

The map itself is :func:`shapegen.geom.eigen_sphere.conformal_sphere` (it
replaced conformalized mean-curvature flow on 2026-09-27).
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .operators import vertex_areas

__all__ = ["mobius_normalize", "ball_automorphism"]


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



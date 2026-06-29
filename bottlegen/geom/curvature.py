"""Curvature: mean ``H``, Gaussian ``K``, the shape operator, principal
directions, and the Hopf differential ``q`` (spin-2).

Conventions (``CONVENTIONS.md``): outward normal, convex-positive. A sphere of
radius ``R`` has ``H = +1/R``, ``K = +1/R²``, ``q = 0``. The Hopf differential is
the trace-free second fundamental form packed as a spin-2 complex,

    q = ½(II_11 − II_22) − i·II_12   in an orthonormal tangent frame (e1, e2),

so ``|q| = ½|κ₁ − κ₂|`` (umbilic ⇒ q = 0) and ``∫|q|² dA = ¼∫(κ₁−κ₂)² dA =
W − 4π``. The phase of ``q`` is frame-dependent (spin 2): rotating (e1,e2) by
angle ``α`` multiplies ``q`` by ``e^{-2iα}``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .operators import (angle_defect, cotan_laplacian, vertex_areas,
                        vertex_normals)

__all__ = [
    "mean_curvature", "gaussian_curvature", "tangent_frames",
    "second_fundamental_form", "principal_curvatures", "hopf_q",
    "vertex_one_ring",
]


def mean_curvature(V, F, L=None, M=None):
    """Pointwise mean curvature ``H`` (convex-positive), the mean-curvature
    vector, and outward normals, from ``Hvec = ½ M⁻¹ L V``.

    Returns ``(H, Hvec, normals)``.
    """
    if L is None or M is None:
        L, M = cotan_laplacian(V, F)
    Minv = 1.0 / M.diagonal()
    Hvec = 0.5 * (Minv[:, None] * (L @ V))
    n = vertex_normals(V, F)
    H = np.linalg.norm(Hvec, axis=1) * np.sign((Hvec * n).sum(1))
    return H, Hvec, n


def gaussian_curvature(V, F, M=None):
    """Pointwise Gaussian curvature ``K = (2π − Σθ) / A_i`` (angle defect over
    lumped area). ``Σ K_i A_i = 2πχ`` holds exactly (Gauss–Bonnet)."""
    if M is None:
        A = vertex_areas(V, F)
    else:
        A = M.diagonal()
    return angle_defect(V, F) / np.maximum(A, 1e-30)


def vertex_one_ring(V, F):
    """Adjacency as a list of neighbor-index arrays, one per vertex."""
    n = len(V)
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2],
                           F[:, 1], F[:, 2], F[:, 0]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0],
                           F[:, 0], F[:, 1], F[:, 2]])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    A = A.tolil()
    return [np.asarray(A.rows[i], dtype=np.int64) for i in range(n)]


def tangent_frames(V, F, normals=None):
    """Deterministic orthonormal tangent frames ``(e1, e2)`` per vertex with
    ``e1 × e2 = n``. The reference direction is fixed (projection of the global
    x- or z-axis), so the frame — and hence the spin-2 phase of ``q`` — is
    reproducible. Returns ``(e1, e2, normals)``."""
    if normals is None:
        normals = vertex_normals(V, F)
    n = normals
    # pick a reference axis least aligned with n to avoid degeneracy
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(V), 1))
    bad = np.abs(n[:, 0]) > 0.9
    ref[bad] = np.array([0.0, 0.0, 1.0])
    e1 = ref - (ref * n).sum(1, keepdims=True) * n
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-30)
    e2 = np.cross(n, e1)
    return e1, e2, n


def second_fundamental_form(V, F, e1=None, e2=None, normals=None, rings=None):
    """Per-vertex second fundamental form ``II = [[e, f], [f, g]]`` in the
    tangent frame ``(e1, e2)``, by least-squares fitting normal curvatures of
    1-ring edges: ``κ_n(t) = 2 n·(V_j − V_i)/|V_j − V_i|²`` and
    ``κ_n = e c² + 2 f c s + g s²`` for unit tangent ``(c, s)``.

    Returns ``(e, f, g)`` each shape ``(n,)``.
    """
    if normals is None:
        e1, e2, normals = tangent_frames(V, F)
    elif e1 is None:
        e1, e2, normals = tangent_frames(V, F, normals)
    if rings is None:
        rings = vertex_one_ring(V, F)
    n = len(V)
    E = np.zeros(n)
    Fc = np.zeros(n)
    G = np.zeros(n)
    for i in range(n):
        nb = rings[i]
        if len(nb) < 3:
            continue
        d = V[nb] - V[i]                       # (k,3)
        d2 = (d * d).sum(1)
        d2 = np.maximum(d2, 1e-30)
        # κ_n = -2 n·(V_j - V_i)/|V_j - V_i|²: a convex surface curves away from
        # the outward normal, so this is +1/R on a sphere (convex-positive).
        kn = -2.0 * (d @ normals[i]) / d2
        u = d @ e1[i]
        v = d @ e2[i]
        t = np.sqrt(np.maximum(u * u + v * v, 1e-30))
        c, s = u / t, v / t
        A = np.stack([c * c, 2.0 * c * s, s * s], axis=1)   # (k,3)
        sol, *_ = np.linalg.lstsq(A, kn, rcond=None)
        E[i], Fc[i], G[i] = sol
    return E, Fc, G


def principal_curvatures(V, F, e1=None, e2=None, normals=None, rings=None):
    """Principal curvatures ``(k1, k2)`` (k1 ≥ k2) from the fitted shape
    operator. Mean ``H = (k1+k2)/2``, Gaussian ``K = k1 k2``."""
    e, f, g = second_fundamental_form(V, F, e1, e2, normals, rings)
    tr = e + g
    det = e * g - f * f
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    k1 = tr / 2.0 + disc
    k2 = tr / 2.0 - disc
    return k1, k2


def hopf_q(V, F, e1=None, e2=None, normals=None, rings=None):
    """Hopf differential ``q = ½(II_11 − II_22) − i·II_12`` (spin-2 complex,
    one value per vertex, in the frame ``(e1, e2)``). ``|q| = ½|κ₁ − κ₂|``."""
    e, f, g = second_fundamental_form(V, F, e1, e2, normals, rings)
    return 0.5 * (e - g) - 1j * f

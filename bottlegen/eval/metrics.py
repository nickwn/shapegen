"""Mesh-quality metrics.

Pure NumPy/SciPy. Trimmed to what the codec and the scripts actually call — the
generative-eval metrics (novelty/coverage, distribution overlap, per-degree power
of a σ̂ that no longer exists) went with the models they scored.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..geom.curvature import mean_curvature
from ..geom.mesh import Mesh, face_normals
from ..geom.operators import vertex_areas

__all__ = ["willmore_energy", "flip_fraction", "genus", "chamfer",
           "triangle_quality"]


def willmore_energy(V, F) -> float:
    """Discrete ``W = ∫ H² dA`` (≥ 4π for a smooth genus-0 surface)."""
    H, _, _ = mean_curvature(V, F)
    return float((H ** 2 * vertex_areas(V, F)).sum())


def flip_fraction(V, F) -> float:
    """Fraction of faces whose outward orientation disagrees with the radial
    direction from the centroid — a cheap degeneracy / self-fold proxy."""
    fn = face_normals(V, F)
    fc = V[F].mean(1) - V.mean(0)
    return float(((fn * fc).sum(1) < 0).mean())


def genus(V, F) -> int:
    return Mesh(V, F).genus


def chamfer(A, B, scale_align=False) -> float:
    """Symmetric Chamfer (mean nearest-neighbor distance). ``scale_align``
    removes the global scale gauge (the decode fixes scale only up to |λ|²)."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    if scale_align:
        A = A * np.sqrt((B ** 2).sum() / max((A ** 2).sum(), 1e-30))
    da, _ = cKDTree(B).query(A)
    db, _ = cKDTree(A).query(B)
    return 0.5 * (da.mean() + db.mean())


def triangle_quality(V, F) -> dict:
    """Triangle-angle quality: min angle (deg) and its distribution."""
    ang = np.zeros((len(F), 3))
    for k in range(3):
        i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u = V[j] - V[i]
        w = V[l] - V[i]
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-30)
        w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-30)
        ang[:, k] = np.degrees(np.arccos(np.clip((u * w).sum(1), -1, 1)))
    mn = ang.min(1)
    return {"min_angle": float(mn.min()), "mean_min_angle": float(mn.mean()),
            "frac_below_20deg": float((mn < 20).mean())}

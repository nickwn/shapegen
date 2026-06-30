"""Mesh-quality and distribution metrics for generated bottles.

Pure NumPy/SciPy (no torch). These are the primitives the decode trust region
and the eval report both use.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..geom.curvature import mean_curvature
from ..geom.mesh import Mesh, face_normals, triangle_areas
from ..geom.operators import vertex_areas

__all__ = [
    "willmore_energy", "flip_fraction", "genus", "nonstar_fraction",
    "chamfer", "per_degree_power", "triangle_quality", "valence_stats",
    "is_valid_mesh", "novelty_coverage", "distribution_overlap", "summarize",
]


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


def is_valid_mesh(V, F, max_flip=0.02, w_cap=None) -> bool:
    """Non-degenerate genus-0 check used by the decode trust region."""
    if not np.isfinite(V).all():
        return False
    if flip_fraction(V, F) > max_flip:
        return False
    if w_cap is not None and willmore_energy(V, F) > w_cap:
        return False
    return True


def nonstar_fraction(V, F, n_dirs=1500, seed=0) -> float:
    """Fraction of rays from the centroid that cross the surface more than once
    (vectorized Möller–Trumbore). ``0`` ⇒ star-shaped; ``>0`` ⇒ a radial height
    field cannot represent the shape."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((n_dirs, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = V.mean(0)
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    counts = np.zeros(n_dirs, dtype=int)
    s = o - p0
    q = np.cross(s, e1)
    for r in range(n_dirs):
        dir_ = d[r]
        h = np.cross(dir_, e2)
        a = (e1 * h).sum(1)
        ok = np.abs(a) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
        u = inv * (s * h).sum(1)
        v = inv * (dir_ @ q.T)
        t = inv * (e2 * q).sum(1)
        hit = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        counts[r] = int(hit.sum())
    return float((counts > 1).mean())


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


def per_degree_power(sigma_hat) -> np.ndarray:
    """Per-degree power ``Σ_m |σ̂_{ℓm}|²`` (ℓ ∈ [0, L])."""
    L = sigma_hat.shape[0] - 1
    return np.array([np.sum(np.abs(sigma_hat[l, 0:2 * l + 1]) ** 2)
                     for l in range(L + 1)])


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


def valence_stats(V, F) -> dict:
    """Vertex-valence distribution (6 is ideal interior valence)."""
    val = np.zeros(len(V), dtype=int)
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.unique(np.sort(edges, 1), axis=0)
    np.add.at(val, edges[:, 0], 1)
    np.add.at(val, edges[:, 1], 1)
    return {"mean": float(val.mean()), "min": int(val.min()),
            "max": int(val.max()),
            "frac_irregular": float((val != 6).mean())}


def novelty_coverage(gen, train):
    """``(novelty, coverage)`` between generated and training feature vectors
    (e.g. real-packed σ̂). Novelty = mean nearest-train distance of generated
    samples (large ⇒ not memorized); coverage = fraction of train items that are
    some generated sample's nearest neighbor (large ⇒ diverse, mode-covering)."""
    gen = np.atleast_2d(gen)
    train = np.atleast_2d(train)
    D = np.linalg.norm(gen[:, None, :] - train[None, :, :], axis=2)
    novelty = float(D.min(1).mean())
    coverage = len(np.unique(D.argmin(1))) / len(train)
    return novelty, coverage


def distribution_overlap(a, b, bins=12):
    """Histogram-intersection overlap ∈ [0,1] of two 1-D samples (1 = identical
    distributions). Used for Willmore / non-star distribution match."""
    a, b = np.asarray(a), np.asarray(b)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha = np.histogram(a, edges, density=False)[0] / len(a)
    hb = np.histogram(b, edges, density=False)[0] / len(b)
    return float(np.minimum(ha, hb).sum())


def summarize(values) -> dict:
    """``mean/median/std/min/max`` of a list of scalars."""
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}

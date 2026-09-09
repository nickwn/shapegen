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
           "nose_tip", "chamfer_region",
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


def chamfer(A, B, scale_align=False, center=True) -> float:
    """Symmetric Chamfer (mean nearest-neighbor distance). ``scale_align``
    removes the global scale gauge (the decode fixes scale only up to |λ|²).

    Unlike a Procrustes error this is **topology-free**, so it is the right metric
    for comparing meshes with different vertex counts — a decode on an icosphere
    against one on a cMCF sphere, say.

    **Use it whenever a pipeline change alters the mesh.** `codec.shape_error` is
    per-vertex and can only compare meshes that share topology, which forces you
    to score against the *resampled* target — and if the change under test also
    moved that target, the comparison is confounded. That happened three times in
    one session, reporting 25-32% improvements that were entirely a softer
    reference. The rule: fix the reference to the original input and compare
    everything to it with this function.

    ``scale_align`` normalises by RMS radius. It used to use the raw sums, which
    is **vertex-count dependent** and therefore wrong for exactly the case this
    function is otherwise good at: two unit spheres with 10242 and 40962 vertices
    scored 0.99999 (the smaller was scaled by 2.0) instead of 0.00708. With equal
    counts the two agree, which is why it went unnoticed.

    Note ``A.mean(0)`` is the vertex mean, not the area-weighted centroid, so on a
    badly non-uniform mesh the centring is itself off — the cMCF sphere of a head
    has 1e6x area spread and a vertex mean 0.35 from its centre. **Pass
    ``center=False`` and centre the clouds yourself when the two have different
    vertex distributions.** A FaceScape head carries 40% of its vertices in the
    face while its icosphere resample is uniform, so the two vertex means differ
    enough that this default shifted them apart and reported 0.155 where the
    area-weighted centring gives 0.0152 — a 10x error that looks like a plausible
    number, not like a bug.
    """
    if center:
        A = A - A.mean(0)
        B = B - B.mean(0)
    if scale_align:
        A = A * np.sqrt((B ** 2).mean() / max((A ** 2).mean(), 1e-30))
    da, _ = cKDTree(B).query(A)
    db, _ = cKDTree(A).query(B)
    return 0.5 * (da.mean() + db.mean())


def nose_tip(V, r: float = 0.8):
    """The nose tip of a head, as the extreme point along the depth axis that has
    vertices *around* it.

    Both ends of that axis are extreme points of a head, and picking the wrong one
    silently scores the occiput: with FaceScape's registered topology the face
    end had 11156 vertices within ``r`` and the cranium end 670, so a
    face-region metric built on the wrong end covered 1% of the mesh and reported
    a 17% improvement that was measuring the back of the skull.

    Choosing by local vertex count is a property of the sampling rather than a
    hand-picked index, so it survives a change of subject. ``V`` is expected in a
    frame where z is depth and the scale is RMS-normalised.
    """
    V = np.asarray(V, float)
    cand = [V[np.argmax(V[:, 2])], V[np.argmin(V[:, 2])]]
    n = [int((np.linalg.norm(V - c, axis=1) < r).sum()) for c in cand]
    return cand[int(np.argmax(n))]


def chamfer_region(A, B, centre, radius: float) -> float:
    """Chamfer restricted to points within ``radius`` of ``centre``.

    **A whole-head mean hides the face.** Most of a FaceScape head is bald
    cranium and neck, which are nearly featureless and nearly identical between
    subjects, so a mean over all of it flatters any model and — worse — moves for
    reasons that have nothing to do with the face. Dropping ``remesh`` from 2.5%
    to 0.8% improved the whole-head decode 8% and the face decode 0%: the entire
    gain was cranium faceting.

    Nearest neighbours are searched over the **full** other cloud and only the
    average is restricted, because masking both clouds first would invent error
    at the region boundary where the true match lies just outside it.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    ma = np.linalg.norm(A - centre, axis=1) < radius
    mb = np.linalg.norm(B - centre, axis=1) < radius
    if not ma.any() or not mb.any():
        raise ValueError("region is empty on at least one cloud; check `centre` "
                         "is in the same frame and `radius` is not too small")
    da, _ = cKDTree(B).query(A[ma])
    db, _ = cKDTree(A).query(B[mb])
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

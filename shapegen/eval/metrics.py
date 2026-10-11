"""Mesh-quality metrics.

Pure NumPy/SciPy. For scoring a reconstruction of a scanned head use
:func:`chamfer` against the ORIGINAL head, plus :func:`chamfer_region` about
:func:`nose_tip` for the face.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..geom.mesh import Mesh, face_normals

__all__ = ["flip_fraction", "genus", "chamfer", "nose_tip", "chamfer_region"]


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

    Unlike a Procrustes error this is **topology-free**, so it compares meshes
    with different vertex counts.

    **Use it whenever a pipeline change alters the mesh.** ``codec.shape_error``
    is per-vertex and needs shared topology, which forces scoring against the
    *resampled* target -- and if the change under test also moved that target,
    the comparison is confounded (this has reported 25-32% gains that were
    entirely a softer reference). Fix the reference to the original input and
    compare everything to it with this function.

    ``scale_align`` normalises by RMS radius, which is vertex-count independent.

    **Centring trap:** ``center=True`` subtracts the *vertex* mean, not the
    area-weighted centroid. A FaceScape head carries 40% of its vertices in the
    face while its icosphere resample is uniform, so the two vertex means differ
    and this default reported 0.155 where area-weighted centring gives 0.0152 --
    a 10x error that looks plausible. **Pass ``center=False`` and centre the
    clouds yourself when their vertex distributions differ.**
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

    **Trap:** both ends of the depth axis are extreme points of a head, and
    picking the wrong one silently scores the occiput (a face metric on the wrong
    end covered 1% of the mesh and reported a 17% gain on the back of the skull).
    The face end is the **dense** end on FaceScape's registered topology -- 11156
    vertices within ``r`` against 670 at the cranium end -- so the candidate with
    more vertices around it wins. That is a property of the sampling, not a
    hand-picked index, so it survives a change of subject; pick the tip on the
    original registered head, whose sampling this relies on.

    ``V`` is expected in a frame where z is depth and the scale is RMS-normalised.
    """
    V = np.asarray(V, float)
    cand = [V[np.argmax(V[:, 2])], V[np.argmin(V[:, 2])]]
    n = [int((np.linalg.norm(V - c, axis=1) < r).sum()) for c in cand]
    return cand[int(np.argmax(n))]


def chamfer_region(A, B, centre, radius: float) -> float:
    """Chamfer restricted to points within ``radius`` of ``centre``.

    **A whole-head mean hides the face.** Most of a FaceScape head is bald
    cranium and neck, so a mean over all of it flatters any model and moves for
    reasons unrelated to the face (a 2.5% -> 0.8% remesh improved the whole head
    8% and the face 0%). Report this alongside the whole-head number; the face
    numbers in this repo use ``centre=nose_tip(V)``, ``radius=0.7`` on an
    RMS-normalised head (~40% of its vertices).

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


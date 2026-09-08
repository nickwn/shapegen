"""Resample a sphere-parameterised mesh onto a different sphere discretisation.

A cMCF sphere inherits its source mesh's triangulation, which for a registered
face set means **2.1e6x triangle-area spread** — areas down to 3.4e-9. Two stages
of the codec divide by those areas: the Dirac assembly's ``-e_p e_q / (4 A_f)``,
which reaches ``1/(4A) = 7.4e7``, and :func:`~bottlegen.geom.encode.sh_project`'s
Galerkin mass. Moving to a uniform icosphere (1.3x spread) is worth about **27%**
of reconstruction error and flattens the error-vs-L curve, so the optimum sits at
L=16-24 rather than 12.

What is *not* claimed: that the cMCF sphere is unusable. Decoding on it
reconstructs at 0.047-0.060 across L=8..32, which is serviceable. This is a
conditioning improvement, not a rescue.

The parameterisation is unchanged by resampling. The cMCF sphere is used once, to
answer "which point of the target does this direction correspond to", and then
discarded; the same conformal map is simply sampled at uniform positions instead
of at the source mesh's own vertices.
"""

from __future__ import annotations

import numpy as np

__all__ = ["locate_on_sphere", "resample_to"]


def locate_on_sphere(S: np.ndarray, F: np.ndarray, dirs: np.ndarray, k: int = 8):
    """For each unit direction, the containing triangle of ``(S, F)`` and its
    barycentric weights. Returns ``(idx (m,), bary (m,3), n_fallback)``.

    A hybrid, because neither half is sufficient alone. libigl's
    ``point_mesh_squared_distance`` supplies the containing face from an AABB tree
    in C++ — robust, and the part that was hand-rolled and buggy. But it returns
    the **nearest** point on the mesh, and what is wanted is the **radial**
    intersection: where the ray from the origin along ``d`` crosses the triangle.
    Those coincide on a well-shaped sphere and diverge on a distorted one, so the
    weights are solved for here rather than derived from libigl's closest point.
    Taking igl's point directly produced 88 sliver triangles on a deliberately
    degenerate source.

    This replaced a hand-rolled search that shipped two bugs, both worth recording
    because they are the failure modes any replacement must avoid:

    * searching from the nearest triangle **centroid** misses points inside large
      triangles — on a mesh with 1e6x area spread a centroid can be far from its
      own interior — which left 39 of 10242 directions unresolved;
    * leaving those unresolved as index ``-1`` with zero weights silently indexes
      the last face and places the vertex at the **origin**. That produced 44
      zero-area triangles, rho values of 1e91, and finally an ARPACK "starting
      vector is zero" hundreds of lines away with nothing in between to explain it.

    ``n_fallback`` counts faces whose plane passes through the origin, where the
    radial weights are undefined and the centroid is substituted. On a sphere mesh
    that should be zero; a nonzero count means the "sphere" is not star-shaped
    about the origin and the whole resampling is suspect.

    **Limitation.** Radial projection is ill-conditioned when a face's plane
    passes near the origin, and a real cMCF sphere does contain such faces —
    measured, the closest face plane on a FaceScape head's sphere sits 0.128 from
    the origin against an ideal 1.0. It is benign there only because those faces
    are small and absorb few queries; the production path returns 0 fallbacks and
    0 sliver triangles at icosphere subdiv 5 and 6. A source with *few, huge*
    near-origin faces does produce collapsed output triangles, so check
    `triangle_areas` on the result if the source is unusual.

    ``k`` is unused and kept for signature compatibility.
    """
    import igl  # noqa: PLC0415

    S = np.asarray(S, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    dirs = np.asarray(dirs, dtype=float)
    _, idx, C = igl.point_mesh_squared_distance(dirs, S, F)
    idx = np.asarray(idx, dtype=np.int64).ravel()

    # Radial weights: solve [a b c] w = d, so that w0*a + w1*b + w2*c is parallel
    # to d. Normalising w to sum to 1 then lands the point where the ray crosses
    # the triangle's plane -- the projection of the sphere onto that face.
    M = np.stack([S[F[idx, 0]], S[F[idx, 1]], S[F[idx, 2]]], axis=2)
    try:
        w = np.linalg.solve(M, dirs[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:                # a singular face somewhere
        w = np.stack([np.linalg.lstsq(M[i], dirs[i], rcond=None)[0]
                      for i in range(len(dirs))])
    s = w.sum(1, keepdims=True)
    bad = np.abs(s).ravel() < 1e-12              # face plane through the origin
    w = np.where(bad[:, None], 1.0 / 3.0, w / np.where(np.abs(s) < 1e-12, 1.0, s))
    return idx, np.clip(w, 0.0, None) / np.maximum(
        np.clip(w, 0.0, None).sum(1, keepdims=True), 1e-30), int(bad.sum())


def resample_to(S, F, X, dirs):
    """Resample meshes ``X`` ``(N, n, 3)``, parameterised by ``(S, F)``, onto
    ``dirs``. Returns ``(X_out (N, m, 3), idx, bary, n_fallback)``.

    ``X`` and ``S`` share connectivity ``F`` — that is what makes the barycentric
    weights transferable: a weight found on the sphere applies unchanged to the
    target, because vertex *i* means the same thing in both.
    """
    idx, bary, nf = locate_on_sphere(S, F, dirs)
    X = np.asarray(X, dtype=float)
    single = X.ndim == 2
    if single:
        X = X[None]
    out = np.stack([(x[F[idx]] * bary[:, :, None]).sum(1) for x in X])
    return (out[0] if single else out), idx, bary, nf

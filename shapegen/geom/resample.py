"""Resample a sphere-parameterised mesh onto a different sphere discretisation.

A sphere map (:func:`shapegen.codec.base_sphere`) lives on its source mesh's
triangulation, which for a FaceScape head is far from area-uniform (vertex-area
spread ~2e7 on the remeshed mean head). The ρ projection weights by sphere vertex
areas, so :func:`shapegen.data.faces.build_dataset` uses the map once, to answer
"which point of the head does this direction correspond to", and then samples it
at the vertices of a uniform icosphere. The parameterisation is unchanged; only
the sampling is.
"""

from __future__ import annotations

import numpy as np

__all__ = ["locate_on_sphere", "locate_on_mesh", "resample_to"]


def locate_on_sphere(S: np.ndarray, F: np.ndarray, dirs: np.ndarray):
    """For each unit direction, the containing triangle of ``(S, F)`` and its
    barycentric weights. Returns ``(idx (m,), bary (m,3), n_fallback)``.

    A hybrid, because neither half is sufficient alone. libigl's
    ``point_mesh_squared_distance`` supplies the containing face from an AABB tree
    in C++ — robust. But it returns
    the **nearest** point on the mesh, and what is wanted is the **radial**
    intersection: where the ray from the origin along ``d`` crosses the triangle.
    Those coincide on a well-shaped sphere and diverge on a distorted one, so the
    weights are solved for here rather than derived from libigl's closest point.
    Taking igl's point directly produced 88 sliver triangles on a deliberately
    degenerate source.

    Traps any replacement must avoid (both shipped once):

    * searching from the nearest triangle **centroid** misses points inside large
      triangles on a mesh with large area spread;
    * leaving unresolved directions as index ``-1`` with zero weights silently
      indexes the last face and places the vertex at the **origin** -- zero-area
      triangles, ρ values of 1e91, and an ARPACK "starting vector is zero" far
      downstream.

    ``n_fallback`` counts faces whose plane passes through the origin, where the
    radial weights are undefined and the centroid is substituted. On a sphere mesh
    that should be zero; a nonzero count means the "sphere" is not star-shaped
    about the origin and the whole resampling is suspect.

    **Limitation.** Radial projection is ill-conditioned when a face's plane
    passes near the origin. On the eigen sphere of a remeshed FaceScape mean head
    the closest face plane sits 0.69 from the origin (ideal 1.0), and the
    production path returns 0 fallbacks at icosphere subdiv 6. A source with
    *few, huge* near-origin faces does produce collapsed output triangles, so
    check triangle areas on the result if the source is unusual.
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


def locate_on_mesh(V: np.ndarray, F: np.ndarray, pts: np.ndarray):
    """Nearest-point barycentric location of ``pts`` on the surface ``(V, F)``.

    The **non-radial** companion to :func:`locate_on_sphere`. Use it for
    surface-to-surface transfer, where the query points already lie on (or very
    near) the surface and "closest point" is the right answer — locating a
    remeshed copy's vertices back on the original, for instance.

    :func:`locate_on_sphere` must not be used for that: it projects radially from
    the origin, which is only meaningful when the surface is a sphere centred
    there.

    Returns ``(idx (m,), bary (m,3), max_dist)``. ``max_dist`` is how far the
    farthest query sat from the surface — for a remesh of the same shape it should
    be at the scale of the remesh error, and anything larger means the two meshes
    are not the same surface.
    """
    import igl  # noqa: PLC0415

    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    pts = np.asarray(pts, dtype=float)
    d2, idx, C = igl.point_mesh_squared_distance(pts, V, F)
    idx = np.asarray(idx, dtype=np.int64).ravel()
    a, b, c = V[F[idx, 0]], V[F[idx, 1]], V[F[idx, 2]]
    v0, v1, v2 = b - a, c - a, C - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    w1 = (d11 * (v2 * v0).sum(1) - d01 * (v2 * v1).sum(1)) / den
    w2 = (d00 * (v2 * v1).sum(1) - d01 * (v2 * v0).sum(1)) / den
    w = np.clip(np.stack([1.0 - w1 - w2, w1, w2], axis=1), 0.0, None)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-30)
    return idx, w, float(np.sqrt(np.asarray(d2).max()))


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

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

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["locate_on_sphere", "resample_to"]


def locate_on_sphere(S: np.ndarray, F: np.ndarray, dirs: np.ndarray, k: int = 8):
    """For each unit direction, the containing triangle of ``(S, F)`` and its
    barycentric weights. Returns ``(idx (m,), bary (m,3), n_fallback)``.

    Searches from the nearest **vertex** and its 2-ring, not from the nearest
    triangle *centroid*: on a mesh whose triangles span 1e6x in area, a large
    triangle's centroid can be far from points inside it, and a centroid search
    misses them. Measured, that left 39 of 10242 directions unresolved.

    Anything still unresolved falls back to the nearest vertex with unit weight —
    **never** to index -1 with zero weights, which silently indexes the last face
    and places those vertices at the origin. That produced 44 zero-area triangles
    and rho values of 1e91 before the fallback existed, and the failure surfaced
    only much later as an ARPACK "starting vector is zero".
    """
    S = np.asarray(S, dtype=float)
    dirs = np.asarray(dirs, dtype=float)
    v2f = defaultdict(list)
    for f, tri in enumerate(F):
        for v in tri:
            v2f[int(v)].append(f)
    ring = defaultdict(set)
    for tri in F:
        for a in tri:
            ring[int(a)].update(int(x) for x in tri)

    _, near = cKDTree(S).query(dirs, k=k)
    near = np.atleast_2d(near.T).T if near.ndim == 1 else near
    idx = np.full(len(dirs), -1, dtype=np.int64)
    bary = np.zeros((len(dirs), 3))
    n_fallback = 0
    for i, d in enumerate(dirs):
        cands = []
        for v in near[i]:
            for u in ring[int(v)]:
                cands.extend(v2f[u])
        for f in dict.fromkeys(cands):
            M = np.stack([S[F[f, 0]], S[F[f, 1]], S[F[f, 2]]], axis=1)
            try:
                w = np.linalg.solve(M, d)
            except np.linalg.LinAlgError:
                continue
            if (w > -1e-9).all():
                idx[i] = f
                bary[i] = w / w.sum()
                break
        if idx[i] < 0:
            v = int(near[i][0])
            f = v2f[v][0]
            idx[i] = f
            bary[i] = (F[f] == v).astype(float)
            n_fallback += 1
    return idx, bary, n_fallback


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

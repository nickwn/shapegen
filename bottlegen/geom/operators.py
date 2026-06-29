"""Core discrete operators: cotan Laplacian, lumped mass, vertex normals.

Sign convention (used everywhere downstream): ``L`` is the cotan **stiffness**
matrix ``L = D - W`` with ``W_ij = ½(cot α_ij + cot β_ij) ≥ 0`` and
``D_ii = Σ_j W_ij``. ``L`` is symmetric positive-semidefinite (nullspace =
constants) and approximates ``-Δ`` (so ``(L V)_i`` points along the *outward*
mean-curvature vector for a convex surface). ``M`` is the lumped barycentric
mass (diagonal, vertex areas = ⅓ of incident face areas).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = [
    "cotan_laplacian", "vertex_areas", "vertex_normals",
    "edge_cotan_weights", "angle_defect",
]


def _face_cotangents(V: np.ndarray, F: np.ndarray):
    """Cotangent of the angle at each of the three corners of every face.

    Returns ``(cot0, cot1, cot2)`` where ``cot0`` is at vertex ``F[:,0]`` (the
    angle opposite edge ``(F[:,1], F[:,2])``), etc.
    """
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]

    def cot_at(a, b, c):  # angle at vertex a between edges a->b and a->c
        u, v = V[b] - V[a], V[c] - V[a]
        num = (u * v).sum(-1)
        den = np.linalg.norm(np.cross(u, v), axis=-1)
        return num / np.maximum(den, 1e-30)

    return cot_at(i0, i1, i2), cot_at(i1, i2, i0), cot_at(i2, i0, i1)


def cotan_laplacian(V: np.ndarray, F: np.ndarray):
    """Return ``(L, M)``: cotan stiffness (sparse CSR, ``L = D - W``, PSD) and
    lumped barycentric mass (sparse diagonal)."""
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    c0, c1, c2 = _face_cotangents(V, F)
    # Off-diagonal weight on edge (j,k) opposite corner i gets ½·cot(angle at i).
    # Edge (i1,i2)<-c0, edge (i2,i0)<-c1, edge (i0,i1)<-c2 (both directions).
    I = np.concatenate([i1, i2, i2, i0, i0, i1])
    J = np.concatenate([i2, i1, i0, i2, i1, i0])
    Wt = 0.5 * np.concatenate([c0, c0, c1, c1, c2, c2])
    n = len(V)
    W = sp.coo_matrix((Wt, (I, J)), shape=(n, n)).tocsr()
    D = sp.diags(np.asarray(W.sum(axis=1)).ravel())
    L = (D - W).tocsr()
    M = sp.diags(vertex_areas(V, F))
    return L, M


def vertex_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Lumped barycentric vertex areas (⅓ of each incident face area)."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fa = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    a = np.zeros(len(V))
    np.add.at(a, F.ravel(), np.repeat(fa, 3) / 3.0)
    return a


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (unit length)."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)   # length = 2*area (area weighting built in)
    n = np.zeros_like(V)
    np.add.at(n, F[:, 0], fn)
    np.add.at(n, F[:, 1], fn)
    np.add.at(n, F[:, 2], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def edge_cotan_weights(V: np.ndarray, F: np.ndarray):
    """Per-undirected-edge cotan weights ``w_ij = ½(cot α + cot β)``.

    Returns ``(edges, w)`` with ``edges`` shape ``(E,2)`` (sorted ``i<j``) and
    ``w`` shape ``(E,)``. Matches the magnitude of ``L``'s off-diagonal entries;
    used by the Dirac edge-integration Poisson solve.
    """
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    c0, c1, c2 = _face_cotangents(V, F)
    eI = np.concatenate([i1, i2, i0])      # edge (i1,i2),(i2,i0),(i0,i1)
    eJ = np.concatenate([i2, i0, i1])
    cc = 0.5 * np.concatenate([c0, c1, c2])
    key = np.sort(np.stack([eI, eJ], axis=1), axis=1)
    edges, inv = np.unique(key, axis=0, return_inverse=True)
    w = np.zeros(len(edges))
    np.add.at(w, inv.ravel(), cc)
    return edges, w


def angle_defect(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Angle defect ``2π − Σ θ_i`` per vertex (integrated Gaussian curvature)."""
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]

    def angle(a, b, c):  # interior angle at vertex a
        u = V[b] - V[a]
        v = V[c] - V[a]
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-30)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)
        return np.arccos(np.clip((u * v).sum(1), -1.0, 1.0))

    defect = np.full(len(V), 2.0 * np.pi)
    np.add.at(defect, i0, -angle(i0, i1, i2))
    np.add.at(defect, i1, -angle(i1, i2, i0))
    np.add.at(defect, i2, -angle(i2, i0, i1))
    return defect

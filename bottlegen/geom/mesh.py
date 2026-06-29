"""Lightweight triangle-mesh container and analytic test generators.

The numerical kernels in ``operators.py`` / ``curvature.py`` / ``dirac.py`` take
bare ``(V, F)`` arrays (``V`` float64 ``(n,3)``, ``F`` int ``(m,3)``). ``Mesh`` is
a thin convenience wrapper exposing the trimesh-like attributes the data pipeline
(``data/shapenet.py``) reads — ``vertices``, ``faces``, ``area``, ``center_mass``,
``euler_number``, ``genus`` — without taking a hard dependency on ``trimesh``.

Generators (``icosphere``, ``ellipsoid``) build meshes whose curvature has a
closed form, so the curvature kernels can be verified against exact values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Mesh", "icosphere", "ellipsoid", "triangle_areas", "face_normals"]


def triangle_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-face areas (shape ``(m,)``)."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def face_normals(V: np.ndarray, F: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Per-face normals (un-normalized = 2*area*unit-normal unless ``normalize``)."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    if normalize:
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return n


@dataclass
class Mesh:
    """Triangle mesh ``(V, F)`` with a small trimesh-compatible surface API."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)

    # --- trimesh-like attributes used by the data pipeline -----------------
    @property
    def area(self) -> float:
        return float(triangle_areas(self.vertices, self.faces).sum())

    @property
    def center_mass(self) -> np.ndarray:
        """Area-weighted surface centroid (matches trimesh ``center_mass`` for
        a closed, well-tessellated surface closely enough for normalization)."""
        fa = triangle_areas(self.vertices, self.faces)
        c = self.vertices[self.faces].mean(axis=1)
        return (fa[:, None] * c).sum(0) / max(fa.sum(), 1e-30)

    @property
    def euler_number(self) -> int:
        V = len(self.vertices)
        F = len(self.faces)
        # closed triangle mesh: 3F = 2E
        E = self._num_edges()
        return V - E + F

    @property
    def genus(self) -> int:
        # closed surface: V - E + F = 2 - 2g
        return int(1 - self.euler_number / 2)

    def _num_edges(self) -> int:
        e = np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]],
                            self.faces[:, [2, 0]]], axis=0)
        e = np.sort(e, axis=1)
        return len(np.unique(e, axis=0))

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy())


# --------------------------------------------------------------------------- #
# Icosphere / ellipsoid generators (closed-form curvature for tests)
# --------------------------------------------------------------------------- #

def _icosahedron():
    """Unit-radius icosahedron (12 vertices, 20 faces, consistent winding)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    V = np.array([
        [-1,  t,  0], [1,  t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1,  t], [0,  1, t], [0, -1, -t], [0,  1, -t],
        [t,  0, -1], [t,  0, 1], [-t,  0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    return V, F


def icosphere(subdiv: int = 4) -> Mesh:
    """Geodesic icosphere: ``subdiv`` loop subdivisions of the icosahedron, each
    new vertex projected to the unit sphere. Outward, consistently oriented.

    Vertex counts: subdiv 0->12, 1->42, 2->162, 3->642, 4->2562, 5->10242.
    """
    V, F = _icosahedron()
    for _ in range(subdiv):
        V, F = _loop_split(V, F)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
    V, F = _orient_outward(V, F)
    return Mesh(V, F)


def ellipsoid(subdiv: int = 4, axes=(1.0, 0.7, 0.5)) -> Mesh:
    """Icosphere scaled by ``axes`` — points lie exactly on the implicit
    ellipsoid ``Σ (x_i/a_i)^2 = 1``, so closed-form curvature is exact there."""
    m = icosphere(subdiv)
    V = m.vertices * np.asarray(axes, dtype=np.float64)[None, :]
    V2, F2 = _orient_outward(V, m.faces)
    return Mesh(V2, F2)


def _loop_split(V: np.ndarray, F: np.ndarray):
    """One 4-1 midpoint subdivision (no smoothing); dedupes shared midpoints."""
    mid_cache: dict[tuple[int, int], int] = {}
    V_list = [v for v in V]

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        idx = mid_cache.get(key)
        if idx is None:
            idx = len(V_list)
            V_list.append(0.5 * (V[a] + V[b]))
            mid_cache[key] = idx
        return idx

    new_F = []
    for a, b, c in F:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_F += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(V_list, dtype=np.float64), np.array(new_F, dtype=np.int64)


def _orient_outward(V: np.ndarray, F: np.ndarray):
    """Flip faces whose normal points toward the centroid (valid for the
    star-shaped generator meshes used in tests)."""
    c = V.mean(0)
    fn = face_normals(V, F)
    fc = V[F].mean(axis=1) - c
    flip = (fn * fc).sum(1) < 0.0
    F = F.copy()
    F[flip] = F[flip][:, ::-1]
    return V, F

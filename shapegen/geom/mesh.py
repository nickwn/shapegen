"""Lightweight triangle-mesh container and analytic test generators.

The numerical kernels in ``operators.py`` / ``dirac.py`` take
bare ``(V, F)`` arrays (``V`` float64 ``(n,3)``, ``F`` int ``(m,3)``). ``Mesh`` is
a thin convenience wrapper exposing trimesh-like attributes — ``vertices``,
``faces``, ``euler_number``, ``genus`` — without taking a hard dependency on
``trimesh``.

Generators (``icosphere``, ``ellipsoid``) build meshes whose curvature has a
closed form, so the curvature kernels can be verified against exact values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Mesh", "icosphere", "ellipsoid", "triangle_areas", "face_normals",
           "remesh_isotropic"]


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


def remesh_isotropic(V: np.ndarray, F: np.ndarray, n_faces: int = 12000,
                     iterations: int = 10, feature_deg: float = 30.0):
    """The same surface with ~``n_faces`` well-shaped triangles: pymeshlab's
    isotropic explicit remeshing at the edge length that gives ``n_faces``
    equilateral triangles of the input's area, with vertices reprojected onto the
    input and creases sharper than ``feature_deg`` kept. Returns ``(V, F)``.

    For coarse inputs, which the conformal sphere map cannot use directly (a
    12-triangle cube crashes it, and 768 triangles decode to garbage). The result
    is independent of how the input was triangulated. Measured on a cube, the
    12-triangle mesh, a centre fan, sliver fans (angles 2.9-132 deg) and a
    12288-triangle grid all decode with the same deviation from the cube:
    mean 0.0033, p99 0.019. Midpoint subdivision alone keeps slivers slivers,
    and the sphere map degrades on them (spectrum error 0.0145 vs 0.0030).
    Geometry is unchanged for polyhedra: the remeshed cube's vertices lie on the
    cube to 0.
    """
    import pymeshlab  # noqa: PLC0415

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    h = np.sqrt(4.0 * triangle_areas(V, F).sum() / (np.sqrt(3.0) * n_faces))
    while len(F) < 200:                   # give the remesher enough edges to work with
        V, F = _loop_split(V, F)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F.astype(np.int32)))
    ms.meshing_isotropic_explicit_remeshing(iterations=iterations,
                                           targetlen=pymeshlab.PureValue(h),
                                           featuredeg=feature_deg, reprojectflag=True)
    m = ms.current_mesh()
    return m.vertex_matrix().astype(np.float64), m.face_matrix().astype(np.int64)


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

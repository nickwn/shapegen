"""Coarse inputs encode the *surface*, not its tessellation: ``codec.encode``
remeshes meshes too coarse for the sphere map, and a 12-triangle cube decodes
to the same shape as any other triangulation of the cube."""

from __future__ import annotations

import numpy as np
import pytest

from shapegen import codec
from shapegen.geom.mesh import Mesh, remesh_isotropic
from shapegen.quad.lift import lift_spectral


def _orient(V, F):
    c = V[F].mean(1)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    F = F.copy()
    F[(n * c).sum(1) < 0] = F[(n * c).sum(1) < 0][:, ::-1]
    return V, F


def cube(n: int = 1):
    """[-1,1]³, each face an n×n grid of quads split into triangles."""
    verts, faces, index = [], [], {}

    def vid(p):
        key = tuple(np.round(p, 12))
        if key not in index:
            index[key] = len(verts)
            verts.append(p)
        return index[key]
    g = np.linspace(-1, 1, n + 1)
    for axis in range(3):
        for sgn in (-1.0, 1.0):
            u, v = [a for a in range(3) if a != axis]
            for i in range(n):
                for j in range(n):
                    q = []
                    for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        p = np.zeros(3)
                        p[axis], p[u], p[v] = sgn, g[a], g[b]
                        q.append(vid(p))
                    faces += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    return _orient(np.array(verts), np.array(faces))


def fan_cube():
    """Each face a fan of 4 triangles about its centre: 14 vertices, 24 faces."""
    V, _ = cube(1)
    verts, faces = list(V), []
    for axis in range(3):
        for sgn in (-1.0, 1.0):
            c = np.zeros(3)
            c[axis] = sgn
            verts.append(c)
            on = [i for i in range(8) if V[i, axis] == sgn]
            u, v = [a for a in range(3) if a != axis]
            on.sort(key=lambda i: np.arctan2(V[i, v], V[i, u]))
            faces += [[len(verts) - 1, on[k], on[(k + 1) % 4]] for k in range(4)]
    return _orient(np.array(verts), np.array(faces))


def _spectrum(coeffs, L):
    """Per-degree power of the decoded positions: invariant to rotations of both
    the sphere parameterisation and the output."""
    fh = lift_spectral(np.eye(3), coeffs, L)[1]["f_hat"]
    ell = np.floor(np.sqrt(np.arange(len(fh))))
    P = np.array([(fh[ell == l] ** 2).sum() for l in range(1, int(ell.max()) + 1)])
    return P / P.sum()


def test_remesh_keeps_the_cube():
    V, F = remesh_isotropic(*cube(1), n_faces=3000)
    assert 2000 < len(F) < 4500
    q = np.abs(V) - 1.0                                     # box SDF of every new vertex
    sdf = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    assert np.abs(sdf).max() < 1e-9


def test_sphere_map_refuses_coarse_meshes_clearly():
    with pytest.raises(ValueError, match="too coarse"):
        codec.base_sphere(Mesh(*cube(1)))


def test_coarse_cube_decodes_like_any_cube():
    L = 8
    r = codec.encode(Mesh(*cube(1)), L, method="quotient")     # 12 triangles: auto-remeshed
    assert r.sphere.shape == (8, 3) and "refined_mesh" in r.meta
    assert r.meta["err_surface"] < 0.04                      # the band limit at L=8: 0.033
    assert codec.shape_error(cube(1)[0], codec.decode(r.coeffs, r.sphere)) == pytest.approx(r.err)
    ref = _spectrum(codec.encode(Mesh(*cube(16)), L, method="quotient", refine=True).coeffs, L)
    for V, F in (cube(1), fan_cube()):
        p = _spectrum(codec.encode(Mesh(V, F), L, method="quotient").coeffs, L)
        assert np.linalg.norm(p - ref) < 1e-4 * np.linalg.norm(ref)   # measured 1-3e-6

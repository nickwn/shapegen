"""The exact polyhedral encoder (:mod:`shapegen.geom.poly_conformal`,
``codec.encode(method="polyhedral")``): the sphere map is exact, its Jacobian is
right, and the code depends on the polyhedron, not on its tessellation."""

from __future__ import annotations

import numpy as np
import pytest

from shapegen import codec
from shapegen.geom import poly_conformal as pc
from shapegen.geom.mesh import Mesh

L = 8


def cube_mesh(n, flip=False, dims=(1.0, 1.0, 1.0)):
    """[-1,1]³ (scaled by ``dims``), each face an n×n grid of quads split in two."""
    verts, faces, index = [], [], {}

    def vid(p):
        k = tuple(np.round(p, 12))
        if k not in index:
            index[k] = len(verts)
            verts.append(p)
        return index[k]

    g = np.linspace(-1, 1, n + 1)
    for axis in range(3):
        for sgn in (-1.0, 1.0):
            u, v = [a for a in range(3) if a != axis]
            for i in range(n):
                for j in range(n):
                    def P(a, b):
                        p = np.zeros(3); p[axis] = sgn; p[u] = g[a]; p[v] = g[b]
                        return p
                    q = [vid(P(i, j)), vid(P(i + 1, j)), vid(P(i + 1, j + 1)), vid(P(i, j + 1))]
                    if flip and (i + j) % 2:
                        faces += [[q[0], q[1], q[3]], [q[1], q[2], q[3]]]
                    else:
                        faces += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    V, F = np.array(verts) * np.array(dims), np.array(faces)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    bad = (nrm * V[F].mean(1)).sum(1) < 0
    F[bad] = F[bad][:, ::-1]
    return V, F


def subdivide_flat(V, F, k=1):
    """Split every triangle into 4 in its own plane: the same polyhedron."""
    for _ in range(k):
        edges, V, NF = {}, list(V), []

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in edges:
                edges[key] = len(V)
                V.append((np.asarray(V[i]) + np.asarray(V[j])) / 2)
            return edges[key]

        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            NF += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        V, F = np.array(V), np.array(NF)
    return V, F


def hull_poly(seed=3, n=14):
    """An irregular convex polyhedron (9 vertices, all cones)."""
    from scipy.spatial import ConvexHull

    P = np.random.default_rng(seed).normal(size=(n, 3)) * [1.0, 0.7, 1.3]
    F = ConvexHull(P).simplices.copy()
    nrm = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    bad = (nrm * (P[F].mean(1) - P.mean(0))).sum(1) < 0
    F[bad] = F[bad][:, ::-1]
    used = np.unique(F)
    remap = -np.ones(n, int)
    remap[used] = np.arange(len(used))
    return P[used], remap[F]


def spectrum(rho):
    Lc = int(round(np.sqrt(len(rho)))) - 1
    return np.array([np.linalg.norm(rho[l * l:(l + 1) ** 2]) for l in range(1, Lc + 1)])


def test_angle_defects_find_only_the_corners():
    V, F = cube_mesh(3)
    k = pc.angle_defects(V, F)
    assert np.isclose(k.sum(), 4 * np.pi)
    corner = np.all(np.isclose(np.abs(V), 1.0), axis=1)
    assert np.allclose(k[corner], np.pi / 2) and np.allclose(k[~corner], 0, atol=1e-12)


def test_jacobian_matches_finite_differences():
    from scipy.optimize._numdiff import approx_derivative

    V, F = subdivide_flat(*hull_poly(), 1)
    pm = pc.PolyMap(V, F)
    x = np.r_[pm.z.real, pm.z.imag, 0.1] + 0.02 * np.random.default_rng(0).normal(size=2 * len(V) + 1)
    pm.x0 = x.copy()
    Jfd = approx_derivative(pm.residual, x, method="3-point")
    assert np.abs(pm.jacobian(x).toarray() - Jfd).max() < 1e-6 * np.abs(Jfd).max()


@pytest.mark.parametrize("mesh", ["cube", "box", "hull", "hull_sub2"])
def test_sphere_map_is_exact(mesh):
    """Every face's developed triangle is congruent to its 3D triangle, and the
    quadrature samples carry the exact surface area and cover the sphere."""
    V, F = {"cube": lambda: cube_mesh(1), "box": lambda: cube_mesh(1, dims=(1, 2, 3)),
            "hull": hull_poly, "hull_sub2": lambda: subdivide_flat(*hull_poly(), 2)}[mesh]()
    pm = pc.solve_map(V, F)
    assert pm.fit["max"] < 1e-7, pm.fit
    pm.gauge_cones()
    xs, wA, Xs = pm.surface_samples(6000)
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()
    assert pm.inv_fail <= 3 and np.isclose(wA.sum(), area, rtol=1e-3)
    _, s2 = pm.spinor_at(xs, pm.sample_faces)
    assert np.isclose((wA / s2).sum(), 4 * np.pi, rtol=1e-3)                 # sphere area


def test_code_is_tessellation_invariant():
    """Same polyhedron, different triangles (coarse, flipped diagonals, flat-subdivided):
    the codes agree to ~1e-5 in rotation-invariant spectrum."""
    ref = spectrum(pc.encode_polyhedral(*cube_mesh(1), L)["coeffs"])
    for V, F in (cube_mesh(4, flip=True), subdivide_flat(*cube_mesh(1), 2)):
        s = spectrum(pc.encode_polyhedral(V, F, L)["coeffs"])
        assert np.linalg.norm(s - ref) < 1e-4 * np.linalg.norm(ref)
    ref = spectrum(pc.encode_polyhedral(*hull_poly(), L)["coeffs"])
    s = spectrum(pc.encode_polyhedral(*subdivide_flat(*hull_poly(), 2), L)["coeffs"])
    assert np.linalg.norm(s - ref) < 1e-4 * np.linalg.norm(ref)


def test_codec_polyhedral_method():
    V, F = cube_mesh(1)
    mesh = Mesh(V, F)
    r = codec.encode(mesh, L, method="polyhedral")
    assert r.gamma < 1e-10
    assert codec.shape_error(V, codec.decode(r.coeffs, r.sphere)) == pytest.approx(r.err, rel=1e-3, abs=1e-8)
    remesh = codec.encode(mesh, L, method="quotient")
    assert r.meta["err_surface"] < 1.05 * remesh.meta["err_surface"], (r.meta["err_surface"], remesh.meta["err_surface"])
    with pytest.raises(ValueError):
        codec.encode(mesh, L, sphere=r.sphere, method="polyhedral")

"""The polyhedral map and encode (``codec.encode(method="polyhedral")``): the start
is the discrete uniformization, the exact map's Newton solve (:class:`EdgeSystem`,
the encoder's fallback) reaches round-off, the face developments of
:mod:`shapegen.geom.poly_conformal` agree with the edge periods, and the code
depends on the polyhedron, not on its tessellation."""

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


def tetrahedron():
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1.0]])
    F = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    return V, F


def prism(n):
    """An n-gon prism, each cap fanned from a centre vertex: 2n cones."""
    th = 2 * np.pi * np.arange(n) / n
    ring = np.c_[np.cos(th), np.sin(th)]
    V = np.r_[np.c_[ring, -np.ones(n)], np.c_[ring, np.ones(n)], [[0, 0, -1.0], [0, 0, 1.0]]]
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V, np.array(F)


def solve_map(V, F):
    """The polyhedron's exact sphere map: :meth:`EdgeSystem.solve` from its discrete
    uniformization (the encoder's fallback on very coarse meshes). Returns
    ``(X, info)``."""
    from shapegen.geom.edge_periods import EdgeSystem
    from shapegen.geom.uniformize import uniformize

    X, _u, info = EdgeSystem(V, F).solve(uniformize(V, F))
    return X, info


def congruence(V, F, X):
    """Largest deviation of a developed face from its 3D triangle (log side ratio,
    angle), every face developed from its vertex-mean chart centre."""
    pm = pc.PolyMap(V, F, S0=X)
    _, _, D = pm._develop_at(X, np.arange(len(F)), X[F].mean(1))
    return pm._shape_err(D)[0].max()


def encode(V, F, L):
    from shapegen.geom.chord_map import encode_chord
    return encode_chord(V, F, L)["coeffs"]


def test_angle_defects_find_only_the_corners():
    V, F = cube_mesh(3)
    k = pc.angle_defects(V, F)
    assert np.isclose(k.sum(), 4 * np.pi)
    corner = np.all(np.isclose(np.abs(V), 1.0), axis=1)
    assert np.allclose(k[corner], np.pi / 2) and np.allclose(k[~corner], 0, atol=1e-12)


def star():
    """An icosahedron with a pyramid on every face: saddle cones at its vertices. Its
    uniformization folds two faces, so the solve's length phase must unfold them."""
    from scipy.spatial import ConvexHull

    p = (1 + 5 ** 0.5) / 2
    P = np.array([[0, a, b] for a in (-1, 1) for b in (-p, p)] + [[a, b, 0] for a in (-1, 1) for b in (-p, p)]
                 + [[b, 0, a] for a in (-1, 1) for b in (-p, p)], float)
    F = ConvexHull(P).simplices
    out = (np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]) * P[F].mean(1)).sum(1) < 0
    F = np.where(out[:, None], F[:, ::-1], F)
    V, NF = list(P), []
    for a, b, c in F:
        V.append(1.6 * (P[a] + P[b] + P[c]) / 3)
        NF += [[a, b, len(V) - 1], [b, c, len(V) - 1], [c, a, len(V) - 1]]
    return np.array(V), np.array(NF)


def test_uniformize_recovers_an_inscribed_polyhedron():
    """Discrete uniformization of a polyhedron inscribed in the sphere returns its own
    vertices up to a Möbius map: all cross-ratios agree."""
    from scipy.spatial import ConvexHull

    from shapegen.geom.uniformize import uniformize

    rng = np.random.default_rng(0)
    for n in (8, 50, 400):
        P = rng.normal(size=(n, 3))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        F = ConvexHull(P).simplices
        out = (np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]) * P[F].mean(1)).sum(1) < 0
        S = uniformize(2.0 * P, np.where(out[:, None], F[:, ::-1], F))
        q = rng.choice(n, (500, 4))
        q = q[np.all(np.diff(np.sort(q, 1), axis=1) > 0, 1)]

        def cr(X):
            z = (X[:, 0] + 1j * X[:, 1]) / (1 - X[:, 2])
            a, b, c, d = (z[q[:, k]] for k in range(4))
            return np.abs((a - c) * (b - d) / ((a - d) * (b - c)))

        assert np.abs(np.log(cr(S) / cr(P))).max() < 1e-9


def test_edge_periods_match_face_developments():
    """Away from the solution the edge periods (FMM, great arcs) equal the faces'
    developed sides (rays from each face's chart centre), and the Möbius directions
    leave them unchanged."""
    from shapegen.geom.conformal import ball_automorphism
    from shapegen.geom.edge_periods import EdgeSystem
    from shapegen.geom.uniformize import uniformize

    V, F = subdivide_flat(*hull_poly(), 1)
    es = EdgeSystem(V, F)
    X = uniformize(V, F) + 0.01 * np.random.default_rng(0).normal(size=V.shape)
    X = es.centre(X / np.linalg.norm(X, axis=1, keepdims=True))
    I = np.abs(es.periods(X))
    pm = pc.PolyMap(V, F, S0=X)
    _, _, D = pm._develop_at(X, np.arange(len(F)), X[F].mean(1))
    for k in range(3):
        i, j = F[:, k], F[:, (k + 1) % 3]
        e = np.searchsorted(es.E[:, 0] * len(V) + es.E[:, 1], np.minimum(i, j) * len(V) + np.maximum(i, j))
        assert np.allclose(np.abs(D[:, (k + 1) % 3] - D[:, k]), I[e], rtol=1e-10)
    Y = ball_automorphism(X, np.array([0.05, -0.03, 0.02]))
    assert np.allclose(np.abs(es.periods(Y)) / I, np.mean(np.abs(es.periods(Y)) / I), rtol=1e-10)


def test_cone_opposite_a_face():
    """A regular tetrahedron's vertex sits at the pole of the opposite face's chart."""
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1.0]])
    F = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    _, info = solve_map(V, F)
    assert info["max"] < 1e-12, info


@pytest.mark.parametrize("make", [lambda: hull_poly(seed=0), hull_poly, star,
                                  lambda: subdivide_flat(*hull_poly(), 2)], ids=["hull0", "hull", "star", "hull_s2"])
def test_hard_starts_converge(make):
    """Spiky hulls (the face solver stalled on hull0) and a start with folded faces
    (star) reach round-off, and the face development agrees."""
    V, F = make()
    X, info = solve_map(V, F)
    assert info["max"] < 1e-10, info
    assert congruence(V, F, X) < 1e-10


def test_prism_encodes_deterministically():
    """Fanned caps make the sample density jump across edges, which disconnected the
    sign-alignment graph. The rotation gauge makes repeat encodes identical."""
    V, F = prism(12)
    a, b = (encode(V, F, L) for _ in range(2))
    assert np.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("mesh", ["cube", "box", "hull", "hull_sub2"])
def test_sphere_map_is_exact(mesh):
    """The map's Newton solve reaches round-off with equal scales (a true root), and
    every face's developed triangle is congruent to its 3D triangle."""
    V, F = {"cube": lambda: cube_mesh(1), "box": lambda: cube_mesh(1, dims=(1, 2, 3)),
            "hull": hull_poly, "hull_sub2": lambda: subdivide_flat(*hull_poly(), 2)}[mesh]()
    X, info = solve_map(V, F)
    assert info["max"] < 1e-10 and info["u_spread"] < 1e-9, info
    assert congruence(V, F, X) < 1e-10


def test_code_is_tessellation_invariant():
    """Same polyhedron, different triangles (coarse, flipped diagonals, flat-subdivided):
    the codes themselves agree (the rotation is fixed by the cones), to the
    quadrature's error (measured: cube 5e-7, hull 1.6e-6; small faces next to
    strong cones held this at 1.3e-4 before the cone term in the order rule)."""
    ref = encode(*cube_mesh(1), L)
    for V, F in (cube_mesh(4, flip=True), subdivide_flat(*cube_mesh(1), 2)):
        c = encode(V, F, L)
        assert np.linalg.norm(c - ref) < 5e-6 * np.linalg.norm(ref)
    ref = encode(*hull_poly(), L)
    c = encode(*subdivide_flat(*hull_poly(), 2), L)
    assert np.linalg.norm(c - ref) < 1e-5 * np.linalg.norm(ref)


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


def test_edge_system_short_arcs_and_strain_steps():
    """Short great arcs keep their digits (an ``arccos`` angle lost half of them: 4e-8
    relative at 5e-5), so the strain-scaled differences of ``J·v`` are linear."""
    from shapegen.geom.edge_periods import _slerp

    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 3))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    T = np.cross(A, rng.normal(size=(50, 3)))
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    for beta in (5e-5, 1e-7):
        B = A * np.cos(beta) + T * np.sin(beta)
        x, dx = _slerp(A, B, np.ones((50, 1)))
        assert np.abs(np.linalg.norm(dx[:, 0], axis=1) / beta - 1).max() < 3e-15 / beta   # arccos: 1e-16 / beta²
        assert np.abs(x[:, 0] - B).max() < 1e-15


def test_near_flat_face_rows_are_flipped():
    """A face with a 3D angle near 180° gets its rows from an intrinsic flip (the
    metric unchanged), and the map still converges to a true root (equal scales)."""
    from shapegen.geom.edge_periods import EdgeSystem

    V, F = tetrahedron()
    m = 0.5 * (V[0] + V[1]) + 0.004 * (V[2] - 0.5 * (V[0] + V[1]))
    V = np.r_[V, [m]]
    F = np.r_[F[1:], [[0, 1, 4], [1, 2, 4], [2, 0, 4]]]
    es = EdgeSystem(V, F)
    assert (np.sort(es.F, 1) != np.sort(F, 1)).any(1).sum() == 2
    _, info = solve_map(V, F)
    assert info["max"] < 1e-10 and info["u_spread"] < 1e-9, info

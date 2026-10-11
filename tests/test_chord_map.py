"""The fast polyhedral encoder (:mod:`shapegen.geom.chord_map`): its local periods
are the exact periods when every cone is near, its Möbius spinor is the derivative
of its own face maps, its nodes tile the sphere, and on a dense smooth mesh its
code is the exact encoder's."""

from __future__ import annotations

import numpy as np
import pytest

from shapegen.geom import chord_map as cm
from shapegen.geom import poly_conformal as pc
from shapegen.geom.edge_periods import EdgeSystem
from shapegen.geom.uniformize import uniformize

from .test_poly_conformal import hull_poly, star


def fib_ellipsoid(n, axes=(1.0, 0.7, 0.5)):
    """The convex hull of a Fibonacci sphere, stretched: a dense, smooth, anisotropic mesh."""
    from scipy.spatial import ConvexHull

    k = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * k / n), np.pi * (1 + 5 ** 0.5) * k
    P = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)] * axes
    F = ConvexHull(P).simplices
    bad = (np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]) * P[F].mean(1)).sum(1) < 0
    F[bad] = F[bad][:, ::-1]
    return P, F


@pytest.mark.parametrize("make", [lambda: fib_ellipsoid(200), hull_poly, star])
def test_local_periods_with_every_cone_are_exact(make):
    """With every cone in the near set, ``log ρ_e`` is the exact edge period's
    ``log|I_e| − log chord_e − (φ_i + φ_j)/2``: the complex period (not the length of
    the arc's image, which differs by up to 3e-2 on the hull)."""
    V, F = make()
    V = np.asarray(V, float)
    es = EdgeSystem(V, F)
    X = es.centre(uniformize(V, es.F, lengths=(es.E, es.logl)))
    ne, n = len(es.E), es.n
    lr, A = cm.local_log_ratio(X, es.E, es.a_v, np.arange(ne + 1) * n, np.tile(np.arange(n), ne), nq=8)
    i, j = es.E.T
    lc = 0.5 * np.log(np.maximum(2 - 2 * X @ X.T, 1e-300))
    np.fill_diagonal(lc, 0.0)
    phi = lc @ es.a_v
    exact = np.log(es.lengths(X)) - np.log(np.linalg.norm(X[i] - X[j], axis=1)) - 0.5 * (phi[i] + phi[j])
    assert np.abs(lr - exact).max() < 1e-10
    assert np.allclose(A, -2.0)


def test_mobius_spinor_is_the_face_maps_derivative():
    """Each face's Möbius map sends its sphere corners to its 3D corners, and λ is its
    derivative: ``dX = λ̄ t λ`` for a tangent ``t`` (finite differences)."""
    from shapegen.spectral.quaternion import sandwich

    V, F = fib_ellipsoid(400)
    X = uniformize(V, F)
    mf = cm.MobiusFaces(X, V, F)
    allf = np.arange(len(F))
    for c in range(3):
        assert np.abs(mf.points(X[F[:, c]], allf) - V[F[:, c]]).max() < 1e-12
    rng = np.random.default_rng(0)
    f = rng.integers(0, len(F), 200)
    x = pc._normalize(np.einsum("nk,nki->ni", rng.dirichlet([1, 1, 1], 200), X[F[f]]))
    t = pc._normalize(np.cross(x, rng.normal(size=(200, 3))))
    eps = 1e-6
    dX = (mf.points(pc._normalize(x + eps * t), f) - mf.points(pc._normalize(x - eps * t), f)) / (2 * eps)
    assert np.abs(dX - sandwich(mf.spinor(x, f), t)).max() < 1e-8 * np.abs(dX).max()


def test_sphere_nodes_tile_the_sphere():
    V, F = fib_ellipsoid(400)
    X = uniformize(V, F)
    x, w, f = cm.sphere_nodes(X, F, 12.0)
    assert np.isclose(w.sum(), 4 * np.pi, atol=1e-12)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)
    assert len(np.unique(f)) == len(F)


def test_fast_encode_matches_the_exact_one_on_a_dense_mesh(monkeypatch):
    """A dense smooth mesh: the fast code is the exact code to its approximation
    (measured 3.4e-3 at 400 vertices, L=8). ``exact=None`` picks the exact map below
    ``EXACT_MAX_VERTICES`` and the fast one above."""
    V, F = fib_ellipsoid(400)
    L = 8
    e = pc.encode_polyhedral(V, F, L)
    assert e["info"]["solver"] == "exact"
    monkeypatch.setattr(pc, "EXACT_MAX_VERTICES", 100)
    c = pc.encode_polyhedral(V, F, L)
    assert c["info"]["solver"] == "fast" and c["gamma"] < 1e-10
    assert np.linalg.norm(c["coeffs"] - e["coeffs"]) < 1e-2 * np.linalg.norm(e["coeffs"])

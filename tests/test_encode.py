"""The SH primitives the encoders share -- basis, projection, the Procrustes
objective and the preconditioner -- and the direct measurement of ρ."""

import numpy as np
import pytest

from shapegen.geom import encode as enc
from shapegen.geom import spin
from shapegen.geom.mesh import icosphere


def _sphere(sub):
    m = icosphere(sub)
    return m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True), m.faces


# --------------------------------------------------------------------------- #
# The basis                                                                   #
# --------------------------------------------------------------------------- #

def test_sh_basis_slices_by_bandwidth():
    """The first ``(L'+1)²`` columns *are* the bandwidth-L' basis.

    The sweep builds one basis at the largest L and slices it for every smaller
    one, so this layout is load-bearing, not a convention.
    """
    V, _ = _sphere(3)
    Y = enc.real_sh_basis(V, 8)
    for L in (0, 1, 4, 8):
        assert np.array_equal(Y[:, :enc.n_coeffs(L)], enc.real_sh_basis(V, L))


def test_sh_basis_is_orthonormal_under_exact_quadrature():
    """Orthonormal on the sphere. Checked against a Gauss–Legendre grid rather
    than the mesh, whose lumped-area quadrature is only ~1e-3 accurate and would
    make a real basis error indistinguishable from a quadrature error."""
    L = 6
    nth, nph = 2 * L + 2, 2 * L + 2
    x, w = np.polynomial.legendre.leggauss(nth)
    th = np.arccos(x)
    ph = np.arange(nph) * 2 * np.pi / nph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W = np.repeat(w[:, None], nph, axis=1) * (2 * np.pi / nph)
    dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                     np.cos(TH)], axis=-1).reshape(-1, 3)
    Y = enc.real_sh_basis(dirs, L)
    G = (Y * W.ravel()[:, None]).T @ Y
    assert np.abs(G - np.eye(G.shape[0])).max() < 1e-10


def test_band_index_matches_layout():
    assert len(enc.band_index(5)) == enc.n_coeffs(5)
    assert list(enc.band_index(2)) == [0, 1, 1, 1, 2, 2, 2, 2, 2]


# --------------------------------------------------------------------------- #
# The objective                                                               #
# --------------------------------------------------------------------------- #

def test_procrustes_gradient_matches_fd():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 3))
    X = A + 0.3 * rng.standard_normal((30, 3))
    _, g = enc.procrustes(A, X)
    h = 1e-7
    gfd = np.empty_like(X)
    for v in range(len(X)):
        for c in range(3):
            xp, xm = X.copy(), X.copy()
            xp[v, c] += h
            xm[v, c] -= h
            gfd[v, c] = (enc.procrustes(A, xp)[0] - enc.procrustes(A, xm)[0]) / (2 * h)
    assert np.linalg.norm(g - gfd) / np.linalg.norm(gfd) < 1e-6
    assert float((g * gfd).sum() / (gfd * gfd).sum()) == pytest.approx(1.0, abs=1e-6)


def test_procrustes_is_similarity_invariant():
    """Rotation, translation and scale invariance — the gauge-invariance the
    adjoint requires of any loss it differentiates."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((30, 3))
    X = A + 0.2 * rng.standard_normal((30, 3))
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    base = enc.procrustes(A, X)[0]
    moved = enc.procrustes(A, 3.7 * X @ Q + np.array([1.0, -2.0, 0.5]))[0]
    assert moved == pytest.approx(base, rel=1e-12)
    assert enc.procrustes(A, A)[0] < 1e-20


# --------------------------------------------------------------------------- #
# End to end                                                                  #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Preconditioning                                                             #
# --------------------------------------------------------------------------- #

def test_band_precond_shape_and_monotonicity():
    d = enc.band_precond(6, alpha=0.75)
    assert d.shape == (enc.n_coeffs(6),)
    assert (d > 0).all() and d.max() == pytest.approx(1.0)
    # constant within a band, increasing across bands
    b = enc.band_index(6)
    per_band = [d[b == l][0] for l in range(7)]
    assert all(np.allclose(d[b == l], per_band[l]) for l in range(7))
    assert all(x < y for x, y in zip(per_band[1:], per_band[2:]))
    assert np.allclose(enc.band_precond(6, alpha=0.0), 1.0)


def test_direct_encoder_is_exact_on_spheres():
    """``ρ = H·e^u − 1`` must vanish on a sphere at any radius: H = 1/R and
    e^u = R, so the product is 1 and ρ is 0.

    The *median* is the convergence quantity, not the max — an icosphere's 12
    valence-5 vertices are a fixed tessellation artifact whose curvature error
    does not shrink (exactly 12 vertices exceed 0.05 at every resolution).
    """
    from shapegen.geom.mesh import Mesh, icosphere
    prev = None
    for sub in (2, 3, 4):
        m = icosphere(sub)
        S = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
        for R in (1.0, 2.5):                     # scale-invariant by construction
            rho = spin.mesh_to_rho_direct(Mesh(S * R, m.faces), S)
            assert int((np.abs(rho) > 0.05).sum()) == 12
        med = float(np.median(np.abs(rho)))
        if prev is not None:
            assert med < prev / 2.0              # O(h^2) under refinement
        prev = med
    assert prev < 1e-3

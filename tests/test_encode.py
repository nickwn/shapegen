"""The SH primitives the encoders share -- basis, projection, the Procrustes
objective and the preconditioner -- and the direct measurement of ρ."""

import numpy as np
import pytest

from shapegen.geom import encode as enc
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


def test_eval_real_sh_is_the_basis_product():
    """:func:`eval_real_sh` evaluates ``real_sh_basis(V, L) @ c`` without building the
    basis, including at the poles (where ``φ`` is undefined)."""
    rng = np.random.default_rng(2)
    V = np.r_[rng.standard_normal((300, 3)), [[0, 0, 1.0], [0, 0, -2.0]]]
    for L in (0, 3, 20):
        c = rng.standard_normal((enc.n_coeffs(L), 2))
        ref = enc.real_sh_basis(V, L) @ c
        assert np.abs(enc.eval_real_sh(V, c, L) - ref).max() < 1e-12 * np.abs(ref).max()
        assert np.allclose(enc.eval_real_sh(V, c[:, 0], L), ref[:, 0])


# --------------------------------------------------------------------------- #
# Procrustes                                                                  #
# --------------------------------------------------------------------------- #

def test_procrustes_is_similarity_invariant():
    """Rotation, translation and scale invariance: the decode is defined up to a
    similarity."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((30, 3))
    X = A + 0.2 * rng.standard_normal((30, 3))
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    base = enc.procrustes(A, X)
    moved = enc.procrustes(A, 3.7 * X @ Q + np.array([1.0, -2.0, 0.5]))
    assert moved == pytest.approx(base, rel=1e-12)
    assert enc.procrustes(A, A) < 1e-20



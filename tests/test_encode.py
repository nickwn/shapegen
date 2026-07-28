"""Acceptance tests for the SH encoder.

The adjoint is gated in ``test_dirac_adjoint.py``; what is tested here is the
layer on top — the basis, the objective, and that optimising actually beats the
local encoder it starts from.
"""

import numpy as np
import pytest

from bottlegen.geom import encode as enc
from bottlegen.geom.mesh import icosphere
from bottlegen.geom.operators import vertex_areas


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

def test_encode_sh_beats_its_initialiser():
    """The whole point of the optimised encoder: it must improve on the local
    encoder it starts from, on a real non-star-shaped mesh."""
    from bottlegen.data.dataset import demo_bottle_meshes
    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
    out = enc.encode_sh(mesh, L=8, maxiter=20)
    print(f"[encode L=8] init={out['err0']:.4f} -> {out['err']:.4f} "
          f"({out['err0'] / out['err']:.2f}x, {out['niter']} iters)")
    assert out["err"] < out["err0"]
    assert out["err"] < 0.6 * out["err0"]
    assert out["coeffs"].shape == (enc.n_coeffs(8),)


def test_decode_sh_inverts_encode_sh():
    """``decode_sh(encode_sh(mesh))`` reproduces the mesh to the reported error —
    i.e. the number the encoder reports is the number you actually get back."""
    from bottlegen.data.dataset import demo_bottle_meshes
    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
    out = enc.encode_sh(mesh, L=8, maxiter=20)
    X = enc.decode_sh(out["coeffs"], out["sph"], mesh.faces, L=8)
    err = float(np.sqrt(enc.procrustes(mesh.vertices, X)[0]))
    assert err == pytest.approx(out["err"], rel=1e-9)

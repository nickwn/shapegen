"""The quotient encoder (:mod:`shapegen.geom.dirac_quotient`) and the two pieces
it rests on: the separable face projection in ``dirac_direct.mesh_spinor`` and
the spanning-tree sign alignment in ``spin.face_spinors``."""

from __future__ import annotations

import numpy as np
import pytest

from shapegen import codec
from shapegen.geom import dirac_direct as dd, dirac_quotient as dq, dirac_spectral as ds, spin
from shapegen.geom.encode import n_coeffs
from shapegen.geom.mesh import icosphere
from shapegen.spectral import swsh as sw
from shapegen.spectral.quaternion import qmul

from tests.quad_fixtures import bundle_style_coeffs

L = 8
LS = ds.spinor_band(L, ds.potential_band_extra(L))


@pytest.fixture(scope="module")
def closed():
    rho = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * 2.0
    rho = ds.close_potential(rho, ds.solve_spinor(rho, L, Ls=LS))
    res = ds.solve_spinor(rho, L, Ls=LS)
    return rho, np.r_[res.psi1, res.psi2]


def _sphere(k):
    m = icosphere(k)
    return m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True), m.faces


def _dual_edges(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    fid = np.tile(np.arange(len(F)), 3)
    o = np.lexsort((e[:, 1], e[:, 0]))
    e, fid = e[o], fid[o]
    same = np.all(e[1:] == e[:-1], axis=1)
    return fid[:-1][same], fid[1:][same]


def test_face_projection_matches_exact_ladders():
    """The θ-table interpolation reproduces the per-point Wigner ladders."""
    rng = np.random.default_rng(0)
    th, ph = np.arccos(rng.uniform(-1, 1, 3000)), rng.uniform(-np.pi, np.pi, 3000)
    g1, g2 = rng.standard_normal((2, 3000)) + 1j * rng.standard_normal((2, 3000))
    exact = np.r_[sw.half_eval_adjoint(+0.5, 12.5, th, ph, g1),
                  sw.half_eval_adjoint(-0.5, 12.5, th, ph, g2)]
    fast = dd._project_faces(12.5, th, ph, g1, g2)
    assert np.linalg.norm(fast - exact) < 1e-5 * np.linalg.norm(exact)


def test_signs_smooth_field_is_one_global_sign():
    S, F = _sphere(3)
    c = S[F].mean(1)
    lam = np.c_[np.cos(c[:, 2]), np.sin(c[:, 2]), 0.3 * c[:, 0], 0 * c[:, 0]]
    flips = np.random.default_rng(2).choice([-1.0, 1.0], len(F))[:, None]
    out = spin.align_signs(lam * flips, F)
    assert np.allclose(out, lam) or np.allclose(out, -lam)


def test_signs_seam_stays_between_nearby_branch_points():
    """Two nearby branch points (as a folded ear makes): any tree needs a seam
    between them, but the breadth-first tree this replaced ran one across the
    surface (72 dual edges here); the best-agreeing tree keeps it short and local."""
    S, F = _sphere(4)
    c = S[F].mean(1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)

    def azimuth(p):
        e1 = np.cross(p, [0.3, 0.1, 1.0])
        e1 /= np.linalg.norm(e1)
        return np.arctan2(c @ np.cross(p, e1), c @ e1)

    p1, p2 = np.array([1.0, 0, 0]), np.array([np.cos(0.2), np.sin(0.2), 0])
    # double-valued around each point alone; a tangent-plane azimuth also winds
    # about the antipode, so there are two nearby pairs, at ±p1 and ±p2
    half = 0.5 * (azimuth(p1) - azimuth(p2))
    z = np.zeros_like(half)
    lam = qmul(np.c_[np.cos(0.4 * c[:, 2]), np.sin(0.4 * c[:, 2]), z, z],
               np.c_[np.cos(half), z, z, np.sin(half)])
    lam *= np.random.default_rng(1).choice([-1.0, 1.0], len(F))[:, None]
    out = spin.align_signs(lam, F)
    a, b = _dual_edges(F)
    seam = np.einsum("ij,ij->i", out[a], out[b]) < 0
    assert 2 <= seam.sum() <= 12, seam.sum()                      # 7
    mid = 0.5 * (c[a[seam]] + c[b[seam]])
    near = np.min([np.linalg.norm(mid - p, axis=1) for p in (p1, p2, -p1, -p2)], axis=0)
    assert near.max() < 0.3


def test_code_is_recovered_from_its_own_spinor(closed):
    """The decoder's spinor satisfies its Galerkin equation exactly at the code,
    so the corrected quotient converges to the code: ~4 orders per pass. Pass 1
    (the pointwise quotient) is off by the decoder's truncation."""
    rho, v = closed
    err = [np.linalg.norm(dq.rho_quotient(v, L, LS, iters=k) - rho) / np.linalg.norm(rho)
           for k in (1, 2, 5)]
    assert 1e-7 < err[0] < 1e-3 and err[1] < 1e-3 * err[0] and err[2] < 1e-12, err


def test_rho_quotient_pads_a_lower_band_spinor(closed):
    """A band-Lm spinor is the prefix of each spin's band-Ls modes."""
    _rho, v = closed
    Lm = L + 0.5
    n, k = len(v) // 2, len(sw.half_modes(0.5, Lm))
    vt = np.r_[v[:k], v[n:n + k]]
    vp = np.r_[v[:k], np.zeros(n - k), v[n:n + k], np.zeros(n - k)]
    assert np.allclose(dq.rho_quotient(vt, L, LS, Lm), dq.rho_quotient(vp, L, LS))


def test_codec_quotient_between_direct_and_dirac():
    """Always better than ``direct``. Against ``dirac`` it depends on shape and
    band: this bottle is 1.6x off at L=8 and 0.97x at L=16 (heads 1.0-1.1x at
    every L), because ``dirac``'s resolvent weighting has no cheap analogue."""
    from shapegen.data.dataset import demo_bottle_meshes

    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
    sph = codec.base_sphere(mesh)
    r = codec.encode(mesh, L, sphere=sph, method="quotient")
    assert codec.shape_error(mesh.vertices, codec.decode(r.coeffs, sph)) == pytest.approx(r.err, rel=1e-6)
    assert r.gamma < 1e-10
    assert r.err < 0.6 * codec.encode(mesh, L, sphere=sph, method="direct").err, r      # 0.021 vs 0.043
    fast = dq.encode_quotient(mesh.vertices, sph, mesh.faces, L, close=False)["coeffs"]
    assert np.allclose(fast[1:], r.coeffs[1:])                  # closing moves only ℓ = 0
    r16 = codec.encode(mesh, 16, sphere=sph, method="quotient")
    assert r16.err < 1.1 * codec.encode(mesh, 16, sphere=sph, method="dirac").err, r16

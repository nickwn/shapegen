"""Session 4 acceptance tests: spherical conformal param + Möbius normalization.

Maps to Appendix B rows 5 & 6:
  5. encode∘decode ≈ id on σ̂                       small
  6. Möbius gauge invariance of σ̂                  small

Plus the make-or-break rotation gauge: the encode is *exactly* rotation
equivariant (σ̂ of a rotated mesh = the Wigner-D image of σ̂), because cMCF, the
conformal Hopf field, and the least-squares SHT are all rotation covariant on
the identical (rotated) discrete mesh.
"""

import numpy as np
import pytest
import quaternionic
import spherical
from scipy.spatial import cKDTree

from bottlegen.geom import conformal as cf
from bottlegen.geom import dirac
from bottlegen.geom.mesh import Mesh, ellipsoid, icosphere
from bottlegen.geom.operators import vertex_areas
from bottlegen.spectral import swsh


def bumpy(sub=4):
    """A smooth non-spherical genus-0 test mesh (q ≠ 0)."""
    s = icosphere(sub)
    V = s.vertices
    f = 1.0 + 0.22 * (V[:, 2] ** 2 - 0.3 * V[:, 0]) + 0.10 * V[:, 0] * V[:, 1]
    return Mesh(V * f[:, None], s.faces)


def rot(ax, ang):
    ax = np.asarray(ax, float)
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def wigner_rotate(sig, R, L):
    """Apply the rotation ``R`` to spin-0 coefficients: the convention verified
    against rotating the field on the grid is ``conj(D[Dindex(ℓ,m',m)])``."""
    q = quaternionic.array.from_rotation_matrix(R)
    W = spherical.Wigner(L)
    D = W.D(q)
    out = np.zeros_like(sig)
    for l in range(2, L + 1):
        for mp in range(-l, l + 1):
            out[l, l + mp] = sum(
                np.conj(D[W.Dindex(l, mp, mm)]) * sig[l, l + mm]
                for mm in range(-l, l + 1))
    return out


# --------------------------------------------------------------------------- #
# cMCF: any genus-0 mesh -> round sphere                                       #
# --------------------------------------------------------------------------- #

def test_cmcf_ellipsoid_to_sphere():
    m = ellipsoid(4, axes=(1.0, 0.7, 0.5))
    sph = cf.spherical_conformal_map(m)
    r = np.linalg.norm(sph, axis=1)
    assert r.std() < 1e-6                       # lands exactly on the unit sphere
    assert abs(r.mean() - 1.0) < 1e-9


def test_cmcf_conformal_distortion_decreases():
    # angle distortion of the conformal map -> 0 with resolution
    def angle_distortion(m):
        sph = cf.spherical_conformal_map(m)
        F = m.faces

        def angles(X):
            a = np.zeros((len(F), 3))
            for k in range(3):
                i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
                u = X[j] - X[i]
                v = X[l] - X[i]
                u /= np.linalg.norm(u, axis=1, keepdims=True)
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                a[:, k] = np.arccos(np.clip((u * v).sum(1), -1, 1))
            return a
        return np.mean(np.abs(angles(m.vertices) - angles(sph)))

    d3 = angle_distortion(bumpy(3))
    d4 = angle_distortion(bumpy(4))
    print(f"[conformal distortion] sub3={d3:.4f} sub4={d4:.4f}")
    assert d4 < d3 and d4 < 0.05


# --------------------------------------------------------------------------- #
# Round / Möbius sphere -> σ̂ ≈ 0                                              #
# --------------------------------------------------------------------------- #

def test_round_sphere_sigma_zero():
    sig = cf.mesh_to_sigma(icosphere(4), L=8)
    assert np.max(np.abs(sig)) < 2e-3


def test_mobius_sphere_sigma_zero():
    s = icosphere(4)
    Vm = cf.ball_automorphism(s.vertices, np.array([0.0, 0.0, 0.3]))
    sig = cf.mesh_to_sigma(Mesh(Vm, s.faces), L=8)
    assert np.max(np.abs(sig)) < 5e-3


# --------------------------------------------------------------------------- #
# Test (make-or-break): rotation equivariance via the Wigner-D oracle          #
# --------------------------------------------------------------------------- #

def test_rotation_equivariance_exact():
    L = 10
    m = bumpy(4)
    R = rot([0.3, -0.7, 0.6], 1.1)
    sig = cf.mesh_to_sigma(m, L=L)
    sigR = cf.mesh_to_sigma(Mesh(m.vertices @ R.T, m.faces), L=L)
    sigD = wigner_rotate(sig, R, L)
    rel = np.linalg.norm(sigR - sigD) / np.linalg.norm(sig)
    print(f"[rotation equivariance] rel err = {rel:.2e}")
    assert rel < 1e-9                            # exact on the identical mesh


def test_rotation_perdegree_norm_invariant():
    L = 10
    m = bumpy(4)
    R = rot([1.0, 0.2, -0.5], 0.9)
    sig = cf.mesh_to_sigma(m, L=L)
    sigR = cf.mesh_to_sigma(Mesh(m.vertices @ R.T, m.faces), L=L)
    for l in range(2, L + 1):
        assert abs(np.linalg.norm(sig[l]) - np.linalg.norm(sigR[l])) < 1e-8


# --------------------------------------------------------------------------- #
# Test 6: Möbius gauge invariance of σ̂                                        #
# --------------------------------------------------------------------------- #

def test_mobius_gauge_invariance():
    L = 12
    m = bumpy(4)
    sph = cf.spherical_conformal_map(m)

    def sig_from(sphp):
        q, th, ph, ar = cf.hopf_on_sphere(m, cf.mobius_normalize(m, sphp))
        qh = swsh.fit_coeffs(2, L, th, ph, q, weights=ar)
        sc = swsh.hopf_scaling(L)
        sg = swsh.empty_coeffs(L)
        for l in range(2, L + 1):
            sg[l] = qh[l] / sc[l]
        return cf.gauge_fix(sg)

    sig0 = sig_from(sph)
    sigA = sig_from(cf.ball_automorphism(sph, np.array([0.18, -0.1, 0.05])))
    rel = np.linalg.norm(sigA - sig0) / np.linalg.norm(sig0)
    print(f"[möbius gauge] σ̂ rel diff = {rel:.4f}")
    assert rel < 2e-2


def test_mobius_normalize_centers_barycenter():
    m = bumpy(4)
    sph = cf.spherical_conformal_map(m)
    nrm = cf.mobius_normalize(m, sph)
    w = vertex_areas(m.vertices, m.faces)
    w = w / w.sum()
    assert np.linalg.norm((w[:, None] * nrm).sum(0)) < 1e-6


# --------------------------------------------------------------------------- #
# Test 5: encode∘decode ≈ id (approximate inverse; the hard cross-session one) #
# --------------------------------------------------------------------------- #

def test_encode_decode_roundtrip():
    L = 10
    m = bumpy(4)
    sig = cf.mesh_to_sigma(m, L=L)
    base = icosphere(4)
    V, F = dirac.reconstruct(base.vertices, base.faces, sig)   # linear init
    sig2 = cf.mesh_to_sigma(Mesh(V, F), L=L)
    spec = np.linalg.norm(sig2 - sig) / np.linalg.norm(sig)
    print(f"[encode∘decode] spectral rel err = {spec:.3f}")
    assert spec < 0.45
    # the decoded shape is non-trivial and the right amplitude band
    rstd = np.std(np.linalg.norm(V - V.mean(0), axis=1))
    assert 0.3 < rstd / np.std(np.linalg.norm(m.vertices - m.vertices.mean(0),
                                              axis=1)) < 3.0


def test_decode_chamfer_matches_source():
    L = 10
    m = bumpy(4)
    sig = cf.mesh_to_sigma(m, L=L)
    base = icosphere(4)
    V, _ = dirac.reconstruct(base.vertices, base.faces, sig)
    # similarity-align decode to source, then Chamfer
    X0 = V - V.mean(0)
    Y0 = m.vertices - m.vertices.mean(0)
    U, S, Vt = np.linalg.svd(X0.T @ Y0)
    Va = (S.sum() / (X0 ** 2).sum()) * (X0 @ (U @ Vt))
    da, _ = cKDTree(Y0).query(Va)
    db, _ = cKDTree(Va).query(Y0)
    chamfer = 0.5 * (da.mean() + db.mean())
    feat = np.std(np.linalg.norm(Y0, axis=1))
    print(f"[decode chamfer] {chamfer:.4f} (feature scale {feat:.4f})")
    assert chamfer < feat

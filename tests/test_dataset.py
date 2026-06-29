"""Session 6 acceptance tests: dataset + Wigner-D rotation augmentation.

Maps to Appendix B row 9:
  9. rotation aug preserves per-ℓ ‖σ̂‖             1e-9

Also: Wigner-D verified against the grid-rotation ground truth and the
``spherical`` oracle; rotation composition; decode equivariance (an augmented
σ̂ decodes to a rotated bottle); deterministic seeding.
"""

import numpy as np
import pytest
import quaternionic
import spherical

from bottlegen.data import classprior, dataset
from bottlegen.data.dataset import random_so3, wigner_rotate_sigma
from bottlegen.flow import state
from bottlegen.geom import conformal as cf
from bottlegen.geom import dirac
from bottlegen.geom.mesh import Mesh, icosphere
from bottlegen.spectral import swsh

L = 10


def _rand_sigma(seed=0, Lm=L):
    rng = np.random.default_rng(seed)
    C = swsh.empty_coeffs(Lm)
    for l in range(2, Lm + 1):
        for m in range(-l, l + 1):
            C[l, l + m] = (rng.standard_normal() + 1j * rng.standard_normal()) / l
    return C


def _rot(ax, ang):
    ax = np.asarray(ax, float)
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


# --------------------------------------------------------------------------- #
# Test 9: rotation preserves per-degree norm (Wigner-D is unitary)            #
# --------------------------------------------------------------------------- #

def test_rotation_preserves_perdegree_norm():
    C = _rand_sigma(1)
    R = random_so3(np.random.default_rng(7))
    Cr = wigner_rotate_sigma(C, R)
    for l in range(2, L + 1):
        assert abs(np.linalg.norm(Cr[l]) - np.linalg.norm(C[l])) < 1e-9


def test_identity_rotation_is_identity():
    C = _rand_sigma(2)
    assert np.max(np.abs(wigner_rotate_sigma(C, np.eye(3)) - C)) < 1e-12


def test_rotation_composition():
    C = _rand_sigma(3)
    R1 = _rot([0.2, 1.0, -0.3], 0.6)
    R2 = _rot([-0.7, 0.4, 0.5], 1.1)
    lhs = wigner_rotate_sigma(wigner_rotate_sigma(C, R1), R2)
    rhs = wigner_rotate_sigma(C, R2 @ R1)
    assert np.max(np.abs(lhs - rhs)) < 1e-9


def test_wigner_matches_spherical_oracle():
    C = _rand_sigma(4)
    R = _rot([0.3, -0.6, 0.8], 0.9)
    mine = wigner_rotate_sigma(C, R)
    # spherical oracle: a'_{ℓm'} = Σ_m conj(D[ℓ,m',m]) a_{ℓm}
    q = quaternionic.array.from_rotation_matrix(R)
    W = spherical.Wigner(L)
    D = W.D(q)
    ref = swsh.empty_coeffs(L)
    for l in range(2, L + 1):
        for mp in range(-l, l + 1):
            ref[l, l + mp] = sum(np.conj(D[W.Dindex(l, mp, mm)]) * C[l, l + mm]
                                 for mm in range(-l, l + 1))
    assert np.max(np.abs(mine - ref)) < 1e-9


def test_random_so3_is_a_rotation():
    rng = np.random.default_rng(0)
    for _ in range(5):
        R = random_so3(rng)
        assert np.max(np.abs(R @ R.T - np.eye(3))) < 1e-12
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# Augmented σ̂ decodes to a rotated bottle (decode equivariance)              #
# --------------------------------------------------------------------------- #

def test_augmented_sigma_decodes_to_rotated_bottle():
    # a realistic in-class σ̂ from a bumpy mesh
    s = icosphere(4)
    V = s.vertices
    f = 1.0 + 0.2 * (V[:, 2] ** 2 - 0.3 * V[:, 0]) + 0.08 * V[:, 0] * V[:, 1]
    sig = cf.mesh_to_sigma(Mesh(V * f[:, None], s.faces), L=L)
    R = _rot([0.4, 0.5, -0.7], 0.8)
    base = icosphere(4)
    V0, _ = dirac.reconstruct(base.vertices, base.faces, sig)
    V1, _ = dirac.reconstruct(base.vertices, base.faces,
                              wigner_rotate_sigma(sig, R))
    # The continuous decode is rotation-equivariant; the discrete decode on a
    # fixed icosphere is so only approximately (the base mesh carries only
    # icosahedral symmetry). After similarity alignment the rotated decode
    # recovers the original shape (residual ≈ 0.14 here) far better than the
    # un-aligned R·V0 baseline (≈ 0.66) — i.e. it is recognizably the rotated
    # bottle. The exact statement of "rotation is free augmentation" is the
    # per-degree-norm preservation tested above.
    A = V0 - V0.mean(0)
    B = V1 - V1.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    Om = U @ Vt
    sc = S.sum() / (A ** 2).sum()
    resid = np.linalg.norm(B - sc * (A @ Om)) / np.linalg.norm(B)
    baseline = np.linalg.norm(B - A @ R.T) / np.linalg.norm(B)
    print(f"[decode equivariance] aligned={resid:.3f} baseline={baseline:.3f}")
    assert resid < 0.2 and resid < 0.5 * baseline


# --------------------------------------------------------------------------- #
# Dataset object (NumPy core; torch path gated)                               #
# --------------------------------------------------------------------------- #

def test_dataset_inmemory_stats_and_len():
    items = [state.gauge_fix(_rand_sigma(i)) for i in range(6)]
    ds = dataset.BottleSigma(items=items, L=L, rotate=True, seed=0)
    assert len(ds) == 6
    assert ds.stats.shape == (L + 1,)
    assert (ds.stats[2:] > 0).all()


def test_dataset_getitem_and_determinism():
    torch = pytest.importorskip("torch")
    items = [state.gauge_fix(_rand_sigma(i)) for i in range(4)]
    ds1 = dataset.BottleSigma(items=items, L=L, rotate=True, seed=123)
    ds2 = dataset.BottleSigma(items=items, L=L, rotate=True, seed=123)
    x1 = ds1[0]
    x2 = ds2[0]
    assert x1.shape[0] == state.flat_dim(L)
    assert torch.allclose(x1, x2)              # deterministic under seed


def test_dataset_no_rotation_roundtrips():
    torch = pytest.importorskip("torch")
    item = state.gauge_fix(_rand_sigma(0))
    ds = dataset.BottleSigma(items=[item], L=L, rotate=False)
    v = ds[0].numpy()
    assert np.max(np.abs(state.real_unpack(v, L) - item)) < 1e-6


# --------------------------------------------------------------------------- #
# Class prior: per-degree power invariant under rotation augmentation          #
# --------------------------------------------------------------------------- #

def test_classprior_power_spectrum_rotation_invariant():
    stack = np.stack([state.gauge_fix(_rand_sigma(i)) for i in range(8)])
    cp = classprior.ClassPrior(stack)
    base = cp.power_spectrum
    aug = cp.rotation_augmented_spectrum(n_rot=6, rng=np.random.default_rng(0))
    # rotation only mixes orders within a degree -> per-degree power unchanged
    rel = np.max(np.abs(aug[2:] - base[2:]) / (base[2:] + 1e-12))
    print(f"[classprior] max per-degree power drift = {rel:.2e}")
    assert rel < 1e-9

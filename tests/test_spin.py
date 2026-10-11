"""Acceptance tests for the corrected Dirac operator and the ρ encode.

The gate: ``mesh → λ → ρ → mesh`` must round trip, and the Dirac residual
``‖Dλ − ρλ‖/‖Dλ‖`` must fall under refinement. The previous mass-coupled
operator failed the residual test at ~0.93 regardless of resolution, because
its ``ρ`` was a 1/length² quantity rather than a curvature.
"""

import numpy as np
import pytest

from shapegen.data.dataset import demo_bottle_meshes
from shapegen import codec
from shapegen.geom import dirac, spin
from shapegen.geom.mesh import icosphere


def _unit_sphere(sub):
    m = icosphere(sub)
    return m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True), m.faces


def _procrustes_err(A, B):
    """RMS error after optimal rotation/scale/translation of B onto A."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    Br = B @ R
    s = (A * Br).sum() / (Br * Br).sum()
    return float(np.sqrt(((A - s * Br) ** 2).sum(1).mean())
                 / np.sqrt((A ** 2).sum(1).mean()))


def _smooth_rho(S, seed=0, lmax=6, amp=2.0):
    from scipy.special import sph_harm_y
    th = np.arccos(np.clip(S[:, 2], -1, 1))
    ph = np.arctan2(S[:, 1], S[:, 0])
    rng = np.random.default_rng(seed)
    rho = np.zeros(len(S))
    for l in range(2, lmax + 1):
        for m in range(-l, l + 1):
            rho += rng.standard_normal() * np.real(sph_harm_y(l, m, th, ph)) / l ** 2
    return rho * (amp / np.abs(rho).max())


# --------------------------------------------------------------------------- #
# Operator structure                                                          #
# --------------------------------------------------------------------------- #

def test_operator_is_psd_and_reduces_to_X():
    """A(ρ) is PSD (it is a squared residual) and A(0) == X."""
    S, F = _unit_sphere(2)
    A0, _ = dirac.dirac_operator(S, F, np.zeros(len(S)))
    X = dirac.dirac_assembly(S, F)
    assert abs(A0 - X).max() < 1e-12
    A, _ = dirac.dirac_operator(S, F, _smooth_rho(S))
    assert np.linalg.eigvalsh(A.toarray()).min() > -1e-9


def test_dirac_annihilates_constants():
    """(Dλ)_f = 0 for λ ≡ 1, since the opposite edges of a face sum to zero."""
    S, F = _unit_sphere(3)
    lam = np.tile([1.0, 0, 0, 0], (len(S), 1))
    assert np.abs(dirac.apply_dirac(S, F, lam)).max() < 1e-12


def test_rho_is_a_curvature():
    """Linear response: a radial mode ĥY_{ℓm} has ρ = ½ℓ(ℓ+1)ĥ.

    This is the units check the old operator failed — it was off by exactly the
    Laplacian eigenvalue ℓ(ℓ+1).
    """
    from scipy.special import sph_harm_y
    S, F = _unit_sphere(4)
    th = np.arccos(np.clip(S[:, 2], -1, 1))
    ph = np.arctan2(S[:, 1], S[:, 0])
    eps = 1e-3
    for l in (2, 3, 4):
        Y = np.real(sph_harm_y(l, 0, th, ph))
        lam, _ = dirac.spin_transform(S, F, eps * Y)
        r = np.linalg.norm(dirac.integrate_edges(S, F, lam), axis=1)
        r = r / r.mean()
        gain = ((r - 1.0) * Y).sum() / (Y * Y).sum() / eps
        assert 1.0 / gain == pytest.approx(0.5 * l * (l + 1), rel=0.05)


# --------------------------------------------------------------------------- #
# Round trip                                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sub,tol", ((3, 1e-2), (4, 3e-3)))
def test_rho_roundtrip_on_sphere(sub, tol):
    """ρ → mesh → λ → ρ' → mesh': the two meshes agree, converging in h."""
    S, F = _unit_sphere(sub)
    rho = _smooth_rho(S)
    lam, _ = dirac.spin_transform(S, F, rho)
    V1 = dirac.integrate_edges(S, F, lam)
    rho2 = dirac.rho_from_lambda(S, F, spin.lambda_from_correspondence(S, V1, F))
    V2 = dirac.integrate_edges(S, F, dirac.spin_transform(S, F, rho2)[0])
    err = _procrustes_err(V1, V2)
    print(f"[rho round trip sub={sub}] n={len(S)} err={err:.2e}")
    assert err < tol


def test_lambda_recovery_converges():
    """λ recovered from the correspondence reproduces the mesh it came from,
    at second order — halving h must cut the error by at least 2×."""
    errs = []
    for sub in (2, 3, 4):
        S, F = _unit_sphere(sub)
        lam, _ = dirac.spin_transform(S, F, _smooth_rho(S))
        V = dirac.integrate_edges(S, F, lam)
        lam_r = spin.lambda_from_correspondence(S, V, F)
        errs.append(_procrustes_err(V, dirac.integrate_edges(S, F, lam_r)))
    print(f"[lambda recovery] errs={['%.2e' % e for e in errs]}")
    assert all(b < a / 2.0 for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 3e-3


def test_dirac_residual_converges_on_real_meshes():
    """The gate the old operator failed.

    For a bottle's conformal parameterization the residual must fall with refinement
    (it is bounded by the map's quasi-conformal distortion, which halves per
    subdivision). The old mass-coupled operator sat at ~0.93 at every level.
    """
    prev_res, prev_qc = np.inf, np.inf
    for sub in (2, 3):
        mesh = demo_bottle_meshes(n=1, subdiv=sub, seed=0)[0]
        F = mesh.faces
        sph = codec.base_sphere(mesh)
        lam = spin.lambda_from_correspondence(sph, mesh.vertices, F)
        rho = dirac.rho_from_lambda(sph, F, lam)
        res = dirac.dirac_residual(sph, F, lam, rho)
        qc = spin.qc_distortion(sph, mesh.vertices, F)
        print(f"[bottle sub={sub}] n={len(mesh.vertices)} resid={res:.4f} qc={qc:.4f}")
        assert res < prev_res and qc < prev_qc
        assert res < 0.5                      # old operator: ~0.93 at every level
        prev_res, prev_qc = res, qc



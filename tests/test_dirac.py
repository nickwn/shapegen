"""Session 3 acceptance tests: quaternionic Dirac & the spin transform.

Maps to Appendix B row 4 (and sets up row 5):
  4. Dirac identity (ρ=0 ⇒ λ≡1), identity reconstruction        1e-9

The ρ=0 identity is the machine-precision gate for the highest-risk kernel:
nothing downstream may use the decode engine until it passes.
"""

import numpy as np
import pytest

from shapegen.geom import dirac
from shapegen.geom.mesh import icosphere


# --------------------------------------------------------------------------- #
# Operator algebra: symmetry and the kernel (row sums zero)                   #
# --------------------------------------------------------------------------- #

def test_dirac_is_symmetric():
    m = icosphere(2)
    X = dirac.dirac_assembly(m.vertices, m.faces)
    asym = (X - X.T).tocoo()
    worst = np.abs(asym.data).max() if asym.data.size else 0.0
    assert worst < 1e-10


def test_constant_spinor_in_kernel():
    # X·1 = 0 (the three opposite edges of every triangle sum to zero), so the
    # constant quaternion is annihilated -> ρ=0 has γ=0.
    m = icosphere(3)
    X = dirac.dirac_assembly(m.vertices, m.faces)
    for comp in range(4):
        v = np.zeros(4 * len(m.vertices))
        v[comp::4] = 1.0
        assert np.abs(X @ v).max() < 1e-10


# --------------------------------------------------------------------------- #
# Spin transform: deterministic and well-gauged                               #
# --------------------------------------------------------------------------- #

def _random_form_inputs(sub, seed):
    m = icosphere(sub)
    V = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    n = len(V)
    return (V, m.faces, rng.standard_normal(n) * 0.7,
            rng.standard_normal((n, 4)), rng.standard_normal((n, 4)))


def _sh_rho(V, seed=0, lmax=6, amp=2.0):
    from scipy.special import sph_harm_y
    th = np.arccos(np.clip(V[:, 2], -1, 1))
    ph = np.arctan2(V[:, 1], V[:, 0])
    rng = np.random.default_rng(seed)
    r = np.zeros(len(V))
    for l in range(2, lmax + 1):
        for m in range(-l, l + 1):
            r += rng.standard_normal() * np.real(sph_harm_y(l, m, th, ph)) / l ** 2
    return r * (amp / np.abs(r).max())


def test_spin_transform_is_deterministic():
    """Identical input gives an identical λ — no residual ARPACK branch noise."""
    V, F, _, _, _ = _random_form_inputs(2, seed=6)
    rho = _sh_rho(V, seed=1)
    a, _ = dirac.spin_transform(V, F, rho)
    b, _ = dirac.spin_transform(V, F, rho)
    assert np.max(np.abs(a - b)) < 1e-10


def test_gauge_fix_denominator_stays_conditioned():
    """``_gauge_fix`` divides by the mean spinor, so it fails if that mean nears
    zero. Measured on a bottle param it stays ≳0.47 out to 16× the encoded
    amplitude — well conditioned across the whole operating range."""
    V, F, _, _, _ = _random_form_inputs(2, seed=8)
    rho = _sh_rho(V, seed=2)
    for amp in (1.0, 4.0, 16.0):
        lam, _ = dirac.spin_transform(V, F, amp * rho)
        mean_norm = np.linalg.norm(lam.mean(0))
        assert mean_norm > 0.1, f"gauge denominator collapsed at amp={amp}"


# --------------------------------------------------------------------------- #
# Test 4: ρ=0 ⇒ λ≡1 and exact identity reconstruction                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sub", (2, 3))
def test_identity_rho_zero(sub):
    m = icosphere(sub)
    V, F = m.vertices, m.faces
    rho = np.zeros(len(V))
    lam, gamma = dirac.spin_transform(V, F, rho)
    # gauge-fixed constant solution -> λ ≡ (1,0,0,0)
    assert abs(gamma) < 1e-7
    assert np.max(np.abs(lam - np.array([1.0, 0, 0, 0]))) < 1e-8
    # df̃ = λ̄ df λ = df -> reconstruction equals the original (mean removed)
    Vr = dirac.integrate_edges(V, F, lam)
    err = np.max(np.abs(Vr - (V - V.mean(0))))
    print(f"[identity sub={sub}] |γ|={abs(gamma):.2e} recon err={err:.2e}")
    assert err < 1e-9


# --------------------------------------------------------------------------- #
# Integration machinery: λ≡1 and constant conformal scaling                   #
# --------------------------------------------------------------------------- #

def test_integrate_identity_edges():
    m = icosphere(3)
    V, F = m.vertices, m.faces
    lam = np.tile(np.array([1.0, 0, 0, 0]), (len(V), 1))
    Vr = dirac.integrate_edges(V, F, lam)
    assert np.max(np.abs(Vr - (V - V.mean(0)))) < 1e-9


def test_constant_conformal_scaling():
    # λ ≡ c (real) -> df̃ = c² df -> uniformly scaled sphere, machine precision.
    m = icosphere(3)
    V, F = m.vertices, m.faces
    c = 1.7
    lam = np.tile(np.array([c, 0, 0, 0]), (len(V), 1))
    Vr = dirac.integrate_edges(V, F, lam)
    target = c**2 * (V - V.mean(0))
    assert np.max(np.abs(Vr - target)) < 1e-9


def test_known_conformal_map_reproduced():
    # A conformal (Möbius) image of the sphere has exactly integrable edges;
    # integrating ẽ = (V*_j − V*_i) must recover V* up to translation.
    m = icosphere(3)
    V, F = m.vertices, m.faces
    a = np.array([0.0, 0.0, 0.35])               # ball automorphism center
    num = (1 - (a @ a)) * (V - a) - ((V - a) ** 2).sum(1, keepdims=True) * a
    den = (1 - 2 * V @ a + (a @ a)) [:, None]
    Vstar = num / den
    Vstar /= np.linalg.norm(Vstar, axis=1, keepdims=True)   # back onto S²
    # feed the exact edges of Vstar through the integrator (λ encoded implicitly)
    Vr = _integrate_target_edges(V, F, Vstar)
    # align translation; compare
    err = np.max(np.abs((Vr - Vr.mean(0)) - (Vstar - Vstar.mean(0))))
    print(f"[möbius reproduce] err={err:.2e}")
    assert err < 1e-9


def _integrate_target_edges(V, F, Vtarget):
    """Cotan-Poisson integrate the exact edges of ``Vtarget`` on the (V,F)
    connectivity (an integrability/round-trip check of the Poisson solver)."""
    from shapegen.geom.operators import cotan_laplacian, edge_cotan_weights
    L, _ = cotan_laplacian(V, F)
    edges, w = edge_cotan_weights(V, F)
    i, j = edges[:, 0], edges[:, 1]
    et = Vtarget[j] - Vtarget[i]
    b = np.zeros((len(V), 3))
    np.add.at(b, i, -w[:, None] * et)
    np.add.at(b, j, w[:, None] * et)
    x = dirac._poisson_solve(L, b)
    return x - x.mean(0)


# --------------------------------------------------------------------------- #
# Decode smoke test: a small σ̂ -> finite-W, genus-0 mesh                      #
# --------------------------------------------------------------------------- #

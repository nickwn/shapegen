"""Session 3 acceptance tests: quaternionic Dirac & the spin transform.

Maps to Appendix B row 4 (and sets up row 5):
  4. Dirac identity (ρ=0 ⇒ λ≡1), identity reconstruction        1e-9

The ρ=0 identity is the machine-precision gate for the highest-risk kernel:
nothing downstream may use the decode engine until it passes.
"""

import numpy as np
import pytest

from bottlegen.geom import dirac
from bottlegen.geom.curvature import mean_curvature
from bottlegen.geom.mesh import icosphere
from bottlegen.geom.operators import vertex_areas


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
# ∂A/∂ρ — the adjoint's one ρ-dependent term                                  #
# --------------------------------------------------------------------------- #

def _random_form_inputs(sub, seed):
    m = icosphere(sub)
    V = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    n = len(V)
    return (V, m.faces, rng.standard_normal(n) * 0.7,
            rng.standard_normal((n, 4)), rng.standard_normal((n, 4)))


def _form(V, F, rho, a, b):
    A, _ = dirac.dirac_operator(V, F, rho)
    return float(a.ravel() @ (A @ b.ravel()))


@pytest.mark.parametrize("sub", (1, 2))
def test_dirac_grad_rho_matches_fd(sub):
    """``dirac_grad_rho`` is the true ∂/∂ρ of ⟨a, A(ρ)b⟩, magnitude included.

    The **ratio** is the assertion that matters. A wrong chain factor on the
    ρ-coupling leaves the direction exact (cosine 1.0) and scales the whole
    gradient, so a cosine-only check passes a gradient that is a constant factor
    off — the failure mode this term is prone to.
    """
    V, F, rho, a, b = _random_form_inputs(sub, seed=1)
    g = dirac.dirac_grad_rho(V, F, rho, a, b)

    h = 1e-6
    gfd = np.empty(len(V))
    for v in range(len(V)):
        rp, rm = rho.copy(), rho.copy()
        rp[v] += h
        rm[v] -= h
        gfd[v] = (_form(V, F, rp, a, b) - _form(V, F, rm, a, b)) / (2.0 * h)

    assert np.linalg.norm(g - gfd) / np.linalg.norm(gfd) < 1e-6
    assert float(g @ gfd / (gfd @ gfd)) == pytest.approx(1.0, abs=1e-6)


def test_dirac_grad_rho_is_quadratic_in_rho():
    """``A`` is quadratic in ρ, so the gradient is affine: the second difference
    in ρ vanishes and the ρ=0 gradient is the pure linear-stencil term."""
    V, F, rho, a, b = _random_form_inputs(2, seed=2)
    g0 = dirac.dirac_grad_rho(V, F, np.zeros(len(V)), a, b)
    g1 = dirac.dirac_grad_rho(V, F, rho, a, b)
    g2 = dirac.dirac_grad_rho(V, F, 2.0 * rho, a, b)
    assert np.abs(g2 - 2.0 * g1 + g0).max() < 1e-10 * max(np.abs(g1).max(), 1.0)


def test_dirac_grad_rho_diagonal_has_no_linear_term():
    """On ``p = q`` the two linear contributions cancel, leaving only ``2A_fρ_p/9``.

    At ρ=0 that term is zero, so a spinor supported on one vertex — which sees
    only the diagonal block — must have zero gradient there.
    """
    V, F, _, _, _ = _random_form_inputs(2, seed=3)
    n = len(V)
    a = np.zeros((n, 4))
    b = np.zeros((n, 4))
    a[0] = b[0] = [1.0, 0.0, 0.0, 0.0]
    g = dirac.dirac_grad_rho(V, F, np.zeros(n), a, b)
    assert np.abs(g[0]) < 1e-12


# --------------------------------------------------------------------------- #
# ρ ↦ λ is smooth — the differentiability the adjoint rests on                #
# --------------------------------------------------------------------------- #

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


def test_spin_transform_is_differentiable_in_rho():
    """``‖λ(ρ+hδ) − λ(ρ)‖ / h`` is constant as h → 0.

    ``spla.eigsh`` returns an arbitrary member of the 4-dimensional eigenspace
    (the right-ℍ action), and that choice is *not* continuous in ρ. What makes
    the map differentiable anyway is :func:`~bottlegen.geom.dirac._gauge_fix`:
    dividing by the mean spinor is a canonical section, so the arbitrary factor
    cancels before anything downstream sees it.

    The adjoint of the eigensolve differentiates through this map, so if a future
    change to ``spin_transform`` drops or weakens the gauge fix, the gradient goes
    silently wrong rather than loudly wrong. This is that alarm.
    """
    V, F, _, _, _ = _random_form_inputs(2, seed=5)
    rho = _sh_rho(V)
    rng = np.random.default_rng(7)
    d = rng.standard_normal(len(V))
    d /= np.linalg.norm(d)

    lam0, _ = dirac.spin_transform(V, F, rho)
    slopes = []
    for h in (1e-4, 1e-5, 1e-6):
        lam, _ = dirac.spin_transform(V, F, rho + h * d)
        slopes.append(np.linalg.norm(lam - lam0) / h)
    print(f"[smoothness] slopes={['%.6f' % s for s in slopes]}")
    assert slopes[0] == pytest.approx(slopes[-1], rel=1e-3)
    assert slopes[1] == pytest.approx(slopes[-1], rel=1e-3)


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
    from bottlegen.geom.operators import cotan_laplacian, edge_cotan_weights
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

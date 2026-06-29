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

def test_reconstruct_smoke():
    from bottlegen.spectral import swsh
    base = icosphere(3)
    L = 8
    sigma = swsh.empty_coeffs(L)
    sigma[2, 2 + 0] = 0.05            # a single low-ℓ, real (Hermitian-safe) mode
    sigma[2, 2 - 0] = 0.05
    V, F = dirac.reconstruct(base.vertices, base.faces, sigma)
    assert np.isfinite(V).all()
    # genus unchanged (same connectivity), finite Willmore energy
    H, _, _ = mean_curvature(V, F)
    A = vertex_areas(V, F)
    W = float((H**2 * A).sum())
    print(f"[reconstruct smoke] W={W:.3f} (sphere=4π≈{4*np.pi:.3f})")
    assert np.isfinite(W) and W > 0
    # the shape actually moved away from the round sphere
    assert np.std(np.linalg.norm(V, axis=1)) > 1e-4

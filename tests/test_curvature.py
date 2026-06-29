"""Session 2 acceptance tests: discrete DG, curvature, Hopf.

Maps to Appendix B row 3:
  3. sphere H=K=1, q=0; Gauss–Bonnet                 →0 / 1e-6

Closed-form curvature of an ellipsoid (implicit surface F = Σ(x_i/a_i)² − 1)
is used as the analytic oracle for H and K.
"""

import numpy as np
import pytest

from bottlegen.geom import curvature as cv
from bottlegen.geom.mesh import ellipsoid, icosphere
from bottlegen.geom.operators import cotan_laplacian, vertex_areas


# --------------------------------------------------------------------------- #
# Closed-form ellipsoid curvature (implicit-surface formulas)                 #
# --------------------------------------------------------------------------- #

def ellipsoid_HK(P, axes):
    """Exact (H, K) at points P on Σ(x_i/a_i)²=1, outward/convex-positive."""
    a, b, c = axes
    g = 2.0 * np.stack([P[:, 0] / a**2, P[:, 1] / b**2, P[:, 2] / c**2], axis=1)
    gn = np.linalg.norm(g, axis=1)
    hess = np.diag([2.0 / a**2, 2.0 / b**2, 2.0 / c**2])
    gHg = (g @ hess * g).sum(1)
    trH = np.trace(hess)
    # mean curvature (negate so the outward normal gives +1/R on a sphere)
    H = -(gHg - gn**2 * trH) / (2.0 * gn**3)
    adj = np.diag([
        (2.0 / b**2) * (2.0 / c**2),
        (2.0 / a**2) * (2.0 / c**2),
        (2.0 / a**2) * (2.0 / b**2),
    ])
    K = (g @ adj * g).sum(1) / gn**4
    return H, K


# --------------------------------------------------------------------------- #
# Test 3a: unit sphere — H→1, K→1, q→0; convergence with resolution           #
# --------------------------------------------------------------------------- #

def test_sphere_curvature_converges():
    prev = None
    for sub in (2, 3, 4):
        m = icosphere(sub)
        V, F = m.vertices, m.faces
        L, M = cotan_laplacian(V, F)
        H, _, _ = cv.mean_curvature(V, F, L, M)
        K = cv.gaussian_curvature(V, F, M)
        q = cv.hopf_q(V, F)
        # max errors are pinned by the 12 valence-5 vertices (a fixed icosphere
        # artifact); the median is the convergence-tracking quantity.
        eH = np.median(np.abs(H - 1.0))
        eK = np.median(np.abs(K - 1.0))
        eq = np.median(np.abs(q))
        print(f"[sphere sub={sub}] med|H-1|={eH:.3e} med|K-1|={eK:.3e} "
              f"med|q|={eq:.3e} max|H-1|={np.max(np.abs(H - 1.0)):.3e}")
        if prev is not None:
            assert eH < prev[0]               # median H error decreases
        prev = (eH, eK, eq)
    assert eH < 2e-2 and eK < 2e-2 and eq < 2e-2


def test_sphere_radius_scaling():
    # sphere of radius R: H = 1/R, K = 1/R²
    R = 2.5
    m = icosphere(4)
    V = m.vertices * R
    L, M = cotan_laplacian(V, m.faces)
    H, _, _ = cv.mean_curvature(V, m.faces, L, M)
    K = cv.gaussian_curvature(V, m.faces, M)
    assert np.median(np.abs(H - 1.0 / R)) < 2e-2
    assert np.median(np.abs(K - 1.0 / R**2)) < 2e-2


# --------------------------------------------------------------------------- #
# Test 3b: ellipsoid vs closed-form H, K                                      #
# --------------------------------------------------------------------------- #

def test_ellipsoid_matches_closed_form():
    axes = (1.0, 0.8, 0.6)
    m = ellipsoid(5, axes=axes)
    V, F = m.vertices, m.faces
    L, M = cotan_laplacian(V, F)
    H, _, _ = cv.mean_curvature(V, F, L, M)
    K = cv.gaussian_curvature(V, F, M)
    Hc, Kc = ellipsoid_HK(V, axes)
    relH = np.median(np.abs(H - Hc) / np.abs(Hc))
    relK = np.median(np.abs(K - Kc) / np.maximum(np.abs(Kc), 1e-6))
    print(f"[ellipsoid] median relH={relH:.3e} relK={relK:.3e}")
    assert relH < 0.05
    assert relK < 0.10


def test_ellipsoid_principal_matches_HK():
    # principal-curvature fit reproduces closed-form H,K (less robust than the
    # Laplacian/angle-defect estimators, so a looser tolerance)
    axes = (1.0, 0.8, 0.6)
    m = ellipsoid(5, axes=axes)
    V, F = m.vertices, m.faces
    k1, k2 = cv.principal_curvatures(V, F)
    Hfit = 0.5 * (k1 + k2)
    Hc, _ = ellipsoid_HK(V, axes)
    assert np.median(np.abs(Hfit - Hc) / np.abs(Hc)) < 0.08


# --------------------------------------------------------------------------- #
# Test 3c: Gauss–Bonnet  Σ K_i A_i = 2πχ  (exact for angle defect)            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sub", (2, 3, 4))
def test_gauss_bonnet(sub):
    m = icosphere(sub)
    V, F = m.vertices, m.faces
    A = vertex_areas(V, F)
    K = cv.gaussian_curvature(V, F)
    total = float((K * A).sum())
    chi = m.euler_number
    assert abs(total - 2.0 * np.pi * chi) < 1e-6
    assert chi == 2 and m.genus == 0


def test_gauss_bonnet_ellipsoid():
    m = ellipsoid(4, axes=(1.0, 0.7, 0.5))
    V, F = m.vertices, m.faces
    total = float((cv.gaussian_curvature(V, F) * vertex_areas(V, F)).sum())
    assert abs(total - 2.0 * np.pi * m.euler_number) < 1e-6


# --------------------------------------------------------------------------- #
# Test 3d: Hopf q vanishes at umbilics (sphere) and is large where κ₁≠κ₂      #
# --------------------------------------------------------------------------- #

def test_hopf_zero_at_umbilics_sphere():
    m = icosphere(4)
    q = cv.hopf_q(m.vertices, m.faces)
    assert np.max(np.abs(q)) < 5e-2


def test_hopf_magnitude_is_half_curvature_gap():
    # |q| = ½|κ1 − κ2| on the ellipsoid (frame-independent magnitude)
    axes = (1.0, 0.7, 0.5)
    m = ellipsoid(5, axes=axes)
    V, F = m.vertices, m.faces
    q = cv.hopf_q(V, F)
    k1, k2 = cv.principal_curvatures(V, F)
    lhs = np.abs(q)
    rhs = 0.5 * np.abs(k1 - k2)
    # compare away from near-umbilic vertices where the fit is noisiest
    mask = rhs > 0.2 * rhs.max()
    rel = np.median(np.abs(lhs[mask] - rhs[mask]) / rhs[mask])
    print(f"[|q| vs ½|k1-k2|] median rel err = {rel:.3e}")
    assert rel < 0.10


def test_mean_curvature_outward_positive():
    # convex shapes -> H > 0 everywhere with the outward-normal convention
    m = ellipsoid(4, axes=(1.0, 0.8, 0.6))
    H, _, _ = cv.mean_curvature(m.vertices, m.faces)
    assert (H > 0).all()

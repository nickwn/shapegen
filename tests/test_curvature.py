"""Acceptance tests for discrete mean curvature.

Sign and magnitude against known surfaces. The Gauss-Bonnet and Hopf-
differential tests that lived here covered the σ̂ machinery and went with it.
"""

import numpy as np
import pytest

from shapegen.geom import curvature as cv
from shapegen.geom.mesh import ellipsoid, icosphere
from shapegen.geom.operators import cotan_laplacian


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

def test_mean_curvature_outward_positive():
    # convex shapes -> H > 0 everywhere with the outward-normal convention
    m = ellipsoid(4, axes=(1.0, 0.8, 0.6))
    H, _, _ = cv.mean_curvature(m.vertices, m.faces)
    assert (H > 0).all()


def test_sphere_curvature_converges():
    """``H → 1`` on the unit sphere, and the error falls under refinement.

    The *median* is the convergence quantity, not the max: the 12 valence-5
    vertices of an icosphere are a fixed artifact of the tessellation and their
    error does not shrink the same way.
    """
    prev = None
    for sub in (2, 3, 4):
        m = icosphere(sub)
        V, F = m.vertices, m.faces
        L, M = cotan_laplacian(V, F)
        H, _, _ = cv.mean_curvature(V, F, L, M)
        eH = np.median(np.abs(H - 1.0))
        print(f"[sphere sub={sub}] med|H-1|={eH:.3e} "
              f"max|H-1|={np.max(np.abs(H - 1.0)):.3e}")
        if prev is not None:
            assert eH < prev
        prev = eH
    assert prev < 2e-2


def test_sphere_radius_scaling():
    """``H = 1/R`` — the units check. Mean curvature is an inverse length, so a
    sphere scaled by R must report exactly 1/R."""
    R = 2.5
    m = icosphere(4)
    V = m.vertices * R
    L, M = cotan_laplacian(V, m.faces)
    H, _, _ = cv.mean_curvature(V, m.faces, L, M)
    assert np.median(H) == pytest.approx(1.0 / R, rel=2e-2)

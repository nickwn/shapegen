"""The base-sphere parameterisation: the eigenbasis conformal map reaches the
round sphere and its angle distortion falls with resolution, and the Möbius
gauge fix centres the conformal barycenter."""

import numpy as np

from shapegen.geom import conformal as cf
from shapegen.geom.eigen_sphere import conformal_sphere
from shapegen.geom.mesh import Mesh, ellipsoid, icosphere
from shapegen.geom.operators import vertex_areas


def bumpy(sub=4):
    """A smooth non-spherical genus-0 test mesh (q ≠ 0)."""
    s = icosphere(sub)
    V = s.vertices
    f = 1.0 + 0.22 * (V[:, 2] ** 2 - 0.3 * V[:, 0]) + 0.10 * V[:, 0] * V[:, 1]
    return Mesh(V * f[:, None], s.faces)


# --------------------------------------------------------------------------- #
# The conformal map: any genus-0 mesh -> round sphere                         #
# --------------------------------------------------------------------------- #

def test_ellipsoid_to_sphere():
    m = ellipsoid(4, axes=(1.0, 0.7, 0.5))
    r = np.linalg.norm(conformal_sphere(m.vertices, m.faces, K=64), axis=1)
    assert r.std() < 1e-12 and abs(r.mean() - 1.0) < 1e-12


def _angle_distortion(m, S):
    F = m.faces

    def angles(X):
        a = np.zeros((len(F), 3))
        for k in range(3):
            i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
            u, v = X[j] - X[i], X[l] - X[i]
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            a[:, k] = np.arccos(np.clip((u * v).sum(1), -1, 1))
        return a
    return np.mean(np.abs(angles(m.vertices) - angles(S)))


def test_conformal_distortion_decreases_with_resolution():
    """Measured 0.0074 -> 0.0037 (cMCF, which this replaced: 0.0078 -> 0.0038)."""
    d3, d4 = (_angle_distortion(m, conformal_sphere(m.vertices, m.faces, K=64))
              for m in (bumpy(3), bumpy(4)))
    assert d4 < 0.6 * d3 and d4 < 0.005


# --------------------------------------------------------------------------- #
# Round / Möbius sphere -> σ̂ ≈ 0                                              #
# --------------------------------------------------------------------------- #

def test_mobius_normalize_centers_barycenter():
    m = bumpy(4)
    sph = cf.ball_automorphism(conformal_sphere(m.vertices, m.faces, K=64),
                               np.array([0.3, -0.2, 0.1]))       # off-centre it first
    nrm = cf.mobius_normalize(m, sph)
    w = vertex_areas(m.vertices, m.faces)
    w = w / w.sum()
    assert np.linalg.norm((w[:, None] * nrm).sum(0)) < 1e-6


# --------------------------------------------------------------------------- #
# Test 5: encode∘decode ≈ id (approximate inverse; the hard cross-session one) #
# --------------------------------------------------------------------------- #


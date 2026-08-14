"""Acceptance tests for the base-sphere parameterisation.

Covers what the ρ codec needs from ``conformal.py``: cMCF reaches a sphere
and reduces conformal distortion, and the Möbius gauge fix actually centers
the conformal barycenter. The σ̂ route this file used to cover went with the
σ̂ stack.
"""

import numpy as np
import pytest

from bottlegen.geom import conformal as cf
from bottlegen.geom.mesh import Mesh, ellipsoid, icosphere
from bottlegen.geom.operators import vertex_areas


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


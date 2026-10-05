"""The eigenbasis conformal sphere (:mod:`shapegen.geom.eigen_sphere`)."""

from __future__ import annotations

import numpy as np
import pytest

from shapegen import codec
from shapegen.data.dataset import demo_bottle_meshes
from shapegen.geom.eigen_sphere import conformal_sphere, spectrum_error
from shapegen.geom.mesh import icosphere
from shapegen.geom.operators import vertex_areas


def _max_angle(S, T):
    """Largest angle (deg) between two sphere maps after the best rotation."""
    U, _, Vt = np.linalg.svd(S.T @ T)
    R = U @ Vt
    return float(np.degrees(np.arccos(np.clip(((S @ R) * T).sum(1), -1, 1))).max())


def _flips(S, F):
    n = np.cross(S[F[:, 1]] - S[F[:, 0]], S[F[:, 2]] - S[F[:, 0]])
    return int(((n * S[F].mean(1)).sum(1) < 0).sum())


@pytest.fixture(scope="module")
def bottle():
    return demo_bottle_meshes(n=1, subdiv=4, seed=0)[0]


def test_round_sphere_maps_to_itself():
    """The sphere is its own conformal image: the fixed point is found at once
    and is the identity up to rotation."""
    m = icosphere(4)
    P = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    S, info = conformal_sphere(P, m.faces, K=64, return_info=True)
    assert info["residual"] < 1e-9 and info["iters"] <= 5
    assert _max_angle(S, P) < 0.01


def test_more_conformal_than_cmcf_and_well_formed(bottle):
    """Converged, no folds, Möbius-normalised like ``mobius_normalize`` (input-
    area centroid at the origin), and more conformal than the cMCF it replaced
    by the spectrum test (measured 0.0072 vs cMCF's 0.0086 here; 0.0016 vs
    0.04-0.05 on scanned heads)."""
    S, info = conformal_sphere(bottle.vertices, bottle.faces, K=64, return_info=True)
    assert info["residual"] < 1e-9 and info["fails"] == 0
    assert np.allclose(np.linalg.norm(S, axis=1), 1.0)
    assert _flips(S, bottle.faces) == 0
    w = vertex_areas(bottle.vertices, bottle.faces)
    assert np.linalg.norm((w[:, None] * S).sum(0) / w.sum()) < 1e-6
    assert spectrum_error(bottle.vertices, bottle.faces, S) < 0.008


def test_anderson_reaches_the_same_map_faster(bottle):
    """Anderson acceleration changes the path, not the fixed point (the Möbius
    gauge makes the fixed point isolated; without it neither iteration
    converges). Measured 12 vs 53 steps."""
    Sa, ia = conformal_sphere(bottle.vertices, bottle.faces, K=64, return_info=True)
    Sp, ip = conformal_sphere(bottle.vertices, bottle.faces, K=64, m_hist=0,
                              return_info=True)
    assert ia["iters"] < ip["iters"]
    assert _max_angle(Sa, Sp) < 1e-4


def test_a_mirrored_eigenbasis_is_corrected(bottle):
    """The ℓ = 1 eigenvectors are defined up to any orthogonal 3x3, including a
    reflection, which would silently turn the map inside out. A mirrored input
    must still come out with every triangle positively oriented."""
    V = bottle.vertices * np.array([1.0, 1.0, -1.0])
    F = bottle.faces[:, [0, 2, 1]]
    S = conformal_sphere(V, F, K=64)
    assert _flips(S, F) == 0


def test_codec_eigen_base_sphere(bottle):
    S = codec.base_sphere(bottle)
    assert S.shape == bottle.vertices.shape

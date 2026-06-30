"""Curvature-adaptive surface sampling from σ̂.

The Hopf differential ``q = ð²σ`` gives a closed-form sizing field: the target
edge length ``h ~ 1/√(c0 + |q|)`` is small where ``|q|`` (the principal-curvature
gap) is large, so samples concentrate on sharp features. ``sample`` decodes σ̂ to
a mesh and draws points with density ``∝ 1/h² = c0 + |q|`` — the practical input
for the DiffusionNet encoder.
"""

from __future__ import annotations

import numpy as np

from ..geom.curvature import hopf_q
from ..geom.mesh import triangle_areas
from ..geom.operators import vertex_normals
from ..spectral import swsh
from .reconstruct import decode

__all__ = ["sizing_field", "hopf_magnitude_on_sphere", "adaptive_surface_sample",
           "sample"]


def hopf_magnitude_on_sphere(sigma_hat, theta, phi) -> np.ndarray:
    """``|q|`` from σ̂ evaluated at sphere angles ``(θ, φ)`` (spin-2 synthesis)."""
    L = sigma_hat.shape[0] - 1
    q_hat = swsh.q_from_sigma(sigma_hat)
    Y = swsh.eval_basis(2, L, theta, phi)
    modes = swsh.mode_list(2, L)
    qvec = np.array([q_hat[l, l + m] for (l, m) in modes])
    return np.abs(Y @ qvec)


def sizing_field(sigma_hat, theta, phi, c0=1.0) -> np.ndarray:
    """Target edge length ``h = 1/√(c0 + |q|)`` at sphere angles ``(θ, φ)``."""
    return 1.0 / np.sqrt(c0 + hopf_magnitude_on_sphere(sigma_hat, theta, phi))


def adaptive_surface_sample(V, F, n, density=None, seed=0):
    """Sample ``n`` surface points + normals with per-face probability
    ``∝ area · density`` (``density`` is a per-face weight; uniform if None)."""
    rng = np.random.default_rng(seed)
    area = triangle_areas(V, F)
    w = area if density is None else area * density
    w = w / w.sum()
    fidx = rng.choice(len(F), size=n, p=w)
    u = rng.random(n)
    v = rng.random(n)
    over = u + v > 1
    u[over], v[over] = 1 - u[over], 1 - v[over]
    p0, p1, p2 = V[F[fidx, 0]], V[F[fidx, 1]], V[F[fidx, 2]]
    pts = p0 + u[:, None] * (p1 - p0) + v[:, None] * (p2 - p0)
    vn = vertex_normals(V, F)
    nrm = (vn[F[fidx, 0]] + vn[F[fidx, 1]] + vn[F[fidx, 2]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    return pts, nrm


def sample(sigma_hat, base_V, base_F, n=4000, floor=0.1, seed=0):
    """Decode σ̂ and draw ``n`` curvature-adaptive surface samples.

    Point density ``∝ |q| + floor·mean|q|``: a small *relative* floor (default
    0.1) keeps low-curvature regions sampled while concentrating points where
    ``|q|`` is large. Returns ``(points, normals, (V, F))``.
    """
    V, F = decode(base_V, base_F, sigma_hat)
    qv = np.abs(hopf_q(V, F))                 # per-vertex Hopf magnitude
    qf = qv[F].mean(1)
    density = qf + floor * (qf.mean() + 1e-12)
    pts, nrm = adaptive_surface_sample(V, F, n, density=density, seed=seed)
    return pts, nrm, (V, F)

"""Spherical conformal parameterization + Möbius normalization — the encode
engine ``mesh → σ̂``. Required for bottles (non-star-shaped) and all real data.

Pipeline (``mesh_to_sigma``):

  1. ``spherical_conformal_map`` — conformalized mean-curvature flow (cMCF,
     Kazhdan–Solomon–Ben-Chen): the *original* cotan Laplacian ``L0`` is fixed
     while the mass ``M`` evolves, flowing any genus-0 mesh to a round sphere
     with low quasi-conformal distortion.
  2. ``mobius_normalize`` — drive the conformal (area-weighted) barycenter of the
     sphere param to the origin via ball-automorphism inversions, killing the
     ℓ=1 Möbius gauge. (The ℓ=0 scale gauge is killed by the unit-sphere
     projection; rotation is handled separately / left to canonicalization.)
  3. ``hopf_on_sphere`` — express the immersion's Hopf differential ``q`` (the
     trace-free second fundamental form, a spin-2 field) in the sphere's
     θ̂/φ̂ frame, transported through the conformal map.
  4. spin-2 least-squares SHT -> ``q̂``; ``σ̂ = q̂ / √((ℓ−1)ℓ(ℓ+1)(ℓ+2))``;
     gauge-fix (zero ℓ≤1). ``q = ð²σ`` already annihilates ℓ≤1.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from ..spectral import swsh
from .curvature import hopf_q, tangent_frames, vertex_one_ring
from .mesh import Mesh
from .operators import cotan_laplacian, vertex_areas, vertex_normals

__all__ = [
    "spherical_conformal_map", "mobius_normalize", "ball_automorphism",
    "conformal_factor", "hopf_on_sphere", "mesh_to_sigma", "gauge_fix",
]


# --------------------------------------------------------------------------- #
# Conformalized mean-curvature flow (mesh -> sphere)
# --------------------------------------------------------------------------- #

def spherical_conformal_map(mesh: Mesh, dt: float = 5e-2, iters: int = 200,
                            tol: float = 1e-6) -> np.ndarray:
    """Flow ``mesh`` to the unit sphere by cMCF and return the sphere vertex
    positions (unit vectors, same connectivity).

    Backward Euler ``(M_k + dt·L0) f_{k+1} = M_k f_k`` with ``L0`` the *fixed*
    original cotan stiffness and ``M_k`` the evolving lumped mass; recentre and
    rescale each step. Converges (after projection) to a conformal sphere param.
    """
    V = mesh.vertices.astype(np.float64).copy()
    F = mesh.faces
    L0, _ = cotan_laplacian(V, F)
    L0 = L0.tocsc()
    for _ in range(iters):
        _, Mk = cotan_laplacian(V, F)
        A = (Mk + dt * L0).tocsc()
        solve = spla.factorized(A)
        rhs = Mk @ V
        Vn = np.stack([solve(rhs[:, c]) for c in range(3)], axis=1)
        Vn -= Vn.mean(0)
        radii = np.linalg.norm(Vn, axis=1)
        Vn /= max(radii.mean(), 1e-12)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------- #
# Ball automorphisms + Möbius normalization
# --------------------------------------------------------------------------- #

def ball_automorphism(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Conformal automorphism of the unit ball/sphere sending ``a`` to ``0``:

        φ_a(x) = [(1−|a|²)(x−a) − |x−a|² a] / [1 − 2⟨x,a⟩ + |x|²|a|²].

    For ``|x|=1`` the image lies on ``S²``; the result is re-normalized to guard
    against round-off.
    """
    a = np.asarray(a, float)
    a2 = float(a @ a)
    xa = x - a[None, :]
    xa2 = (xa * xa).sum(1, keepdims=True)
    num = (1.0 - a2) * xa - xa2 * a[None, :]
    den = (1.0 - 2.0 * (x @ a) + (x * x).sum(1) * a2)[:, None]
    y = num / den
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def mobius_normalize(mesh: Mesh, sph: np.ndarray, iters: int = 100,
                     tol: float = 1e-9) -> np.ndarray:
    """Drive the conformal barycenter of ``sph`` to the origin (kills ℓ=1).

    The barycenter is weighted by the *original* surface areas (the conformal
    density), so the fixed point is canonical under ball automorphisms of the
    parameterization.
    """
    w = vertex_areas(mesh.vertices, mesh.faces)
    w = w / w.sum()
    x = sph.copy()
    for _ in range(iters):
        c = (w[:, None] * x).sum(0)
        if np.linalg.norm(c) < tol:
            break
        # damp the step for stability when c is large
        step = c / max(1.0, 2.0 * np.linalg.norm(c))
        x = ball_automorphism(x, step)
    return x


# --------------------------------------------------------------------------- #
# Conformal factor and Hopf differential on the sphere
# --------------------------------------------------------------------------- #

def conformal_factor(mesh: Mesh, sph: np.ndarray) -> np.ndarray:
    """Log conformal factor ``u`` per vertex, ``e^{2u} = dA_orig / dA_sphere``
    (ratio of lumped vertex areas; mean removed)."""
    a_orig = vertex_areas(mesh.vertices, mesh.faces)
    a_sph = vertex_areas(sph, mesh.faces)
    u = 0.5 * (np.log(np.maximum(a_orig, 1e-30)) - np.log(np.maximum(a_sph, 1e-30)))
    return u - u.mean()


def _minimal_rotation(n_from: np.ndarray, n_to: np.ndarray, v: np.ndarray):
    """Rotate vectors ``v`` by the minimal rotation taking unit ``n_from`` to
    unit ``n_to`` (per row). Rodrigues with the cross/dot of the normals."""
    axis = np.cross(n_from, n_to)
    s = np.linalg.norm(axis, axis=1, keepdims=True)
    c = (n_from * n_to).sum(1, keepdims=True)
    k = axis / np.maximum(s, 1e-30)
    kv = np.cross(k, v)
    kkv = k * (k * v).sum(1, keepdims=True)
    rot = v * c + kv * s + kkv * (1.0 - c)
    # where normals are (anti)parallel, fall back to identity / flip
    near = (s.ravel() < 1e-12)
    rot[near] = np.where(c[near] > 0, v[near], -v[near])
    return rot


def hopf_on_sphere(mesh: Mesh, sph: np.ndarray):
    """Hopf differential as a spin-2 field on the sphere param.

    Returns ``(q_sphere, theta, phi, areas)``: the complex spin-2 value, the
    sphere angles, and original vertex areas (quadrature weights) per vertex.

    The trace-free II is computed on the *original* mesh; its principal-line
    direction is transported to the sphere tangent plane by the minimal normal
    rotation (the conformal map is angle-preserving) and re-expressed in the
    sphere's θ̂/φ̂ frame, giving the spin-2 phase.
    """
    V, F = mesh.vertices, mesh.faces
    rings = vertex_one_ring(V, F)
    e1o, e2o, no = tangent_frames(V, F)
    q_orig = hopf_q(V, F, e1o, e2o, no, rings)          # spin-2 in (e1o,e2o)
    mag = np.abs(q_orig)
    # arg(q_orig) = -2θ_p, so the principal-line angle is θ_p = -½ arg(q_orig).
    # (Using +½ arg reflects the line across e1, which is frame-dependent and
    # breaks rotation covariance.) The resulting line is frame-independent.
    alpha = -0.5 * np.angle(q_orig)
    d3 = (np.cos(alpha)[:, None] * e1o + np.sin(alpha)[:, None] * e2o)

    s = sph / np.linalg.norm(sph, axis=1, keepdims=True)
    d3_s = _minimal_rotation(no, s, d3)                  # into sphere tangent plane

    theta = np.arccos(np.clip(s[:, 2], -1.0, 1.0))
    phi = np.arctan2(s[:, 1], s[:, 0])
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    that = np.stack([ct * cp, ct * sp, -st], axis=1)     # θ̂
    phat = np.stack([-sp, cp, np.zeros_like(sp)], axis=1)  # φ̂
    beta = np.arctan2((d3_s * phat).sum(1), (d3_s * that).sum(1))
    q_sphere = mag * np.exp(2j * beta)

    areas = vertex_areas(V, F)
    return q_sphere, theta, phi, areas


# --------------------------------------------------------------------------- #
# mesh -> σ̂
# --------------------------------------------------------------------------- #

def gauge_fix(coeffs: np.ndarray) -> np.ndarray:
    """Zero the ℓ≤1 modes (ℓ=0 scale, ℓ=1 translation/Möbius)."""
    out = coeffs.copy()
    out[0] = 0.0
    if out.shape[0] > 1:
        out[1] = 0.0
    return out


def mesh_to_sigma(mesh: Mesh, L: int, dt: float = 5e-2, iters: int = 200,
                  normalize: bool = True) -> np.ndarray:
    """Encode a genus-0 mesh as the spin-0 potential spectrum ``σ̂`` (layout
    ``C[ℓ, ℓ+m]``, ℓ∈[0,L]). ``q̂`` from a spin-2 least-squares SHT;
    ``σ̂ = q̂ / √((ℓ−1)ℓ(ℓ+1)(ℓ+2))``; ℓ≤1 zeroed."""
    sph = spherical_conformal_map(mesh, dt=dt, iters=iters)
    if normalize:
        sph = mobius_normalize(mesh, sph)
    q_sphere, theta, phi, areas = hopf_on_sphere(mesh, sph)
    q_hat = swsh.fit_coeffs(2, L, theta, phi, q_sphere, weights=areas)
    scale = swsh.hopf_scaling(L)
    sig_hat = swsh.empty_coeffs(L)
    for l in range(2, L + 1):
        sig_hat[l] = q_hat[l] / scale[l]
    return gauge_fix(sig_hat)

"""Spin-transform *encode*: ``(sphere, mesh) → λ → ρ``.

The decode side (:mod:`shapegen.geom.dirac`) turns a per-vertex curvature
change ``ρ`` on the base sphere into an immersion. This module inverts it.

A conformal correspondence ``f₀: S² → M`` is by definition a spin transform:
``df = λ̄ df₀ λ`` for some ``λ: S² → ℍ``, unique up to a global right ℍ factor.
Since ``v ↦ λ̄ v λ`` is a scaled rotation, ``λ`` is read off *per face* from the
similarity part of that face's affine map — no solve. Adjacent faces must agree
on the sign of ``λ`` (the ±λ ambiguity is the spin structure); on a genus-0
surface a consistent choice exists globally and is fixed by one BFS over the
dual graph. Averaging to vertices and projecting with
:func:`~shapegen.geom.dirac.rho_from_lambda` gives ``ρ``.

Deliberately convention-free: ``ρ`` is defined by whatever operator
``dirac.py`` assembles, so the encode stays consistent with the decode by
construction rather than by a calibrated multiplier.

The correspondence itself comes from
:func:`~shapegen.geom.conformal.spherical_conformal_map` (cMCF). Its
quasi-conformal distortion is the shear that :func:`face_lambdas` discards — but
it is **not** the accuracy floor of the round trip, and reading it as one is
measurably wrong in both directions. See :func:`qc_distortion`.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..spectral.quaternion import qconj
from .dirac import rho_from_lambda
from .mesh import triangle_areas

__all__ = [
    "rotation_to_quaternion", "face_lambdas", "lambda_from_correspondence",
    "qc_distortion", "mesh_to_rho_direct", "base_sphere_residual",
]


def base_sphere_residual(sph, faces) -> float:
    """Self-test for a base sphere: ``rho`` of the sphere against **itself**.

    Must be ~0 — a unit sphere has ``H = 1`` and the identity map has ``e^u = 1``,
    so ``rho = H e^u - 1 = 0``. Anything larger means the discretisation cannot
    support the curvature operator, and **every** rho computed on that sphere
    carries the same spurious field as a fixed additive offset.

    **Read this number as a discretisation warning, not as a verdict.** The cMCF
    sphere of a FaceScape head scores 8.54 against an icosphere's 0.004, and that
    correctly says its triangulation cannot support the curvature operator
    pointwise (2.1e6x triangle-area spread, areas to 3.4e-9, so the Dirac
    assembly's ``-e_p e_q / (4 A_f)`` reaches 7.4e7). But it does **not** follow
    that the sphere is unusable, and an earlier version of this docstring said so
    on the strength of this test alone. Measured end to end against the original
    heads, decoding on that same cMCF sphere reconstructs at 0.047-0.060 across
    L=8..32 — perfectly serviceable. Resampling onto a uniform icosphere is worth
    about **27%** (0.0357 vs 0.0489 at L=16), not the 2.4x first claimed; that
    figure came from scoring the icosphere route against the *resampled* head,
    which is a coarser and therefore easier target.

    Note also that this test probes a path the codec does not take:
    ``mesh_to_rho_direct`` computes ``H`` on the *target* mesh, never on the
    sphere. What the sphere's conditioning actually governs is the decode and the
    Galerkin projection.

    Guideline: below ~0.05 is fine (an icosphere gives 0.004 at subdiv 5); above
    ~0.5 the sphere is unusable and no amount of bandwidth will help, because the
    error is a fixed offset rather than anything the basis can resolve.
    """
    import numpy as _np  # noqa: PLC0415
    from .mesh import Mesh  # noqa: PLC0415
    from .operators import vertex_areas  # noqa: PLC0415
    sph = _np.asarray(sph, dtype=float)
    r = mesh_to_rho_direct(Mesh(sph, faces), sph)
    w = vertex_areas(sph, faces)
    return float(_np.sqrt((r ** 2 * w).sum() / w.sum()))


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Batched 3×3 rotations ``(m,3,3)`` → unit quaternions ``(m,4)`` ([w,x,y,z]).

    Shepperd's method: pick the branch with the largest pivot so the divisor is
    never near zero (the naive trace formula degenerates at 180°).
    """
    m = np.asarray(R, dtype=float)
    d = np.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], axis=1)
    tr = d.sum(1)
    q = np.zeros((len(m), 4))
    use_tr = tr > 0
    if use_tr.any():
        s = m[use_tr]
        t = np.sqrt(tr[use_tr] + 1.0) * 2.0
        q[use_tr, 0] = 0.25 * t
        q[use_tr, 1] = (s[:, 2, 1] - s[:, 1, 2]) / t
        q[use_tr, 2] = (s[:, 0, 2] - s[:, 2, 0]) / t
        q[use_tr, 3] = (s[:, 1, 0] - s[:, 0, 1]) / t
    pivot = np.argmax(d, axis=1)
    for i in range(3):
        sel = (~use_tr) & (pivot == i)
        if not sel.any():
            continue
        j, k = (i + 1) % 3, (i + 2) % 3
        s = m[sel]
        t = np.sqrt(1.0 + s[:, i, i] - s[:, j, j] - s[:, k, k]) * 2.0
        q[sel, 0] = (s[:, k, j] - s[:, j, k]) / t
        q[sel, 1 + i] = 0.25 * t
        q[sel, 1 + j] = (s[:, j, i] + s[:, i, j]) / t
        q[sel, 1 + k] = (s[:, k, i] + s[:, i, k]) / t
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _face_frames(V, F):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    u1, u2 = p1 - p0, p2 - p0
    n = np.cross(u1, u2)
    return u1, u2, n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def face_lambdas(S, T, F):
    """Per-face ``λ`` with ``λ̄ e λ = e_target``, shape ``(m,4)``.

    The affine map taking face ``f`` of ``S`` to face ``f`` of ``T`` is
    ``M = s·R`` for a conformal correspondence; ``s`` is the length ratio
    (``√(A_T/A_S)``) and ``λ = √s · conj(R)`` since ``λ̄ v λ`` rotates by
    ``R⁻¹``. Any shear in ``M`` (the quasi-conformal error) is discarded here.
    """
    u1, u2, ns = _face_frames(S, F)
    v1, v2, nt = _face_frames(T, F)
    s = np.sqrt(triangle_areas(T, F) / np.maximum(triangle_areas(S, F), 1e-30))
    A = np.stack([u1, u2, ns], axis=2)
    B = np.stack([v1, v2, s[:, None] * nt], axis=2)
    R = (B @ np.linalg.inv(A)) / s[:, None, None]
    return np.sqrt(s)[:, None] * qconj(rotation_to_quaternion(R))


def _dual_adjacency(F):
    emap = defaultdict(list)
    for f in range(len(F)):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            emap[(min(F[f, a], F[f, b]), max(F[f, a], F[f, b]))].append(f)
    adj = defaultdict(list)
    for fs in emap.values():
        if len(fs) == 2:
            adj[fs[0]].append(fs[1])
            adj[fs[1]].append(fs[0])
    return adj


def lambda_from_correspondence(S, T, F):
    """Per-vertex ``λ`` for the correspondence ``S → T`` on shared connectivity.

    Face spinors, sign-propagated across the dual graph (``±λ`` give the same
    transform, so neighbours are aligned by the sign of ``⟨λ_f, λ_g⟩``), then
    area-averaged to vertices.
    """
    lam_f = face_lambdas(S, T, F)
    adj = _dual_adjacency(F)
    sign = np.zeros(len(F))
    sign[0] = 1.0
    stack = [0]
    while stack:
        f = stack.pop()
        for g in adj[f]:
            if sign[g] == 0.0:
                sign[g] = sign[f] * (1.0 if float(lam_f[f] @ lam_f[g]) >= 0 else -1.0)
                stack.append(g)
    lam_f = lam_f * sign[:, None]

    lam_v = np.zeros((len(S), 4))
    wsum = np.zeros(len(S))
    w = triangle_areas(S, F)
    for c in range(3):
        np.add.at(lam_v, F[:, c], w[:, None] * lam_f)
        np.add.at(wsum, F[:, c], w)
    return lam_v / np.maximum(wsum, 1e-30)[:, None]


def qc_distortion(S, T, F):
    """Area-weighted mean quasi-conformal distortion ``(σ₁−σ₂)/(σ₁+σ₂)`` of the
    per-face map ``S → T`` (0 for an exactly conformal correspondence).

    This is the shear :func:`face_lambdas` throws away. **It is not an accuracy
    floor and it does not bound the Dirac residual** — an earlier version of this
    docstring claimed both, and both are false by measurement:

    * *Not a floor.* On a bottle encoded against the shared template, qc is 0.0665
      while the adjoint encoder reaches 0.0078 — 8.6x **below** it. qc is a
      pointwise anisotropy average; the reconstruction error is an L² vertex
      displacement after optimising over all conformal reparameterisations, and by
      uniformisation that family is every genus-0 surface. The second can be far
      smaller than the first.
    * *Not a bound on the Dirac residual.* On a FaceScape mesh qc reads 0.0289 —
      better than the bottle's template — yet the **least-squares optimal** ρ for
      that correspondence leaves a Dirac residual of 0.85 (the bottle: 0.10).

    The likely reason for the second is baked into this function: the weights are
    ``triangle_areas(S, F)``, i.e. **source** area. Wherever the map compresses —
    exactly where it is worst — the sphere-side triangles are smallest, so the
    worst regions carry the least weight. On the face above the source/target area
    ratio spans 2.3e3, so this average is close to blind to the failure.

    Useful for comparing two parameterisations of the *same* mesh, where that bias
    is shared. Do not use it as a health check, and do not tune a parameterisation
    on it: :func:`~shapegen.geom.conformal.guyau` improves qc 22% and makes
    reconstruction slightly worse.
    """
    def local(V):
        u1, u2, n = _face_frames(V, F)
        e1 = u1 / np.maximum(np.linalg.norm(u1, axis=1, keepdims=True), 1e-30)
        e2 = np.cross(n, e1)
        return np.stack([np.stack([(u1 * e1).sum(1), (u1 * e2).sum(1)], 1),
                         np.stack([(u2 * e1).sum(1), (u2 * e2).sum(1)], 1)], axis=2)
    J = local(T) @ np.linalg.inv(local(S))
    sv = np.linalg.svd(J, compute_uv=False)
    w = triangle_areas(S, F)
    return float((w * ((sv[:, 0] - sv[:, 1]) / (sv[:, 0] + sv[:, 1]))).sum() / w.sum())


def mesh_to_rho_direct(mesh, sph, return_diagnostics=False):
    """The **direct** local encoder: ``ρ = H·e^u − 1``, no spin transform.

    Ported from ``rho_codec_pkg`` (rt/exp/arms.py: ``measure``). Reads ρ straight
    off the geometry, with no spin transform and nothing to solve:

    * ``H`` — signed mean curvature of the **target** mesh, calibrated so a unit
      sphere gives ``H = +1`` (``CONVENTIONS.md``: outward normal, convex-positive);
    * ``e^u`` — the conformal factor of the map ``sph → mesh``, from
      :func:`~shapegen.geom.conformal.u_edge_lsq`, which solves
      ``(u_i+u_j)/2 = log(|B_j−B_i|/|S_j−S_i|)`` over every edge and is exact to
      3e-16 on conformal input (a lumped area ratio carries the full O(h²) of the
      mass lumping, and ρ inherits it directly);
    * the ``−1`` is the base sphere's own ``H·e^u``, making ρ a *change* in
      mean-curvature half-density.

    ``return_diagnostics`` also yields the LSQR residual of the ``u`` solve, which
    is a genuine measure of how far the correspondence is from conformal — i.e. of
    the shear the spin transform structurally cannot express. A large residual
    means the parameterisation, not the encoder, is the limiting term.
    """
    from .conformal import u_edge_lsq  # noqa: PLC0415
    from .curvature import mean_curvature  # noqa: PLC0415
    from .operators import edge_cotan_weights  # noqa: PLC0415

    V, F = mesh.vertices, mesh.faces
    E, _ = edge_cotan_weights(np.asarray(sph, dtype=float), F)
    u, resid = u_edge_lsq(np.asarray(sph, dtype=float), V, E)
    H = mean_curvature(V, F)[0]
    rho = H * np.exp(u) - 1.0
    if return_diagnostics:
        return rho, {"u": u, "H": H, "u_resid": resid}
    return rho

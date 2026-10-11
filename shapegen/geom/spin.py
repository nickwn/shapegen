"""Measuring ``ρ`` from a mesh and its sphere correspondence.

:func:`mesh_to_rho_direct` -- the *direct* measurement ``ρ = H·e^u − 1`` -- is
what training uses and the starting point of the encoders
(:mod:`shapegen.codec`, :mod:`shapegen.models.basis`). The sphere
correspondence normally comes from
:func:`shapegen.geom.eigen_sphere.conformal_sphere`.

The rest reads the spinor off a correspondence ``f₀: S² → M``: a conformal map
is a spin transform ``df = λ̄ df₀ λ``, so ``λ`` is the similarity part of each
face's affine map (no solve), sign-aligned along the best-agreeing spanning tree
of the dual graph (the ±λ spin structure; consistent on genus 0). Projecting with
:func:`shapegen.geom.dirac.rho_from_lambda` gives the mesh reference solver's
``ρ``. :func:`qc_distortion` and :func:`base_sphere_residual` are diagnostics.
"""

from __future__ import annotations

import numpy as np

from ..spectral.quaternion import qconj

from .mesh import triangle_areas

__all__ = [
    "rotation_to_quaternion", "face_lambdas", "face_spinors", "align_signs",
    "lambda_from_correspondence",
    "qc_distortion", "mesh_to_rho_direct", "base_sphere_residual",
]


def base_sphere_residual(sph, faces) -> float:
    """Self-test for a base sphere: ``rho`` of the sphere against **itself**.

    Area-weighted RMS of :func:`mesh_to_rho_direct` of the sphere onto
    itself; ideally 0 (``H = 1``, ``e^u = 1``). A large value means the
    triangulation cannot support the curvature operator pointwise, i.e. the
    sphere is badly conditioned (an icosphere at subdiv 5 gives 0.004; below
    ~0.05 is fine).

    Trap: this is a discretisation warning, not a verdict. It probes ``H`` on
    the sphere, which the codec never computes (``H`` is taken on the target),
    and a sphere with a large score can still decode serviceably.
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


def face_spinors(S, T, F):
    """:func:`face_lambdas` with one consistent sign. ``±λ`` give the same
    transform, so each face is aligned with its parent in a spanning tree of the
    dual graph by the sign of ``⟨λ_f, λ_g⟩``. The tree is the one that best
    agrees: a minimum spanning tree on ``2 − |cos∠(λ_f, λ_g)|``. Face 0 keeps
    its sign.

    Genus 0 has a single spin structure, so a continuous λ has one consistent
    sign and any tree finds it. A folded target triangle (e.g. at an ear after
    an icosphere resample) is a branch point instead, and a loop around it
    flips the sign. A breadth-first tree then ran a seam across the surface
    (FaceScape head 10 of ``trainset_001_100``: 72 dual edges at cos = −1, and
    the since-removed ``dirac`` encoder decoded at 0.48 against ~0.007). The best-agreeing tree
    only crosses the few bad edges next to the fold (2, none below cos −0.5),
    and the decode error drops to 0.0081."""
    return align_signs(face_lambdas(S, T, F), F)


def align_signs(lam, F):
    """Per-face quaternions ``lam`` (``±λ`` equivalent) with signs made
    consistent along the best-agreeing spanning tree of the dual graph of ``F``
    (see :func:`face_spinors`)."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg

    m = len(F)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    fid = np.tile(np.arange(m), 3)
    o = np.lexsort((e[:, 1], e[:, 0]))
    e, fid = e[o], fid[o]
    same = np.all(e[1:] == e[:-1], axis=1)                # consecutive rows = a shared edge
    a, b = fid[:-1][same], fid[1:][same]
    u = lam / np.maximum(np.linalg.norm(lam, axis=1, keepdims=True), 1e-300)
    cos = np.einsum("ij,ij->i", u[a], u[b])
    tree = csg.minimum_spanning_tree(
        sp.coo_matrix((2.0 - np.abs(cos), (a, b)), shape=(m, m)).tocsr())
    order, pred = csg.breadth_first_order(tree + tree.T, 0, directed=False,
                                          return_predecessors=True)
    if len(order) < m:
        raise ValueError(f"face_spinors: the dual graph is disconnected "
                         f"({len(order)} of {m} faces reachable from face 0)")
    # parity of sign flips on the path to the root, by pointer jumping
    child = order[1:]
    odd = np.zeros(m, bool)
    odd[child] = np.einsum("ij,ij->i", lam[child], lam[pred[child]]) < 0
    anc = pred.copy()
    anc[order[0]] = order[0]
    while np.any(anc != order[0]):
        odd, anc = odd ^ odd[anc], anc[anc]
    return lam * np.where(odd, -1.0, 1.0)[:, None]


def lambda_from_correspondence(S, T, F):
    """Per-vertex ``λ`` for the correspondence ``S → T`` on shared connectivity:
    :func:`face_spinors`, area-averaged to vertices."""
    lam_f = face_spinors(S, T, F)
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

    This is the shear :func:`face_lambdas` discards. Traps, both measured:

    * *Not an accuracy floor.* Reconstruction optimises over all conformal
      reparameterisations; a bottle with qc 0.0665 has been encoded to 0.0078.
    * *Not a bound on the Dirac residual.* The weights are **source** (sphere)
      area, so the regions the map compresses most weigh least; a FaceScape
      mesh reads qc 0.0289 yet its least-squares ρ leaves a residual of 0.85.

    Use it only to compare parameterisations of the same mesh, never as a
    health check or a tuning target.
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

    ``sph`` is the sphere position of each vertex of ``mesh`` (shared faces):

    * ``H`` — signed mean curvature of the **target** mesh, calibrated so a unit
      sphere gives ``H = +1`` (``CONVENTIONS.md``: outward normal, convex-positive);
    * ``e^u`` — the conformal factor of the map ``sph → mesh``, from
      :func:`~shapegen.geom.conformal.u_edge_lsq`, which solves
      ``(u_i+u_j)/2 = log(|B_j−B_i|/|S_j−S_i|)`` over every edge and is exact to
      3e-16 on conformal input (trap: a lumped area ratio instead carries an
      O(h²) error straight into ρ);
    * the ``−1`` is the base sphere's own ``H·e^u``, making ρ a *change* in
      mean-curvature half-density.

    ``return_diagnostics`` also returns ``{"u", "H", "u_resid"}``; ``u_resid``
    (the LSQR residual) measures how far the correspondence is from conformal,
    i.e. shear ρ cannot express.
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

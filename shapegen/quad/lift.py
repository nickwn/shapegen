"""Lifting the traced grid to ℝ³ (PLAN.md §P7).

**Route 2 is the lift** (:func:`lift_route2`): integrate ``df̃ = λ̄ df₀ λ``
along the traced grid's own edges with the §P9 spectral spinor and solve for
positions by least squares over the edge graph. No triangle mesh anywhere —
what M5 asks for.

**Route 1** (:func:`lift_route1`) decodes ``ρ̂`` on an icosphere with
:func:`shapegen.codec.decode` and interpolates at the traced directions. It is
kept as the *reference* route 2 is measured against, and as an escape hatch;
it is 4.3x slower. History of how the two were compared, including the
2026-09-21 dictionary bug that made route 2 look worse: PLAN.md §P7a.
"""

from __future__ import annotations

import numpy as np

from . import cones as _qc

__all__ = ["chart_to_directions", "lift_route1",
          "chart_coverage", "frame_quaternion", "spinor_quaternion",
          "arc_transformed_edges", "integrate_edges_lsq", "face_holonomy",
          "lift_route2"]


def chart_to_directions(pos: np.ndarray, Rm: np.ndarray) -> np.ndarray:
    """Chart complex positions -> unit directions on the sphere: undoes the
    stereographic projection and the pole rotation from
    :func:`shapegen.quad.cones.project_cones` (``unstereo`` then the inverse
    rotation, ``Rm.T == Rm^-1``)."""
    return _qc.unstereo(pos) @ Rm


def lift_route1(dirs: np.ndarray, rho_hat: np.ndarray,
                subdiv: int | None = None):
    """§P7 route 1, the reference lift: decode ``rho_hat`` on
    ``icosphere(subdiv)`` with :func:`shapegen.codec.decode`, then interpolate
    that surface at ``dirs`` (:func:`shapegen.geom.resample.resample_to`).

    ``subdiv=None`` picks the smallest subdivision (4-6) with at least as many
    vertices as ``dirs``: the interpolation adds no error of its own, but it
    cannot add detail the base mesh lacks. Measured on a FaceScape head at
    L=32, the same directions lifted at subdivisions 4→5 move a median 13.8%
    of the radius and 5→6 6.2% — the decode's own mesh dependence.

    Returns ``(P3, info)``; ``info`` holds the base sphere/faces, the decoded
    positions, the barycentric lookup, and ``n_fallback`` (0 on a well-shaped
    icosphere; see ``locate_on_sphere``).
    """
    from .. import codec
    from ..geom.mesh import icosphere
    from ..geom.resample import resample_to

    if subdiv is None:
        subdiv = 4
        while subdiv < 6 and 10 * 4 ** subdiv + 2 < len(dirs):
            subdiv += 1
    m = icosphere(subdiv)
    S = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    X = codec.decode(rho_hat, S, m.faces)
    P3, idx, bary, n_fallback = resample_to(S, m.faces, X, dirs)
    return P3, {"base_sphere": S, "base_faces": m.faces, "decoded": X,
               "idx": idx, "bary": bary, "n_fallback": n_fallback}


def chart_coverage(pos: np.ndarray) -> float:
    """Fraction of the sphere's area within radius ``max|pos|`` of the chart
    origin (PLAN.md §9: "coverage is a first-class output, measured where the
    trace is bounded, not estimated afterwards"). A stereographic chart's
    disk of radius ``R`` covers solid angle ``4π·R²/(R²+1)`` about the chart's
    origin (the point that got rotated to the sphere's far side).
    """
    if len(pos) == 0:
        return 0.0
    r = float(np.abs(pos).max())
    return r ** 2 / (r ** 2 + 1.0)


# --------------------------------------------------------------------------- #
# §P7 route 2: integrate df̃ = λ̄ df₀ λ along the traced edges                  #
# --------------------------------------------------------------------------- #

def frame_quaternion(theta, phi) -> np.ndarray:
    """The spinor-to-quaternion frame ``R(θ, φ)``, ``(n,4)`` in ``[w,x,y,z]``.

    :func:`~shapegen.geom.dirac_spectral.solve_spinor` returns ``λ`` as
    spin-weighted halves ``ψ1`` (spin +½) and ``ψ2`` (spin −½), components in
    a local tangent frame. The quaternion is

        λ(x) = R(θ, φ) · (ψ1 + ψ̄2 j),     R = q_z(φ) · q_y(θ − π/2)

    — frame on the **left**, ``ψ2`` **conjugated**. ``R`` carries the
    quaternion **i** axis (the one ``ℍ = ℂ ⊕ ℂj`` quantises spin about) onto
    the outward normal. Crane's ``Dλ = E₂∂₁λ − E₁∂₂λ`` multiplies by frame
    vectors on the left, so ``λ = R ψ``; with the normal along ``i``, left
    multiplication by the tangent units is complex-antilinear on ``(a, b)``,
    so the complex-linear unknowns are ``(ψ1, ψ̄2)``.

    **The round sphere cannot pin this.** Any ``A(x)·raw·B(x)`` with
    ``A raw₀ B = 1`` agrees where λ is constant; a wrong map passed that gate
    at 7.2e-15 and was off at first order in ρ (PLAN.md §P7a). The gates that
    discriminate are in ``tests/test_quad_lift_route2.py``: the kernel's
    quaternionic gauge must act as a **right** constant (a rigid motion of
    the output), and ``λ_spec⁻¹ λ_mesh`` must be constant to the mesh
    solver's discretisation error off the sphere.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    a = (theta - 0.5 * np.pi) / 2.0
    ca, sa = np.cos(a), np.sin(a)
    cp, sp = np.cos(phi / 2.0), np.sin(phi / 2.0)
    # q_z(φ) · q_y(θ − π/2)
    return np.stack([cp * ca, -sp * sa, cp * sa, sp * ca], axis=-1)


def spinor_quaternion(res, dirs: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Evaluate the §P9 spinor as a genuine quaternion field ``(n,4)`` at
    arbitrary unit directions: ``R · (ψ1 + ψ̄2 j)``, :func:`frame_quaternion`
    applied on the left of :func:`~shapegen.spectral.swsh.half_eval_basis`.

    **This is single-valued**, even though neither factor is.
    ``half_eval_basis`` warns that its values are antiperiodic in ``φ``
    (half-integer ``m``) — conjugation keeps that — and ``R`` is antiperiodic
    for the same reason, so the product is periodic.

    Chunked because the basis is ``(npts, nmodes)`` and ``nmodes`` is 1640 at
    ``Ls = 39.5`` (L = 32 with §P9's padding) — forming it for a whole traced
    mesh at once would be gigabytes for no gain.
    """
    from ..spectral import swsh as sw
    from ..spectral.quaternion import qmul

    d = np.asarray(dirs, dtype=float)
    theta = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
    phi = np.arctan2(d[:, 1], d[:, 0])
    out = np.empty((len(d), 4))
    for i in range(0, len(d), chunk):
        s = slice(i, min(i + chunk, len(d)))
        a = sw.half_eval_basis(+0.5, res.Ls, theta[s], phi[s]) @ res.psi1
        b = sw.half_eval_basis(-0.5, res.Ls, theta[s], phi[s]) @ res.psi2
        out[s] = qmul(frame_quaternion(theta[s], phi[s]),
                      np.stack([a.real, a.imag, b.real, -b.imag], axis=1))
    return out


def arc_transformed_edges(res, P: np.ndarray, edges: np.ndarray,
                          rule: str = "simpson") -> np.ndarray:
    """``ẽ_{i→j} = ∫ λ̄ df₀ λ`` along the great-circle arc, per edge ``(E,3)``.

    ``df₀`` is the base immersion's differential and the base immersion is the
    identity on the sphere, so ``df₀`` applied to the arc's velocity *is* the
    arc's velocity — no mesh, no interpolation, and the arc is the traced
    grid's own edge.

    ``rule="simpson"`` evaluates at both endpoints and the arc midpoint.
    ``rule="gauss5"`` is a 5-point Gauss-Legendre rule on the same arc, for
    checking the order.  Measured on the round sphere, where the answer is
    exactly the sphere, the Simpson residual falls 4.50e-10 → 1.38e-11 between
    icosphere subdivisions 3 and 4 — a ratio of 32.6, i.e. ``O(h⁵)`` per edge,
    as Simpson gives.  At ``ρ ≠ 0`` the quadrature error is far below the
    ``γ`` term and refining the rule changes nothing.
    """
    from ..spectral.quaternion import qmul, qconj, from_vec3, to_vec3

    P = np.asarray(P, dtype=float)
    edges = np.asarray(edges)
    xi, xj = P[edges[:, 0]], P[edges[:, 1]]
    dot = np.clip((xi * xj).sum(1), -1.0, 1.0)
    Om = np.arccos(dot)
    sO = np.where(Om > 1e-12, np.sin(Om), 1.0)

    if rule == "simpson":
        ts = np.array([0.0, 0.5, 1.0])
        ws = np.array([1.0, 4.0, 1.0]) / 6.0
    elif rule == "gauss5":
        x, w = np.polynomial.legendre.leggauss(5)
        ts, ws = 0.5 * (x + 1.0), 0.5 * w
    else:
        raise ValueError(f"unknown rule {rule!r}")

    out = np.zeros((len(edges), 3))
    for t, wt in zip(ts, ws):
        # point and velocity on the great-circle arc
        c1 = np.sin((1.0 - t) * Om) / sO
        c2 = np.sin(t * Om) / sO
        x = c1[:, None] * xi + c2[:, None] * xj
        n = np.linalg.norm(x, axis=1, keepdims=True)
        x = np.where(n > 1e-12, x / np.maximum(n, 1e-300), xi)
        v = (Om / sO)[:, None] * (-np.cos((1.0 - t) * Om)[:, None] * xi
                                  + np.cos(t * Om)[:, None] * xj)
        lam = spinor_quaternion(res, x)
        out += wt * to_vec3(qmul(qmul(qconj(lam), from_vec3(v)), lam))
    return out


def integrate_edges_lsq(n_vertices: int, edges: np.ndarray, et: np.ndarray):
    """Positions from edge vectors by least squares:
    ``min_x Σ_ij |x_j − x_i − ẽ_ij|²``.  Returns ``(X, G)``.

    The normal equations are the **graph Laplacian** ``GᵀG x = Gᵀ ẽ`` — one
    sparse solve per coordinate, far simpler than
    :func:`shapegen.geom.dirac.integrate_edges`'s cotan-Poisson, and it needs
    no faces, no areas and no triangle mesh.  This is ``05_experiments/
    poisson2.py``'s construction, which PLAN.md §1 records as *exact* rather
    than ``O(h²)`` — "the mesh is scaffolding for linear algebra, not a
    discretisation".

    Translation is the gauge (vertex 0 is pinned, then the mean removed);
    ``G`` is returned so the caller can form the per-edge residual
    ``G X − ẽ``, which is what the least squares absorbed and what Gate P7.A
    is about.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    edges = np.asarray(edges)
    et = np.asarray(et, dtype=float)
    m = len(edges)
    rows = np.concatenate([np.arange(m), np.arange(m)])
    cols = np.concatenate([edges[:, 0], edges[:, 1]])
    vals = np.concatenate([-np.ones(m), np.ones(m)])
    G = sp.coo_matrix((vals, (rows, cols)), shape=(m, n_vertices)).tocsr()
    A = (G.T @ G).tocsc()
    b = G.T @ et
    keep = np.arange(1, n_vertices)
    solve = spla.factorized(A[keep][:, keep])
    X = np.zeros((n_vertices, 3))
    for c in range(3):
        X[keep, c] = solve(b[keep, c])
    return X - X.mean(0), G


def face_holonomy(edges: np.ndarray, et: np.ndarray, faces) -> np.ndarray:
    """**Gate P7.A.** ``Σ ẽ`` around each traced face loop, ``(F,3)``.

    ``df̃`` is closed only where the Dirac equation holds exactly, i.e. at
    ``γ = 0``; at ``γ ≠ 0`` the integral is path-dependent and this is the
    obstruction. It is measured *before* the least-squares solve, which
    would otherwise absorb it silently. Unweighted least squares is enough:
    fed the mesh solver's spinor, it reproduces route 1 to 1e-4 (PLAN.md
    §P7a).
    """
    key = {}
    for e, (i, j) in enumerate(np.asarray(edges)):
        key[(int(i), int(j))] = (e, +1.0)
        key[(int(j), int(i))] = (e, -1.0)
    out = np.zeros((len(faces), 3))
    for f, loop in enumerate(faces):
        loop = list(loop)
        for k in range(len(loop)):
            hit = key.get((int(loop[k]), int(loop[(k + 1) % len(loop)])))
            if hit is not None:
                out[f] += hit[1] * et[hit[0]]
    return out


def lift_route2(dirs: np.ndarray, edges: np.ndarray, rho_hat: np.ndarray,
                L: int, faces=None, res=None, c: float = 1.0,
                rule: str = "simpson"):
    """§P7 route 2: lift the traced grid to ℝ³ **without any triangle mesh**.

    ``dirs`` are the traced vertices as unit directions (``QuadGrid.points``
    rotated by ``chart["Rm"]``), ``edges`` the grid's own ``(E,2)`` edge list.
    ``faces`` is optional and only used to report Gate P7.A.

    Evaluate the §P9 spinor along each arc (:func:`spinor_quaternion`),
    integrate ``df̃`` (:func:`arc_transformed_edges`), and solve the edge graph
    in least squares (:func:`integrate_edges_lsq`).

    Returns ``(P3, info)`` with ``info`` carrying ``gamma``, the spinor, the
    edge vectors, ``edge_residual`` (relative rms of ``G X − ẽ``, the
    path-dependence the solve absorbed) and, when ``faces`` is given,
    ``holonomy_rms`` / ``holonomy_rel`` for Gate P7.A. The result is defined
    up to a similarity — compare with :func:`shapegen.codec.shape_error`.

    **Pass ``res``** — the spinor
    :func:`shapegen.quad.field.solve_potential_spinor` already solved for the
    connection. Re-solving it costs 15-32s at L=32 against the lift's ~5s.

    **Measured** (FaceScape, L=32, 20k faces, traced once and lifted both
    ways; full tables in PLAN.md §P7a): over 10 heads, Procrustes against
    :func:`lift_route1` is 0.006-0.012 at ``band_taper`` t=1e-3 and
    0.021-0.045 untapered, route 2 is closer to the scanned head on 10 of 12
    cases, and quad isotropy is unchanged. The loser is head 252 (+7%
    tapered, +19% untapered at γ = 4.1e-2), where route 1 itself is 3-5x
    further from the scan than on any other head. Gate P7.A holonomy is
    second order in amplitude (≈ 2e-3·γ on the ``ℓ^-2`` fixture; 6e-3 to
    2e-2 relative on faces, a diffuse tail rather than a local defect).
    """
    from ..geom.dirac_spectral import (potential_band_extra, solve_spinor,
                                       spinor_band)

    if res is None:
        res = solve_spinor(np.asarray(rho_hat, float), L, c=c,
                           Ls=spinor_band(L, potential_band_extra(L)))
    dirs = np.asarray(dirs, dtype=float)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    et = arc_transformed_edges(res, dirs, edges, rule=rule)
    X, G = integrate_edges_lsq(len(dirs), edges, et)

    scale = float(np.linalg.norm(et, axis=1).mean())
    resid = G @ X - et
    info = {"res": res, "gamma": float(res.gamma), "et": et,
            "edge_scale": scale,
            "edge_residual": float(np.sqrt((resid ** 2).sum(1).mean()) / max(scale, 1e-300))}
    if faces is not None:
        hol = face_holonomy(edges, et, faces)
        info["holonomy"] = hol
        info["holonomy_rms"] = float(np.sqrt((hol ** 2).sum(1).mean()))
        info["holonomy_rel"] = info["holonomy_rms"] / max(scale, 1e-300)
    return X, info

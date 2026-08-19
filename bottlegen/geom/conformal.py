"""Spherical conformal parameterisation + Möbius gauge fix.

Produces the base sphere the ρ codec decodes from: a bijective embedding of a
genus-0 mesh onto S², on the mesh's own connectivity, with the ℓ=1 Möbius
freedom removed.

  1. :func:`spherical_conformal_map` — conformalized mean-curvature flow (cMCF,
     Kazhdan–Solomon–Ben-Chen): the *original* cotan Laplacian ``L0`` is held
     fixed while the mass ``M`` evolves, flowing any genus-0 mesh to a round
     sphere with low quasi-conformal distortion.
  2. :func:`mobius_normalize` — drive the conformal (area-weighted) barycenter of
     the sphere parameterisation to the origin via ball-automorphism inversions.
     This is the Möbius-registration centering: it kills the 3-dimensional
     Möbius "boost" freedom, leaving only rotation. (Scale is fixed by the unit
     sphere projection; **rotation is not fixed by anything** — see
     ``SIMPLIFICATION_PLAN.md`` §7.)

``scripts/willmore_flow.py`` offers an alternative to step 1 via the spin
transform, which on a bottle at subdiv 3 gives slightly *lower* distortion
(0.017 vs 0.019). Either works: because the encoder optimises decoded *shape*
error rather than the correspondence, the parameterisation's conformal quality is
largely compensated — the round trip lands well below the qc figure.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import Mesh
from .operators import cotan_laplacian, vertex_areas, vertex_normals

__all__ = ["spherical_conformal_map", "mobius_normalize", "ball_automorphism",
           "u_edge_lsq", "harmonic_energy", "guyau", "conformal_param"]


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

# --------------------------------------------------------------------------- #
# mesh -> σ̂
# --------------------------------------------------------------------------- #


def u_edge_lsq(S: np.ndarray, B: np.ndarray, E: np.ndarray):
    """Log conformal factor ``u`` of the map ``S → B``, by edge least squares.

    Ported from ``rho_codec_pkg`` (rt/param/ufactor.py). For a conformal map every
    edge scales by the local factor, so

        (u_i + u_j)/2 = log( |B_j − B_i| / |S_j − S_i| )

    is one equation per edge — overdetermined, and solved by LSQR. Uniquely
    solvable because triangles are odd cycles, which kills the alternating
    null space a bipartite graph would have.

    This is far better than the lumped-area estimate ``e^u = √(a_B/a_S)``: it is
    exact to ~7e-14 on genuinely conformal input, where the area ratio carries the
    full O(h²) error of the mass lumping. ``ρ = H·e^u − 1`` inherits that error
    directly, so it is a first-order term for the direct encoder.

    Returns ``(u, residual_norm)``. The residual is a useful diagnostic on its own:
    it measures how far the correspondence is from conformal, so a large value
    means the map has shear the spin transform cannot express.
    """
    l0 = np.linalg.norm(S[E[:, 1]] - S[E[:, 0]], axis=1)
    l1 = np.linalg.norm(B[E[:, 1]] - B[E[:, 0]], axis=1)
    m, N = len(E), len(S)
    A = sp.coo_matrix((np.full(2 * m, 0.5),
                       (np.repeat(np.arange(m), 2), E.ravel())),
                      shape=(m, N)).tocsr()
    out = spla.lsqr(A, np.log(np.maximum(l1, 1e-300) / np.maximum(l0, 1e-300)),
                    atol=1e-14, btol=1e-14, iter_lim=8000)
    return out[0], float(out[3])


def harmonic_energy(P: np.ndarray, L) -> float:
    """Dirichlet energy ``½ tr(Pᵀ L P)`` of a sphere map, under the *source* cotan
    Laplacian. The harmonic map minimises it; a collapsed map drives it to zero,
    which is the failure mode :func:`guyau` guards against."""
    return float(0.5 * np.einsum("ij,ij->", P, L @ P))


def guyau(mesh: Mesh, sph: np.ndarray, iters: int = 2000, omega: float = 0.5,
          recenter_every: int = 10, tol: float = 1e-13, verbose: bool = False):
    """Gu–Yau relaxation: polish a cMCF sphere into the true harmonic map.

    Ported from ``rho_codec_pkg`` (rt/param/cmcf.py). cMCF lands *near* the
    conformal map; this converges to it. The local update

        φ_i ← normalise( Σ_j w_ij φ_j / Σ_j w_ij )

    minimises Dirichlet energy over ``φ_i`` on S², and Möbius re-centring against
    the *source* measure keeps the flow off the constant map — otherwise the
    global minimiser.

    Measured on bottles: quasi-conformal distortion improves ~22%, 0.0194 → 0.0149
    at subdiv 3 and 0.0097 → 0.0076 at subdiv 4. Note the energy barely moves
    (12.5554 → 12.5549) while qc improves a fifth, so **energy is a poor progress
    signal here** — judge the parameterisation by qc.

    **Two traps, both measured.**

    *Undamped Jacobi overshoots.* This updates every vertex simultaneously;
    per-vertex minimisation is monotone under Gauss–Seidel, not under Jacobi.
    At ``omega=1.0`` (the reference's default) it improves for 66 sweeps at subdiv
    4 and then runs away — unguarded it reached E=340 and qc=0.68, returning a
    sphere that still *looks* like a sphere and is a ruined parameterisation.
    ``omega=0.5`` is stable and reaches a better optimum.

    *Do not clamp negative cotan weights.* Obtuse triangles give ``w_ij < 0``
    (1.67% of edges at subdiv 3, 2.16% at subdiv 4) and clamping them looks like
    the textbook fix for the overshoot. It is not: clamped, the energy never
    decreases at all — 0 improving sweeps at either resolution — and guyau becomes
    a no-op. The negative weights carry real geometry.

    The best-iterate guard is what makes this safe: divergence is silent, so the
    routine keeps the lowest-energy sphere it saw rather than whatever the last
    sweep produced.

    Sign convention: ``cotan_laplacian`` returns ``L = D − W`` (PSD), so the
    off-diagonal block is ``−W``; the ratio in the update cancels the sign.
    """
    L, _ = cotan_laplacian(mesh.vertices, mesh.faces)
    Loff = L - sp.diags(L.diagonal())
    den = np.asarray(Loff.sum(axis=1)).ravel()
    den = np.where(np.abs(den) < 1e-300, -1e-300, den)

    unit = lambda P: P / np.linalg.norm(P, axis=1, keepdims=True)
    P = unit(np.asarray(sph, dtype=float))
    best = (harmonic_energy(P, L), P.copy())
    prev = best[0]
    for it in range(iters):
        Q = unit((Loff @ P) / den[:, None])
        P = unit((1.0 - omega) * P + omega * Q)
        if (it + 1) % recenter_every == 0:
            P = mobius_normalize(mesh, P)
        E = harmonic_energy(P, L)
        if not np.isfinite(E):
            break
        if E < best[0]:
            best = (E, P.copy())
        if it > 50 and abs(E - prev) < tol * max(1.0, abs(E)):
            break
        prev = E
    P = mobius_normalize(mesh, best[1])
    if verbose:
        print(f"  guyau: E {harmonic_energy(unit(np.asarray(sph, float)), L):.6f} "
              f"-> {best[0]:.6f} in {it + 1} sweeps")
    return P


def conformal_param(mesh: Mesh, cmcf_iters: int = 300, **kw) -> np.ndarray:
    """The full parameterisation: cMCF → Möbius centring → Gu–Yau polish.

    This is the three-stage pipeline ``rho_codec_pkg`` uses in production, where
    shapegen historically stopped after two. Everything downstream — ρ, the
    decode, the encoder's floor — inherits the parameterisation's quality.
    """
    S = mobius_normalize(mesh, spherical_conformal_map(mesh, iters=cmcf_iters))
    return guyau(mesh, S, **kw)

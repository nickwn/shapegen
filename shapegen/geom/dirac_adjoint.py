"""Adjoint machinery for the Dirac eigensolve — the null space and its projectors.

Differentiating ``ρ ↦ λ`` means solving ``(A − γM) q = p`` for a cotangent ``q``,
and that system is singular: the eigenproblem is quaternion-*linear*, so if
``Aλ = γMλ`` then ``A(λq) = γM(λq)`` for any constant quaternion ``q``. In the
real 4×4 embedding :func:`~shapegen.geom.dirac.left_matrix` builds — where the
operator is left multiplication and therefore commutes with right multiplication
— the null space is spanned by ``λ·1, λ·i, λ·j, λ·k`` and is **4**-dimensional.
(A complex 2N encoding of the same problem sees only 2 of these; deflating the
wrong number is a silent, large error, not a crash.)

Every adjoint solve must deflate all four, and :meth:`NullProjector.canary` must
be asserted afterwards.

**``P`` and ``Pᵀ`` are different projectors and both are needed.** With
``K = A − γM`` Hermitian and ``Kv = r``, the derivative is taken along the
M-orthogonal section ``v = P dx``, so

    ⟨p, P dx⟩ = ⟨Pᵀp, dx⟩ = ⟨Pᵀp, K⁻¹r⟩ = ⟨q, r⟩   where  Kq = Pᵀp.

The **input** cotangent is therefore deflated with ``Pᵀ`` — only ``Pᵀp`` is
guaranteed to lie in ``range(K)``, so ``q`` exists — and the **output** ``q``
with ``P``, which makes ``⟨q, Mλ⟩ = 0`` and drops the ``dγ·Mλ`` term from the
contraction. They coincide only when ``M`` is a multiple of the identity; the
lumped vertex areas of an icosphere span ~1.5×, so using one for the other is a
real error.

One pleasant fact: ``λ·1, λ·i, λ·j, λ·k`` are *automatically* M-orthonormal when
``λ`` is M-normalised, because

    ⟨λa, λb⟩_M = Σ_v m_v Re(ā λ̄_v λ_v b) = (Σ_v m_v |λ_v|²) ⟨a, b⟩.

So the Gram–Schmidt below is a numerical safeguard, not a construction — which
is why it can be run twice and trusted.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from ..spectral.quaternion import from_vec3, qconj, qmul
from . import dirac
from .operators import cotan_laplacian, edge_cotan_weights, vertex_areas

__all__ = ["null_basis", "NullProjector", "CanaryError", "decode_forward",
           "stage_a_adjoint", "stage_b_adjoint", "stage_c_adjoint",
           "decode_adjoint"]


class CanaryError(RuntimeError):
    """Null-space contamination leaked into the adjoint solve.

    Not a convergence warning — the gradient is wrong by however much of the
    gauge direction ``q`` picked up. Fix this before investigating anything else.
    """


def null_basis(lam: np.ndarray) -> np.ndarray:
    """The four null directions ``λ·{1,i,j,k}`` of ``A − γM``. ``(n,4) → (4,n,4)``.

    Written out per vertex from ``(w,x,y,z)`` rather than called through
    :func:`~shapegen.spectral.quaternion.qmul`, because this sits inside every
    adjoint solve. The table encodes ``[w,x,y,z]`` storage with ``ij=k``; it is
    *checked* against ``qmul`` in the tests rather than trusted.
    """
    lam = np.asarray(lam, dtype=float)
    w, x, y, z = lam[:, 0], lam[:, 1], lam[:, 2], lam[:, 3]
    return np.stack([
        np.stack([w, x, y, z], axis=-1),        # λ·1
        np.stack([-x, w, z, -y], axis=-1),      # λ·i
        np.stack([-y, -z, w, x], axis=-1),      # λ·j
        np.stack([-z, y, -x, w], axis=-1),      # λ·k
    ], axis=0)


class NullProjector:
    """Projectors off ``span{λ·1, λ·i, λ·j, λ·k}`` for a fixed ``λ``.

    ``lam`` is ``(n,4)``; ``mass`` the per-vertex lumped areas ``(n,)`` from
    :func:`~shapegen.geom.operators.vertex_areas` (the diagonal of the
    quaternionic mass, before the repeat-by-4).

    Only the M-orthogonal ``P`` and its plain-inner-product adjoint ``PT`` are
    provided, which is what a *direct* deflated solve needs. An iterative solve
    would additionally need the plain-orthogonal (hence symmetric) projector:
    ``P`` is not self-adjoint in the plain inner product, so ``P·K`` is not
    symmetric and CG has no right to converge on it.
    """

    def __init__(self, lam: np.ndarray, mass: np.ndarray, n_gs: int = 2):
        self.mass = np.asarray(mass, dtype=float)
        self.n = self._orthonormalise(null_basis(lam), n_gs)

    # ------------------------------------------------------------------ inner products
    def _mdot(self, u: np.ndarray, v: np.ndarray) -> float:
        """``⟨u,v⟩_M`` over the ``(n,4)`` axes."""
        return float((u * v * self.mass[:, None]).sum())

    @staticmethod
    def _pdot(u: np.ndarray, v: np.ndarray) -> float:
        """Plain ``⟨u,v⟩``."""
        return float((u * v).sum())

    def _orthonormalise(self, n: np.ndarray, n_gs: int) -> np.ndarray:
        n = n.copy()
        for _ in range(max(n_gs, 1)):           # "twice is enough"
            for c in range(4):
                v = n[c]
                for d in range(c):
                    v = v - self._mdot(n[d], v) * n[d]
                n[c] = v / np.sqrt(max(self._mdot(v, v), 1e-300))
        return n

    # ------------------------------------------------------------------ projectors
    def P(self, z: np.ndarray) -> np.ndarray:
        """M-orthogonal projection off the null space. Deflates the solve *output*."""
        z = np.array(z, dtype=float, copy=True)
        for c in range(4):
            z = z - self._mdot(self.n[c], z) * self.n[c]
        return z

    def PT(self, z: np.ndarray) -> np.ndarray:
        """Adjoint of :meth:`P` in the plain inner product. Deflates the *input*."""
        z = np.array(z, dtype=float, copy=True)
        m = self.mass[:, None]
        for c in range(4):
            u = self.n[c]
            z = z - self._pdot(u, z) * (m * u)
        return z

    # ------------------------------------------------------------------ diagnostics
    def canary(self, q: np.ndarray) -> float:
        """``max_i |⟨n_i, q⟩_M| / ‖q‖_M`` — assert this on every adjoint solve.

        A non-zero canary means null-space contamination leaked into ``q``, and
        the gradient is wrong by however much of the gauge direction it picked
        up. It is the first thing to check when an adjoint disagrees with finite
        differences.
        """
        nq = np.sqrt(max(self._mdot(q, q), 1e-300))
        return max(abs(self._mdot(self.n[c], q)) for c in range(4)) / nq

    def orthonormality_error(self) -> float:
        """``max |⟨n_i,n_j⟩_M − δ_ij|``. At round-off by construction."""
        G = np.array([[self._mdot(self.n[a], self.n[b]) for b in range(4)]
                      for a in range(4)])
        return float(np.abs(G - np.eye(4)).max())


# --------------------------------------------------------------------------- #
# Forward decode, with the cache the backward pass needs                      #
# --------------------------------------------------------------------------- #

def decode_forward(V, F, rho, sigma=1e-6):
    """``ρ → X`` on the base mesh ``(V, F)``. Returns ``(X, cache)``.

    Identical to ``integrate_edges(V, F, spin_transform(V, F, rho)[0])``; it
    exists to hand the backward pass the ``λ`` and ``γ`` it would otherwise
    recompute.
    """
    lam, gamma = dirac.spin_transform(V, F, rho, sigma=sigma)
    X = dirac.integrate_edges(V, F, lam)
    return X, {"V": V, "F": F, "rho": np.asarray(rho, dtype=float),
               "lam": lam, "gamma": gamma}


# --------------------------------------------------------------------------- #
# Stage C: Poisson integration (linear)                                       #
# --------------------------------------------------------------------------- #

def stage_c_adjoint(dJdX, V, F):
    """``dJ/dX → dJ/dẽ``.

    Forward is ``X = L⁺ Gᵀ W ẽ`` with ``L = Gᵀ W G`` symmetric, so the adjoint is
    ``W G L⁺ dJ/dX`` — the *same* Poisson solve, not a different one.

    ``dJdX`` is mean-removed first. That is not cosmetic: ``L⁺`` is realised by
    pinning vertex 0, which silently returns a wrong answer rather than failing
    if the right-hand side is outside ``range(L)``. Projecting first makes the
    pinned system consistent, and the mean it removes would have been killed by
    ``G`` anyway.
    """
    L, _ = cotan_laplacian(V, F)
    edges, w = edge_cotan_weights(V, F)
    i, j = edges[:, 0], edges[:, 1]
    p = np.asarray(dJdX, dtype=float)
    x = dirac._poisson_solve(L, p - p.mean(0))
    return w[:, None] * (x[j] - x[i])


# --------------------------------------------------------------------------- #
# Stage B: the CPS edge rule (bilinear)                                       #
# --------------------------------------------------------------------------- #

_EDGE_QUAD = ((1.0 / 3.0, 0, 0), (1.0 / 6.0, 0, 1),
              (1.0 / 6.0, 1, 0), (1.0 / 3.0, 1, 1))
_CONJ = np.array([1.0, -1.0, -1.0, -1.0])      # conjugation as a sign vector


def stage_b_adjoint(dEt, V, lam, i, j):
    """``dJ/dẽ → dJ/dλ`` for ``ẽ = ∫₀¹ λ̄(t) e λ(t) dt`` — four bilinear terms.

    Pad the cotangent to a pure-imaginary quaternion ``p = (0, dJ/dẽ)``; this is
    what makes ``to_vec3`` in the forward self-adjoint. With ``u = λ̄_a e`` held
    fixed and ``s = u λ_b``, the two quaternion adjoint rules are

        ⟨p, u v⟩ = ⟨ū p, v⟩       (left multiplication)
        ⟨p, u v⟩ = ⟨p v̄, u⟩       (right multiplication)

    giving ``∂/∂λ_b = ū p`` and ``∂/∂u = p λ̄_b``. Then ``u = λ̄_a e``, so
    ``∂/∂λ_a = conj((∂/∂u) ē)`` — conjugation is self-adjoint, being a diagonal
    sign matrix.

    The pairing is the trap: ``∂/∂u`` must be ``qmul(p, qconj(λ_b))``, not
    ``qmul(p, qconj(u))``, and ``p`` must carry its zero real part explicitly.
    Both mistakes leave a plausible-looking gradient, which is why this stage has
    its own isolated finite-difference gate.
    """
    e = from_vec3(V[j] - V[i])
    p = np.concatenate([np.zeros((len(i), 1)), np.asarray(dEt, dtype=float)],
                       axis=1)
    idx = (np.asarray(i), np.asarray(j))
    dlam = np.zeros_like(lam)
    for (w_, ai, bi) in _EDGE_QUAD:
        a, b = lam[idx[ai]], lam[idx[bi]]
        u = qmul(qconj(a), e)
        du = qmul(p, qconj(b)) * w_
        db = qmul(qconj(u), p) * w_
        da = _CONJ * qmul(du, qconj(e))
        np.add.at(dlam, idx[ai], da)
        np.add.at(dlam, idx[bi], db)
    return dlam


# --------------------------------------------------------------------------- #
# Stage A: the eigensolve                                                     #
# --------------------------------------------------------------------------- #

def stage_a_adjoint(dlam, V, F, rho, lam, gamma, shift=1e-9, canary_tol=1e-8,
                    stats=None):
    """``dJ/dλ → dJ/dρ``.

    From ``A(ρ)λ = γMλ``, a perturbation gives ``(A − γM) dλ = −(dA − dγ·M)λ``.
    With ``K = A − γM`` symmetric and ``q`` solving ``Kq = Pᵀ(dJ/dλ)``,

        dJ/dρ_v = −⟨q, (∂A/∂ρ_v) λ⟩,

    which is exactly :func:`~shapegen.geom.dirac.dirac_grad_rho`. The ``dγ·Mλ``
    term drops because ``P`` leaves ``q`` M-orthogonal to the null space.

    **``J`` must be gauge-invariant** — invariant under ``λ → λq`` for constant
    ``q ∈ ℍ*``, i.e. under a global rotation *and* scaling of the decoded
    surface. Only then is ``dJ/dλ`` plain-orthogonal to the null space, and only
    then is ``⟨dJ/dλ, dλ⟩`` independent of which section through the eigenspace
    is differentiated. A Procrustes-aligned shape error qualifies; raw
    ``‖X − X_target‖²`` does not, and would need the Jacobian of ``_gauge_fix``
    on top. This is a restriction on the loss, not an approximation.

    ``shift`` moves ``K`` off its null space so it can be factorised, at the cost
    of a relative gradient error of ~``shift``/gap. Measured against finite
    differences, the error is exactly linear in it::

        shift    1e-6      1e-7      1e-8      1e-9      1e-10
        rel err  1.69e-6   1.81e-7   6.76e-8   6.53e-8   6.53e-8

    flattening at 6.5e-8 where the *finite differences* run out of accuracy, not
    the adjoint. The default sits at that floor. It is not smaller because there
    is no measured benefit below it, rather than because anything breaks: at
    n=2562 a shift of 1e-10 still solves cleanly (canary 1.5e-17). Raise it only
    if a mesh ever makes the factorisation complain.
    """
    A, M = dirac.dirac_operator(V, F, rho)
    proj = NullProjector(lam, vertex_areas(V, F))
    Ks = ((A - gamma * M) + shift * M).tocsc()
    q = proj.P(spla.splu(Ks).solve(proj.PT(dlam).ravel()).reshape(-1, 4))

    canary = proj.canary(q)
    if stats is not None:
        stats["canary"] = canary
        stats["gamma"] = gamma
    if canary > canary_tol:
        raise CanaryError(
            f"null-space canary {canary:.3e} exceeds {canary_tol:.1e}: the "
            f"adjoint solve leaked gauge directions into q and the gradient is "
            f"wrong. Stop and fix this first.")
    return -dirac.dirac_grad_rho(V, F, rho, q, lam)


# --------------------------------------------------------------------------- #
# The composite                                                               #
# --------------------------------------------------------------------------- #

def decode_adjoint(dJdX, cache, **kw):
    """``dJ/dX → dJ/dρ`` — the three stage adjoints in reverse.

    ``cache`` is the second return of :func:`decode_forward`. See
    :func:`stage_a_adjoint` for the gauge-invariance requirement on ``J``.
    """
    V, F = cache["V"], cache["F"]
    edges, _ = edge_cotan_weights(V, F)
    dEt = stage_c_adjoint(dJdX, V, F)
    dlam = stage_b_adjoint(dEt, V, cache["lam"], edges[:, 0], edges[:, 1])
    return stage_a_adjoint(dlam, V, F, cache["rho"], cache["lam"],
                           cache["gamma"], **kw)

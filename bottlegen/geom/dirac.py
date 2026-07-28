"""Quaternionic Dirac operator and the discrete spin transform — the decode
engine that turns a curvature change ``ρ`` (and hence ``σ̂``) into a new
genus-0 immersion, including non-star-shaped shapes a radial graph cannot
represent.

Construction (Crane–Pinkall–Schröder, *Spin Transformations of Discrete
Surfaces*). The intrinsic Dirac operator is the face-assembled quaternionic
matrix

    X_{pq} = − (1 / 4A_f) · e_p e_q        (sum over faces f ∋ p, q)

where ``e_p`` is the edge of face ``f`` opposite vertex ``p`` (a pure-imaginary
quaternion ``V_next − V_prev``). Because the three opposite edges of a triangle
sum to zero, every row of ``X`` sums to zero, so the constant spinor ``λ ≡ 1``
is in its kernel — that is the ``ρ = 0 ⇒ λ ≡ 1`` identity that gates this
module. ``X`` is quaternionic-Hermitian; we represent each quaternion by its
real 4×4 left-multiplication matrix, making the assembled operator a real
*symmetric* ``4|V|×4|V|`` sparse matrix and the eigenproblem a standard
symmetric generalized one.

**``X`` is the Dirac *energy*, not the Dirac operator.** Expanding the form,

    λ* X λ = Σ_f |Σ_{p∈f} e_p λ_p|² / (4 A_f)  ≥ 0,

so ``X = D* M_F D`` for the per-face first-order operator

    (Dλ)_f = −(1/2A_f) Σ_{p∈f} e_p λ_p        [units 1/length, a curvature]

and ``X`` is PSD. The curvature change ``ρ`` must therefore enter *inside* that
norm, where it carries the edge quaternions:

    E(λ) = Σ_f A_f |(Dλ)_f − (ρλ)_f|²,   (ρλ)_f = ⅓ Σ_{p∈f} ρ_p λ_p.

Coupling ``ρ`` through the mass instead (``X − M·diag(ρ)``) makes it a
1/length² quantity — off from a curvature by a factor of the Laplacian — and no
real ``ρ`` can then satisfy ``Dλ = ρλ``. The spin transform minimizes ``E`` via
``A(ρ) λ = γ M λ`` for the *smallest* ``γ`` (``γ ≥ 0`` is the squared Dirac
residual; ``γ = 0`` iff ``ρ`` is realizable), then integrates the transformed
edges ``df̃ = λ̄ df λ`` by a per-coordinate cotan-Poisson solve.

:func:`rho_from_lambda` inverts this: the per-vertex ``ρ`` best explaining a
given ``λ``, so ``mesh → λ → ρ → mesh`` round trips.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..spectral.quaternion import from_vec3, qconj, qinv, qmul, to_vec3
from .mesh import triangle_areas
from .operators import cotan_laplacian, edge_cotan_weights, vertex_areas

__all__ = [
    "left_matrix", "dirac_assembly", "mass_quaternionic", "dirac_operator",
    "apply_dirac", "dirac_grad_rho", "rho_from_lambda", "dirac_residual",
    "spin_transform", "transformed_edges", "integrate_edges",
    "integrate_transformed_edges",
    "rho_from_sigma_linear", "reconstruct",
]


# --------------------------------------------------------------------------- #
# Quaternion <-> real 4x4 left-multiplication matrices
# --------------------------------------------------------------------------- #

def left_matrix(q: np.ndarray) -> np.ndarray:
    """Real left-multiplication matrices ``L(q)`` for ``p ↦ q·p`` ([w,x,y,z]).

    Accepts a single quaternion (returns ``(4,4)``) or a stack ``(...,4)``
    (returns ``(...,4,4)``). ``L(conj q) = L(q)ᵀ``, so a quaternionic-Hermitian
    operator assembles into a real *symmetric* matrix.
    """
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    M = np.stack([
        w, -x, -y, -z,
        x,  w, -z,  y,
        y,  z,  w, -x,
        z, -y,  x,  w,
    ], axis=-1)
    return M.reshape(q.shape[:-1] + (4, 4))


def _opposite_edges(V, F):
    """Per-face opposite-edge quaternions ``(e0, e1, e2)`` (pure imaginary),
    with ``e_p = V_next − V_prev`` so ``e0 + e1 + e2 = 0``."""
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    e0 = from_vec3(V[i2] - V[i1])    # opposite vertex 0
    e1 = from_vec3(V[i0] - V[i2])    # opposite vertex 1
    e2 = from_vec3(V[i1] - V[i0])    # opposite vertex 2
    return e0, e1, e2


# --------------------------------------------------------------------------- #
# Operator assembly
# --------------------------------------------------------------------------- #

def dirac_assembly(V, F):
    """Assemble the intrinsic Dirac operator ``X`` as a real symmetric sparse
    ``(4n, 4n)`` matrix (CSR)."""
    n = len(V)
    m = len(F)
    A = triangle_areas(V, F)
    e = _opposite_edges(V, F)
    coeff = -1.0 / (4.0 * A)                               # (m,)
    ab = np.arange(4)
    rows_all, cols_all, vals_all = [], [], []
    for p in range(3):
        for q in range(3):
            block = qmul(e[p], e[q]) * coeff[:, None]      # (m,4) quaternion X_pq
            B = left_matrix(block)                         # (m,4,4)
            r = (4 * F[:, p])[:, None, None] + ab[None, :, None]   # (m,4,1)
            c = (4 * F[:, q])[:, None, None] + ab[None, None, :]   # (m,1,4)
            rows_all.append(np.broadcast_to(r, (m, 4, 4)).ravel())
            cols_all.append(np.broadcast_to(c, (m, 4, 4)).ravel())
            vals_all.append(B.ravel())
    X = sp.coo_matrix(
        (np.concatenate(vals_all),
         (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(4 * n, 4 * n)).tocsr()
    return X


def mass_quaternionic(V, F):
    """Lumped quaternionic mass as a sparse diagonal ``(4n, 4n)`` (vertex area on
    each of the 4 quaternion components)."""
    a = vertex_areas(V, F)
    return sp.diags(np.repeat(a, 4))


def apply_dirac(V, F, lam):
    """Per-face Dirac image ``(Dλ)_f = −(1/2A_f) Σ_{p∈f} e_p λ_p``, shape
    ``(m, 4)``. Annihilates constants (``Σ_p e_p = 0``), and ``X = D* M_F D``."""
    A = triangle_areas(V, F)
    e = _opposite_edges(V, F)
    acc = sum(qmul(e[p], lam[F[:, p]]) for p in range(3))
    return -acc / (2.0 * A[:, None])


def dirac_operator(V, F, rho):
    """Return ``(A, M)``: the Dirac-energy operator for the per-vertex curvature
    change ``rho``, and the lumped quaternionic mass. Both real symmetric sparse
    ``(4n, 4n)``.

    ``A`` assembles ``E(λ) = Σ_f A_f |(Dλ)_f − ⅓Σ_p ρ_p λ_p|²``; with
    ``c_p = −e_p/(2A_f) − ρ_p/3`` and ``A_{pq} = Σ_f A_f c̄_p c_q``,

        A_{pq} = Σ_f [ −e_p e_q/(4A_f) − ρ_q e_p/6 + ρ_p e_q/6 + A_f ρ_p ρ_q/9 ]

    which is quaternionic-Hermitian, PSD, and reduces to ``X`` at ``ρ = 0``.
    """
    n, m = len(V), len(F)
    Af = triangle_areas(V, F)
    e = _opposite_edges(V, F)
    rp = [np.asarray(rho, dtype=float)[F[:, p]] for p in range(3)]
    ab = np.arange(4)
    rows_all, cols_all, vals_all = [], [], []
    for p in range(3):
        for q in range(3):
            block = -qmul(e[p], e[q]) / (4.0 * Af[:, None])
            block = block - rp[q][:, None] * e[p] / 6.0
            block = block + rp[p][:, None] * e[q] / 6.0
            block[:, 0] += Af * rp[p] * rp[q] / 9.0          # real (w) component
            B = left_matrix(block)                           # (m,4,4)
            r = (4 * F[:, p])[:, None, None] + ab[None, :, None]
            c = (4 * F[:, q])[:, None, None] + ab[None, None, :]
            rows_all.append(np.broadcast_to(r, (m, 4, 4)).ravel())
            cols_all.append(np.broadcast_to(c, (m, 4, 4)).ravel())
            vals_all.append(B.ravel())
    A = sp.coo_matrix(
        (np.concatenate(vals_all),
         (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(4 * n, 4 * n)).tocsr()
    return A, mass_quaternionic(V, F)


def dirac_grad_rho(V, F, rho, a, b):
    """``∂/∂ρ_v ⟨a, A(ρ) b⟩`` for quaternion fields ``a``, ``b`` — the ``∂A/∂ρ``
    contraction, returned as ``(n,)``.

    This is the one ρ-dependent piece of the adjoint of the eigensolve: with a
    cotangent ``q`` from ``(A − γM) q = Pᵀ p``, the gradient of a loss through
    ``λ`` is ``−dirac_grad_rho(V, F, rho, q, lam)``. The sign lives at the call
    site; this returns the plain derivative of the form.

    From :func:`dirac_operator`'s block, ``ρ`` enters ``A_{pq}`` three times, so

        ∂A_{pq}/∂ρ_p = +e_q/6 + (A_f ρ_q/9)·1̂
        ∂A_{pq}/∂ρ_q = −e_p/6 + (A_f ρ_p/9)·1̂

    (``1̂`` the real unit quaternion, since the ρ² term is real). Both are
    scattered, so the diagonal ``p = q`` — where the two linear terms cancel and
    only ``2A_f ρ_p/9`` survives — comes out right without a special case.

    ``ρ`` is per-vertex P1 here, so each block sees the endpoint values directly.
    A face-lumped stencil (``ρ_T`` = the mean over the face) would instead carry
    an extra ⅓ chain factor per vertex; dropping it leaves the gradient direction
    correct and the magnitude 3× off, which a cosine-only check will not catch.
    :func:`~tests.test_dirac.test_dirac_grad_rho_matches_fd` gates the magnitude.
    """
    n = len(V)
    Af = triangle_areas(V, F)
    e = _opposite_edges(V, F)
    rho = np.asarray(rho, dtype=float)
    rp = [rho[F[:, p]] for p in range(3)]
    out = np.zeros(n)
    for p in range(3):
        ap = a[F[:, p]]
        for q in range(3):
            bq = b[F[:, q]]
            d_p = e[q] / 6.0
            d_p[:, 0] += Af * rp[q] / 9.0
            d_q = -e[p] / 6.0
            d_q[:, 0] += Af * rp[p] / 9.0
            np.add.at(out, F[:, p], (ap * qmul(d_p, bq)).sum(1))
            np.add.at(out, F[:, q], (ap * qmul(d_q, bq)).sum(1))
    return out


def rho_from_lambda(V, F, lam):
    """Per-vertex ``ρ`` best explaining the spinor field ``lam`` — the encode
    side of the spin transform.

    Minimizes the same ``E(λ)`` over ``ρ`` with ``λ`` fixed, a linear least
    squares whose normal equations ``K ρ = b`` are sparse and SPD:

        K_{ip} = ⅓ Σ_{f ∋ i,p} A_f ⟨λ_p, λ_i⟩,   b_i = Σ_{f ∋ i} A_f ⟨(Dλ)_f, λ_i⟩.

    Unlike the old mass-coupled form there is no additive gauge: ``ρ`` is pinned
    (a constant shift is no longer absorbed by ``γ``).
    """
    n = len(V)
    Af = triangle_areas(V, F)
    w = apply_dirac(V, F, lam)
    rows, cols, vals = [], [], []
    b = np.zeros(n)
    for p in range(3):
        ip = F[:, p]
        np.add.at(b, ip, Af * (w * lam[ip]).sum(1))
        for q in range(3):
            iq = F[:, q]
            rows.append(ip)
            cols.append(iq)
            vals.append(Af * (lam[iq] * lam[ip]).sum(1) / 3.0)
    K = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsc()
    return spla.spsolve(K, b)


def dirac_residual(V, F, lam, rho):
    """Relative Dirac residual ``‖Dλ − ρλ‖ / ‖Dλ‖`` in the area-weighted norm.

    The acceptance gate for the operator: for a λ coming from a genuinely
    conformal correspondence this must go to zero under refinement.
    """
    Af = triangle_areas(V, F)
    w = apply_dirac(V, F, lam)
    rl = sum(np.asarray(rho, dtype=float)[F[:, p]][:, None] * lam[F[:, p]]
             for p in range(3)) / 3.0
    num = np.sqrt((Af[:, None] * (w - rl) ** 2).sum())
    den = np.sqrt((Af[:, None] * w ** 2).sum())
    return float(num / max(den, 1e-30))


# --------------------------------------------------------------------------- #
# Spin transform: smallest-|γ| eigenpair + gauge fix
# --------------------------------------------------------------------------- #

def _gauge_fix(lam):
    """Remove the global right-multiplication gauge by dividing every spinor by
    the mean spinor: a constant field maps to ``λ ≡ 1`` exactly."""
    mean = lam.mean(0)
    return qmul(lam, qinv(mean)[None, :])


def spin_transform(V, F, rho, sigma=1e-6, k=6):
    """Solve ``A(ρ) λ = γ M λ`` for the *smallest* ``γ`` via shift-invert near
    ``sigma``. Returns ``(lam (n,4), gamma)``, ``lam`` gauge-fixed so a constant
    solution becomes ``λ ≡ 1``.

    ``A`` is PSD, so ``γ ≥ 0`` *is* the squared Dirac residual and the bottom
    eigenvector is unambiguously the wanted λ — no |γ|-branch ambiguity. Note
    each quaternionic eigenvalue appears with multiplicity 4 (the right ℍ
    action), so ``k`` must exceed 4 to see past the bottom cluster.
    """
    A, M = dirac_operator(V, F, rho)
    k = min(k, A.shape[0] - 2)
    gammas, vecs = spla.eigsh(A, k=k, M=M, sigma=sigma, which="LM")
    j = int(np.argmin(gammas))
    lam = vecs[:, j].reshape(-1, 4)
    return _gauge_fix(lam), float(gammas[j])


# --------------------------------------------------------------------------- #
# Transformed edges + integration (cotan-Poisson)
# --------------------------------------------------------------------------- #

def transformed_edges(V, lam, i, j):
    """Transformed edge vectors ``ẽ_{i→j} = ∫₀¹ λ̄(t) e_ij λ(t) dt`` with
    linearly-interpolated ``λ`` (weights ⅓,⅙,⅙,⅓). Returns ℝ³ per edge.

    ``i``, ``j`` are arrays of endpoint indices; for ``λ ≡ 1`` this returns
    exactly ``V_j − V_i``.
    """
    e = from_vec3(V[j] - V[i])
    li, lj = lam[i], lam[j]

    def s(a, b):  # ā · e · b
        return qmul(qmul(qconj(a), e), b)

    et = (1.0 / 3.0) * s(li, li) + (1.0 / 6.0) * (s(li, lj) + s(lj, li)) \
        + (1.0 / 3.0) * s(lj, lj)
    return to_vec3(et)


def integrate_edges(V, F, lam):
    """Integrate ``df̃ = λ̄ df λ`` to vertex positions by the per-coordinate
    cotan-Poisson solve ``L x = b``, ``b_i = −Σ_j w_ij ẽ_{i→j}``. Translation is
    the gauge (mean removed)."""
    edges, _ = edge_cotan_weights(V, F)
    et = transformed_edges(V, lam, edges[:, 0], edges[:, 1])   # (E,3), i->j
    return integrate_transformed_edges(V, F, et)


def integrate_transformed_edges(V, F, et):
    """Poisson-integrate an arbitrary per-edge field ``et`` ``(E,3)`` to vertex
    positions (mean removed).

    Split out of :func:`integrate_edges` because this stage is *linear* in
    ``et``, which is what lets the adjoint test it by a dot-product identity
    rather than finite differences. ``X = L⁺ Gᵀ W et``: the pinned solve plus the
    mean removal together realise the mean-zero pseudoinverse, so the composite
    is symmetric and is its own adjoint up to ``G`` and ``W``.
    """
    L, _ = cotan_laplacian(V, F)
    edges, w = edge_cotan_weights(V, F)
    i, j = edges[:, 0], edges[:, 1]
    b = np.zeros((len(V), 3))
    np.add.at(b, i, -w[:, None] * et)
    np.add.at(b, j, w[:, None] * et)              # ẽ_{j→i} = −ẽ_{i→j}
    x = _poisson_solve(L, b)
    return x - x.mean(0)


def _poisson_solve(L, b):
    """Solve the singular SPD system ``L x = b`` (nullspace = constants) exactly
    by pinning vertex 0, for each of the (possibly several) RHS columns."""
    n = L.shape[0]
    keep = np.arange(1, n)
    Lrr = L[keep][:, keep].tocsc()
    solve = spla.factorized(Lrr)
    b = np.atleast_2d(b.T).T if b.ndim == 1 else b
    x = np.zeros((n, b.shape[1]))
    for c in range(b.shape[1]):
        x[keep, c] = solve(b[keep, c])
    return x


# --------------------------------------------------------------------------- #
# σ̂ -> mesh
# --------------------------------------------------------------------------- #

def sigma_to_rhodot_mult(l):
    """Per-degree multiplier from the Hopf potential ``σ̂`` to the Dirac
    curvature drive: ``ρ_{ℓm} = −ℓ(ℓ+1) σ̂_{ℓm}``.

    Now *derived*, not fitted. A radial perturbation ``r = 1 + ĥ Y_{ℓm}`` has
    ``σ̂ = −½ĥ`` (degree-independent, measured) and geometric curvature change
    ``ρ = ½ℓ(ℓ+1)ĥ``, hence ``−ℓ(ℓ+1)``.

    The old value ``−(ℓ(ℓ+1))²`` was calibrated against the mass-coupled
    operator, whose ``ρ`` was off by a factor of the Laplacian ``ℓ(ℓ+1)``; with
    :func:`dirac_operator` fixed, ``ρ`` is the true mean-curvature half-density
    change. Prefer :func:`rho_from_lambda` over this linearization wherever an
    actual mesh is available — it is exact rather than near-sphere.
    """
    return -(l * (l + 1.0))


def _synth_perdeg(coeffs, theta, phi, mult):
    """Evaluate ``Σ_{ℓ≥2,m} mult(ℓ) coeffs_{ℓm} Y_{ℓm}(θ,φ)`` (real)."""
    from scipy.special import sph_harm_y
    Lmax = coeffs.shape[0] - 1
    out = np.zeros(theta.shape[0])
    for l in range(2, Lmax + 1):
        fac = mult(l)
        for m in range(-l, l + 1):
            c = coeffs[l, l + m]
            if c == 0:
                continue
            out += np.real(fac * c * sph_harm_y(l, m, theta, phi))
    return out


def _sphere_angles(V0):
    r = np.linalg.norm(V0, axis=1)
    theta = np.arccos(np.clip(V0[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(V0[:, 1], V0[:, 0])
    return theta, phi


def rho_from_sigma_linear(sigma_hat, V0):
    """Initialize the per-vertex change-of-curvature ``ρ`` from ``σ̂`` via the
    calibrated linearization ``ρ̇_{ℓm} = −(ℓ(ℓ+1))² σ̂_{ℓm}`` evaluated at the
    base-sphere vertices ``V0`` (R=1)."""
    theta, phi = _sphere_angles(V0)
    return _synth_perdeg(sigma_hat, theta, phi, sigma_to_rhodot_mult)


def reconstruct(base_V, base_F, sigma_hat, encode=None, iters=4, sigma=1e-6,
                damp=0.3):
    """Decode ``σ̂`` into a mesh on the base-sphere connectivity. ``ρ`` is
    initialized from ``σ̂`` via the calibrated linearization and, if an
    ``encode`` callable is supplied (``mesh → σ̂``, Session 4), refined by a
    *damped* residual loop so ``encode(decode(σ̂)) ≈ σ̂``.

    The damping (``damp`` ∈ (0,1]) keeps the loop a contraction: the residual
    drive ``−(ℓ(ℓ+1))² · res`` amplifies high-ℓ error, so an undamped step can
    diverge. The best decode is kept (lowest spectral residual). Returns
    ``(V, F)``.
    """
    theta, phi = _sphere_angles(base_V)
    rho = rho_from_sigma_linear(sigma_hat, base_V)
    lam, _ = spin_transform(base_V, base_F, rho, sigma=sigma)
    V = integrate_edges(base_V, base_F, lam)
    if encode is None:
        return V, base_F

    best_V, best_err = V, _spec_err(sigma_hat, encode(V, base_F))
    for _ in range(iters - 1):
        res = sigma_hat - encode(V, base_F)
        rho = rho + damp * _synth_perdeg(res, theta, phi, sigma_to_rhodot_mult)
        lam, _ = spin_transform(base_V, base_F, rho, sigma=sigma)
        V = integrate_edges(base_V, base_F, lam)
        err = _spec_err(sigma_hat, encode(V, base_F))
        if err < best_err:
            best_V, best_err = V, err
    return best_V, base_F


def _spec_err(target, got):
    return float(np.linalg.norm(got - target) / max(np.linalg.norm(target), 1e-30))

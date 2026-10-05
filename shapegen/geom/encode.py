"""Spherical-harmonic primitives shared by the encoders: the real SH basis, a
mass-weighted projection, the Procrustes shape loss, and the per-band
preconditioner.

Two traps the optimiser in :func:`shapegen.geom.spectral_adjoint.encode_spectral`
inherits, both silent:

* **Tolerances.** The Procrustes ``J`` is a normalised squared error (~3e-5 at
  a 0.6% shape error), so L-BFGS-B's default ``pgtol`` of 1e-5 is met at the
  start and it returns ``nit=0`` having taken no step. Pin ``ftol``/``gtol``
  far below the problem's scale and check ``niter``.
* **Warm starts across bandwidths** are not worth it: with :func:`band_precond`
  a cold start per band reaches the same optimum at the same total cost.
"""

from __future__ import annotations

import numpy as np


__all__ = ["real_sh_basis", "n_coeffs", "band_index", "procrustes",
           "band_precond", "sh_project"]


def n_coeffs(L: int) -> int:
    """``(L+1)²`` — the number of real SH coefficients up to degree ``L``."""
    return (L + 1) ** 2


def band_index(L: int) -> np.ndarray:
    """Degree ``ℓ`` of each coefficient, in the layout :func:`real_sh_basis` uses."""
    return np.concatenate([np.full(2 * l + 1, l) for l in range(L + 1)])


def real_sh_basis(V: np.ndarray, L: int) -> np.ndarray:
    """Real orthonormal SH basis evaluated at the sphere directions ``V``.

    ``(n, (L+1)²)``, ordered ``ℓ`` ascending and ``m`` from ``−ℓ`` to ``+ℓ``, so
    the first ``(L'+1)²`` **columns are exactly the bandwidth-L' basis** — build
    this once at the largest ``L`` and slice for everything smaller.

    ``V`` need not be normalised; only the direction is used.
    """
    # Divide out the radius rather than trusting the caller to normalise; the
    # clip guards acos against |z/r| = 1 + 1e-16.
    from math import sqrt

    from ..spectral.swsh import wigner_d_ladder

    r = np.linalg.norm(V, axis=1)
    theta = np.arccos(np.clip(V[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(V[:, 1], V[:, 0])
    # Separable: Y_lm = c·N_l·d^l_{m0}(θ)·{cos, sin}(|m|φ), with c = √2·(−1)^m for
    # m ≠ 0 (unit norm, Condon-Shortley phase cancelled) -- the standard real basis,
    # equal to the complex route (√2·(−1)^m Re/Im Y_l^|m|) to 1e-15. One Wigner-d
    # ladder per m; 0.19 s at L=79 on 10242 points against 1.5 s through the
    # complex basis. Built transposed so every write is a contiguous row, and
    # returned as that array's transpose (Fortran order, which BLAS takes as is).
    out = np.empty(((L + 1) ** 2, theta.size))
    for m in range(L + 1):
        ls, d = wigner_d_ladder(L, m, 0, theta)
        l = ls.astype(int)
        N = np.sqrt((2 * l + 1) / (4 * np.pi))[:, None]
        if m == 0:
            out[l * l + l] = N * d
        else:
            A = (sqrt(2.0) * (-1) ** m) * N * d
            out[l * l + l + m] = A * np.cos(m * phi)      # column order is load-bearing:
            out[l * l + l - m] = A * np.sin(m * phi)      # callers slice to smaller bands
    return out.T


def sh_project(rho: np.ndarray, Y: np.ndarray, mass: np.ndarray | None = None):
    """SH coefficients of a per-vertex field on the sphere.

    With ``mass`` (the sphere's lumped vertex areas) this is a **Galerkin
    projection in the mesh's own L²**::

        f_L = argmin_g  Σ_i M_i (f_i − g_i)²  =  Y (YᵀMY)⁻¹ YᵀM f

    Without it, plain unweighted least squares. Trap: on a non-uniform sphere
    an unweighted fit weights by triangulation density rather than area
    (measured 3.9x worse at L=8 on a bottle).

    Trap: do not replace this with Gauss-Legendre quadrature. ρ concentrates
    on caps (~0.03 rad) finer than an affordable GL grid (0.043 rad at
    L_quad=72), and ‖ρ‖ through such grids swings 3.6-6.7 against a mesh
    truth of 5.9; the conformal mesh grades itself to the geometry.
    """
    if mass is None:
        # rcond=None keeps numpy's future-proof default (machine-eps cutoff).
        return np.linalg.lstsq(Y, rho, rcond=None)[0]
    Yw = Y * np.asarray(mass, dtype=float)[:, None]
    G = Y.T @ Yw
    return np.linalg.solve(G + 1e-12 * np.eye(G.shape[0]), Yw.T @ rho)


# --------------------------------------------------------------------------- #
# The objective                                                               #
# --------------------------------------------------------------------------- #

def band_precond(L: int, alpha: float = 0.75) -> np.ndarray:
    """Per-coefficient step scaling by SH degree: ``[ℓ(ℓ+1)]^alpha``, normalised.

    The optimiser is reparameterised as ``ĉ = d ⊙ u`` and run on ``u``. This is a
    pure change of variables — the optimum is identical — but it fixes a badly
    scaled Hessian, and that is worth roughly an order of magnitude of work.

    A unit change in a high-ℓ coefficient of ρ moves the surface far less than a
    low-ℓ one (linearised, ``σ̂_{ℓm} = 2R²/[ℓ(ℓ+1)] · ρ̇_{ℓm}``), so J's curvature
    falls off with degree and L-BFGS crawls along the flat high-ℓ directions.
    Probing the diagonal curvature per band gives ``∂²J/∂c² ~ [ℓ(ℓ+1)]^{-1.5}``
    — *not* the ``^-2`` the linearisation alone suggests — and equalising it
    wants ``d ∝ [ℓ(ℓ+1)]^{0.75}``, hence the default.

    Measured (bottle, L=40, cold start): alpha=0.75 reaches alpha=0's final
    error in 8.5x fewer evaluations. The optimum is flat over 0.5-0.75 and
    degrades above ~0.875. (A probed per-band Jacobi preconditioner lost to
    no preconditioning at L=16.)
    """
    b = band_index(L)
    # Clamp l(l+1) at 2 so degrees 0 and 1 share degree 1's scale instead of
    # producing 0 (which would make d = 0 there and freeze those coefficients
    # entirely). They carry the overall size and centring of rho, so freezing
    # them would silently constrain the fit.
    d = np.maximum(b * (b + 1.0), 2.0) ** alpha
    # Normalise so the largest scale is 1. Only *ratios* between coefficients
    # matter for conditioning; fixing the top end keeps L-BFGS's internal step
    # bounds and the ftol/gtol thresholds meaning the same thing across L and
    # across alpha, so runs stay comparable.
    return d / d.max()


def procrustes(target: np.ndarray, X: np.ndarray):
    """Squared relative Procrustes error of ``X`` against ``target``, and
    ``dJ/dX``. Returns ``(J, grad)``; the reported error is ``√J``.

    Optimal rotation, translation and scale are solved for and then **held
    fixed** while differentiating — the envelope theorem, since they are
    themselves minimisers. So the SVD never appears in the gradient.

    Invariant under ``X → sQX + t``, which is what makes ``J`` a legal objective
    for the adjoint: the decode determines the immersion only up to a similarity,
    and a loss that could see the difference would be differentiating a gauge.
    """
    # Translation is removed by centring both clouds; this is the optimal
    # translation, not an approximation of it.
    A = target - target.mean(0)
    B = X - X.mean(0)

    # Kabsch: the R maximising tr(Rᵀ BᵀA) -- equivalently minimising
    # ||A - BR|| -- is U Vt from the SVD of BᵀA. (Write BᵀA = U S Vt and
    # Z = Vt Rᵀ U, orthogonal; then tr(Rᵀ BᵀA) = tr(Z S) = sum_i z_ii s_i, which
    # is maximal at Z = I.)
    U, _, Vt = np.linalg.svd(B.T @ A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        # det = -1 is a reflection. It fits better than any rotation whenever the
        # decode comes out mirrored, so without this the error would be silently
        # reported against a mirror image -- and the spin transform *can* produce
        # the opposite orientation. Flipping the least-significant singular
        # direction gives the best proper rotation.
        U[:, -1] *= -1
        R = U @ Vt

    BR = B @ R
    # Optimal uniform scale, given R: the least-squares coefficient of BR in A.
    # ||BR|| == ||B|| since R is orthogonal, so denom is just ||B||^2.
    denom = float((BR * BR).sum())
    s = float((A * BR).sum() / max(denom, 1e-300))

    # Normalise by ||A||^2 so J is scale-free and comparable across meshes; it
    # also puts J on a fixed numeric scale (~1e-5 at a 0.6% error), which is
    # why the encoders override L-BFGS's default tolerances.
    c = float((A * A).sum())
    E = A - s * BR
    J = float((E * E).sum() / c)

    # Gradient with (s, R, translation) FROZEN at their optimal values. Legal by
    # the envelope theorem: they are minimisers, so dJ/ds = dJ/dR = 0 and their
    # variation contributes nothing to the total derivative. This is what keeps
    # the SVD out of the gradient -- differentiating an SVD is both ugly and
    # ill-defined when singular values collide.
    dB = (-2.0 * s / c) * (E @ R.T)
    # Adjoint of the centring B = X - mean(X). Mean removal is a symmetric
    # projection, so it is its own adjoint; without this the caller would get a
    # gradient with a spurious constant mode.
    return J, dB - dB.mean(0)


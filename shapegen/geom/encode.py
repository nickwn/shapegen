"""Spherical-harmonic primitives shared by the encoders: the real SH basis (and
its evaluation without building it), a mass-weighted projection, and the
Procrustes shape error."""

from __future__ import annotations

import numpy as np


__all__ = ["real_sh_basis", "eval_real_sh", "n_coeffs", "band_index", "procrustes", "sh_project"]


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


def eval_real_sh(V: np.ndarray, coeffs: np.ndarray, L: int, chunk: int = 32768) -> np.ndarray:
    """``real_sh_basis(V, L) @ coeffs`` without building the basis: per order ``m``
    the Legendre ladder ``d^ℓ_{m0}(θ)`` (seeded ``∝ sin^m θ``, then
    ``d^{ℓ+1} = [(2ℓ+1) cos θ d^ℓ − √(ℓ²−m²) d^{ℓ−1}] / √((ℓ+1)²−m²)``),
    contracted with that order's coefficients, times ``cos/sin mφ`` (by complex
    rotation). ``coeffs`` is ``((L+1)², k)`` or ``((L+1)²,)``. At L=79 on 32k
    points ~0.2 s against 0.7 s through the basis."""
    c = np.asarray(coeffs, float)
    vec = c.ndim == 1
    c = c[:, None] if vec else c
    r = np.linalg.norm(V, axis=1)
    out = np.empty((len(V), c.shape[1]))
    li = np.arange(L + 1)
    Nl = np.sqrt((2 * li + 1) / (4 * np.pi))
    for s0 in range(0, len(V), chunk):
        P = V[s0:s0 + chunk]
        x = np.clip(P[:, 2] / r[s0:s0 + chunk], -1.0, 1.0)
        sn = np.sqrt(np.maximum(1.0 - x * x, 0.0))
        rho = np.hypot(P[:, 0], P[:, 1])
        e1 = np.where(rho > 0, (P[:, 0] + 1j * P[:, 1]) / np.where(rho > 0, rho, 1.0), 1.0)   # e^{iφ}
        em = np.ones(len(P), complex)
        acc = np.zeros((len(P), c.shape[1]))
        seed = np.ones(len(P))
        d = np.empty((L + 1, len(P)))
        for m in range(L + 1):
            if m > 0:
                seed = seed * (-sn) * np.sqrt((2 * m - 1) / (2 * m))
                em = em * e1
            d[0] = seed
            if m < L:
                d[1] = np.sqrt(2 * m + 1.0) * x * seed
            for i in range(1, L - m):
                l = m + i
                d[i + 1] = ((2 * l + 1) / np.sqrt((l + 1) ** 2 - m * m)) * x * d[i] \
                    - (np.sqrt(l * l - m * m) / np.sqrt((l + 1) ** 2 - m * m)) * d[i - 1]
            ls = li[m:]
            dm = d[:L + 1 - m].T
            if m == 0:
                acc += dm @ (Nl[:, None] * c[ls * ls + ls])
            else:
                A = (np.sqrt(2.0) * (-1) ** m * Nl[ls])[:, None]
                acc += em.real[:, None] * (dm @ (A * c[ls * ls + ls + m])) + em.imag[:, None] * (dm @ (A * c[ls * ls + ls - m]))
        out[s0:s0 + chunk] = acc
    return out[:, 0] if vec else out


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


def procrustes(target: np.ndarray, X: np.ndarray) -> float:
    """Squared relative Procrustes error ``J`` of ``X`` against ``target``, after the
    optimal rotation, translation and scale; the reported error is ``√J``.
    Invariant under ``X → sQX + t``: the decode determines the immersion only up
    to a similarity."""
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

    # Normalised by ||A||^2 so J is scale-free and comparable across meshes.
    E = A - s * BR
    return float((E * E).sum() / (A * A).sum())


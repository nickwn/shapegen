"""Lifting the traced grid to ℝ³.

:func:`lift_spectral` is the lift (and what :func:`shapegen.codec.decode`
calls): the spectral spinor gives the surface's differential
``df̃ = λ̄ df₀ λ``, the least squares ``∫|∇f − ω|²`` is solved in spherical
harmonics (:class:`SpectralLift`), and the
surface is read off at the requested directions. No mesh and no edge graph.

:func:`frame_quaternion` / :func:`spinor_quaternion` are the
spinor-to-quaternion dictionary it rests on.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..spectral.quaternion import from_vec3, qconj, qmul, to_vec3

__all__ = ["frame_quaternion", "spinor_quaternion", "SpectralLift", "lift_spectral"]


# --------------------------------------------------------------------------- #
# The spinor-to-quaternion dictionary                                         #
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

    **Trap: the round sphere cannot pin this.** Any ``A(x)·raw·B(x)`` with
    ``A raw₀ B = 1`` agrees where λ is constant, so a wrong map passes a
    round-sphere check to machine precision and is off at first order in ρ.
    The discriminating gates are in ``tests/test_quad_lift.py``: the kernel's
    quaternionic gauge must act as a **right** constant (a rigid motion of
    the output), and ``λ_spec⁻¹ λ_mesh`` (against the reference solver
    :func:`shapegen.geom.dirac.spin_transform`) must be constant to its
    discretisation error off the sphere.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    a = (theta - 0.5 * np.pi) / 2.0
    ca, sa = np.cos(a), np.sin(a)
    cp, sp = np.cos(phi / 2.0), np.sin(phi / 2.0)
    # q_z(φ) · q_y(θ − π/2)
    return np.stack([cp * ca, -sp * sa, cp * sa, sp * ca], axis=-1)


def spinor_quaternion(res, dirs: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Evaluate the spectral spinor as a genuine quaternion field ``(n,4)`` at
    arbitrary unit directions: ``R · (ψ1 + ψ̄2 j)``, :func:`frame_quaternion`
    applied on the left of :func:`~shapegen.spectral.swsh.half_eval_basis`.

    **This is single-valued**, even though neither factor is.
    ``half_eval_basis`` warns that its values are antiperiodic in ``φ``
    (half-integer ``m``) — conjugation keeps that — and ``R`` is antiperiodic
    for the same reason, so the product is periodic.

    Chunked because the basis is ``(npts, nmodes)`` and ``nmodes`` is 1640 at
    ``Ls = 39.5`` (L = 32 with the potential's padding) — forming it for a whole traced
    mesh at once would be gigabytes for no gain.
    """
    from ..spectral import swsh as sw

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


class SpectralLift:
    """The lift ``v → f̂`` as a continuum least squares, solved spectrally.

    Minimises ``∫_S² |∇f − ω|² dA`` with ``ω = Im(λ̄ df₀ λ)``, the ℝ³-valued 1-form the spin
    transform gives. Round-sphere harmonics have ``∫∇Y_k·∇Y_k' = ℓ(ℓ+1)δ``, so the
    normal equations are diagonal:

        f̂_k = (1/ℓ(ℓ+1)) ∫ [∂θY_k·ω(e_θ) + (1/sinθ)∂φY_k·ω(e_φ)] dA,   ℓ ≥ 1

    one weighted sum per coefficient on a Gauss-Legendre grid, with separable
    transforms (per-m Wigner-d in θ, trig in φ). ``f̂`` is real SH in
    :func:`~shapegen.geom.encode.real_sh_basis`'s layout, ``(K, 3)``.

    **Exact.** ω is quadratic in λ (band ``Ls``), so ``f`` has band ``2·Ls``
    exactly and the default ``Lf = 2·Ls`` (79 at L=32) keeps all of it; the
    integrand has degree ``Lf + 2·Ls``, which the grid integrates exactly.
    Trap: a grid sized for λ alone aliases (~1% error). Setup is a few MB.

    ``forward(v) -> (f̂, cache)``.
    """

    def __init__(self, Ls: float, Lf: int | None = None):
        from math import sqrt

        from ..geom.dirac_spectral import SpinorTransform
        from ..spectral import swsh as sw

        Lf = int(round(2 * Ls)) if Lf is None else int(Lf)
        self.Ls, self.Lf = Ls, Lf
        B = int(round(Lf + 2 * Ls))                   # degree integrated exactly
        nth, nph = int(np.ceil((B + 1) / 2)), B + 1
        x, w = np.polynomial.legendre.leggauss(nth)
        th = np.arccos(x)
        ph = 2 * np.pi * np.arange(nph) / nph
        self.th, self.ph, self.nth, self.nph = th, ph, nth, nph
        self.Wj = w * (2 * np.pi / nph)
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        self.R = frame_quaternion(TH, PH)
        st, ct, sp_, cp = np.sin(TH), np.cos(TH), np.sin(PH), np.cos(PH)
        self.eth = from_vec3(np.stack([ct * cp, ct * sp_, -st], -1))
        self.eph = from_vec3(np.stack([-sp_, cp, np.zeros_like(sp_)], -1))
        self.tf = SpinorTransform(Ls, th, nph)
        self.nmodes = self.tf.n
        # real SH of band Lf, separably: Y_k = Θ_k(θ) · trig_col(k)(φ)
        K = (Lf + 1) ** 2
        ncol = 2 * Lf + 1
        Tph = np.zeros((nph, ncol))
        Tph[:, 0] = 1.0
        for m in range(1, Lf + 1):
            Tph[:, 2 * m - 1] = np.cos(m * ph)
            Tph[:, 2 * m] = np.sin(m * ph)
        Th, dTh = np.zeros((nth, K)), np.zeros((nth, K))
        col, pcol = np.zeros(K, dtype=np.intp), np.zeros(K, dtype=np.intp)
        sig, ell = np.zeros(K), np.zeros(K)
        for m in range(0, Lf + 1):
            ls, d0 = sw.wigner_d_ladder(Lf, m, 0, th)
            _, dp = sw.wigner_d_ladder(Lf, m, 1, th)
            _, dm = sw.wigner_d_ladder(Lf, m, -1, th)
            for i, l in enumerate(ls):
                l = int(round(l))
                N = sqrt((2 * l + 1) / (4 * np.pi))
                j = l - max(m, 1)
                # ∂θ(N d^l_{m0}) = ½√(l(l+1)) N (d^l_{m,−1} − d^l_{m,1}); pinned by FD
                dth = 0.5 * sqrt(l * (l + 1)) * N * (dm[j] - dp[j]) if l >= 1 else 0.0 * th
                for mm in ((0,) if m == 0 else (m, -m)):
                    k = l * l + l + mm
                    c = 1.0 if m == 0 else sqrt(2) * (-1) ** m
                    Th[:, k], dTh[:, k], ell[k] = c * N * d0[i], c * dth, l
                    if m == 0:
                        col[k] = pcol[k] = 0
                    elif mm > 0:                  # cos(mφ): ∂φ -> −m sin(mφ)
                        col[k], pcol[k], sig[k] = 2 * m - 1, 2 * m, -m
                    else:                         # sin(mφ): ∂φ -> +m cos(mφ)
                        col[k], pcol[k], sig[k] = 2 * m, 2 * m - 1, m
        self.K, self.Tph, self.Th = K, Tph, Th
        self.col, self.pcol = col, pcol
        self.inv_ll = np.where(ell > 0, 1.0 / np.maximum(ell * (ell + 1), 1), 0.0)
        self.A_th = self.Wj[:, None] * dTh
        self.A_ph = self.Wj[:, None] * (Th / np.sin(th)[:, None]) * sig[None, :]

    def spinor(self, v):
        """The spinor on the grid: the dictionary ``λ = R·(ψ1 + conj(ψ2) j)``."""
        k = self.nmodes
        a, b = self.tf.synth(+0.5, v[:k]), self.tf.synth(-0.5, v[k:])
        return qmul(self.R, np.stack([a.real, a.imag, b.real, -b.imag], -1))

    def forward(self, v):
        lam = self.spinor(v)
        lb = qconj(lam)
        wth = to_vec3(qmul(qmul(lb, self.eth), lam))
        wph = to_vec3(qmul(qmul(lb, self.eph), lam))
        Fth = np.einsum("jpc,pt->jtc", wth, self.Tph)
        Fph = np.einsum("jpc,pt->jtc", wph, self.Tph)
        rhs = (np.einsum("jk,jkc->kc", self.A_th, Fth[:, self.col])
               + np.einsum("jk,jkc->kc", self.A_ph, Fph[:, self.pcol]))
        return rhs * self.inv_ll[:, None], {"lam": lam}


@lru_cache(maxsize=2)
def _spectral_lift(Ls: float):
    return SpectralLift(Ls)


def lift_spectral(dirs: np.ndarray, rho_hat: np.ndarray, L: int, res=None,
                  chunk: int = 4000):
    """Positions in ℝ³ at unit directions ``dirs`` ``(n,3)`` for the surface
    encoded by ``rho_hat`` (real-SH ρ coefficients at bandwidth ``L``).

    Solves ``∫_S² |∇f − ω|² dA`` exactly in spherical harmonics
    (:class:`SpectralLift`), ``ω`` the
    spinor's ``df̃``, then evaluates ``f`` at ``dirs`` in chunks of ``chunk``.
    ``f`` is band-limited at ``2·Ls`` (79 at L=32), so the result does not
    depend on how ``dirs`` are sampled or connected. ~1.8 s at L=32 for a
    20k-face grid.

    Returns ``(P3, info)``: ``P3`` is ``(n,3)``, centred on its mean;
    ``info`` carries ``res``, ``gamma`` and ``f_hat`` (the surface's real-SH
    coefficients, ``(K, 3)``). Pass ``res`` (a
    :func:`~shapegen.geom.dirac_spectral.solve_spinor` result) to reuse an
    already-solved spinor.
    """
    from ..geom.dirac_spectral import (potential_band_extra, solve_spinor,
                                       spinor_band)
    from ..geom.encode import real_sh_basis

    if res is None:
        res = solve_spinor(np.asarray(rho_hat, float), L,
                           Ls=spinor_band(L, potential_band_extra(L)))
    lift = _spectral_lift(float(res.Ls))
    fh, _ = lift.forward(np.r_[res.psi1, res.psi2])
    d = np.asarray(dirs, dtype=float)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    X = np.empty((len(d), 3))
    for i in range(0, len(d), chunk):
        X[i:i + chunk] = real_sh_basis(d[i:i + chunk], lift.Lf) @ fh
    return X - X.mean(0), {"res": res, "gamma": float(res.gamma), "f_hat": fh}

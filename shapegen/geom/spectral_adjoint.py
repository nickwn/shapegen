"""The spectral encoder: ρ̂ → spectral spinor → continuum lift → Procrustes shape
error, with its exact gradient back to ρ̂. This is what
:func:`shapegen.codec.encode` runs by default.

Nothing here depends on a mesh: the spinor is spectral
(:class:`SpectralSpinor`), the surface is solved in spherical harmonics
(:class:`SpectralLift`), and the loss compares SH coefficients
(:func:`coef_procrustes`). The target mesh enters once, through
:func:`project_positions`, so codes do not absorb any decode discretisation.

Stages and their adjoints (finite-difference tests in
``tests/test_spectral_adjoint.py``):

1. ``ρ̂ → G`` -- linear, diagonal in ``δρ`` on
   :func:`~shapegen.geom.dirac_spectral.coupling_grid`.
2. ``G → v``, the kernel pair (:func:`~shapegen.geom.dirac_spectral.solve_spinor`).
   For a loss invariant under the pair's quaternionic gauge (Procrustes is:
   the gauge is a rigid rotation of the output), with ``dJ = Re(g_v* dv)``::

       dJ = −Re[(Gv)* dG z + (Gz)* dG v],   z = P (G*G − γ²)⁻¹ P g_v,   P = I − VV*

   ``z`` reuses the forward's factorisation through ``z ← P A⁻¹ P (g_v + γ² z)``,
   which contracts at ``γ²/σ₃²`` (~5e-4 on faces), so one sweep suffices.
3. ``v → f̂`` -- :class:`SpectralLift`, linear in the quadratic form ``λ̄ df₀ λ``.
4. ``f̂ → J`` -- :func:`coef_procrustes`.

**Measured** (FaceScape heads, L=32, on the eigen sphere of
:func:`shapegen.geom.eigen_sphere.conformal_sphere`): Procrustes 0.0050 /
0.0045, converged in 75-100 evaluations; ~0.5 s per evaluation (1.85 s before
the separable coupling, Schur solve and single Neumann sweep, 2026-09-27).
It replaced graph-based lifts and a mesh adjoint encoder.
"""

from __future__ import annotations

import time

import numpy as np

from ..spectral.quaternion import from_vec3, qconj, qmul, to_vec3

__all__ = ["SpectralLift", "SpectralSpinor", "coef_procrustes",
           "project_positions", "encode_spectral"]


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

    ``forward(v) -> (f̂, cache)``; ``backward(dJ/df̂, cache) -> g_v`` with
    ``dJ = Re(g_v* dv)``.
    """

    def __init__(self, Ls: float, Lf: int | None = None):
        from math import sqrt

        from ..quad.lift import frame_quaternion
        from ..spectral import swsh as sw
        from .dirac_spectral import SpinorTransform

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
        self.S = np.zeros((K, ncol))
        self.S[np.arange(K), col] = 1.0
        self.Sp = np.zeros((K, ncol))
        self.Sp[np.arange(K), pcol] = 1.0

    # spinor on the grid (the dictionary λ = R·(ψ1 + conj(ψ2) j)) and its adjoint
    def spinor(self, v):
        k = self.nmodes
        a, b = self.tf.synth(+0.5, v[:k]), self.tf.synth(-0.5, v[k:])
        return qmul(self.R, np.stack([a.real, a.imag, b.real, -b.imag], -1))

    def spinor_adjoint(self, gl):
        graw = qmul(qconj(self.R), gl)
        return np.r_[self.tf.adjoint(+0.5, graw[..., 0] + 1j * graw[..., 1]),
                     self.tf.adjoint(-0.5, graw[..., 2] - 1j * graw[..., 3])]

    @staticmethod
    def _scatter(A, g, S):
        """``out[j, t, c] = Σ_k A[j, k] g[k, c] S[k, t]`` as one matmul."""
        M = (A[:, None, :] * g.T[None, :, :]).reshape(-1, A.shape[1])
        return (M @ S).reshape(A.shape[0], g.shape[1], -1).transpose(0, 2, 1)

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

    def backward(self, gf, cache):
        """``d/dv`` (``dJ = Re(g_v* dv)``) from ``dJ/df̂`` ``(K, 3)``."""
        g = gf * self.inv_ll[:, None]
        gth = np.einsum("jtc,pt->jpc", self._scatter(self.A_th, g, self.S), self.Tph)
        gph = np.einsum("jtc,pt->jpc", self._scatter(self.A_ph, g, self.Sp), self.Tph)
        lam = cache["lam"]
        gl = (-2.0 * qmul(qmul(self.eth, lam), from_vec3(gth))
              - 2.0 * qmul(qmul(self.eph, lam), from_vec3(gph)))
        return self.spinor_adjoint(gl)


def coef_procrustes(T: np.ndarray, F: np.ndarray):
    """Similarity Procrustes between real-SH coefficient matrices ``(K, 3)``, the
    ℓ = 0 row dropped (translation lives there -- do not use
    :func:`~shapegen.geom.encode.procrustes`, which subtracts a row mean). By
    Parseval this is the continuum per-point Procrustes w.r.t. sphere area.
    Returns ``(J, dJ/dF)``."""
    A, B = T[1:], F[1:len(T)]
    U, _, Vt = np.linalg.svd(B.T @ A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    BR = B @ R
    s = float((A * BR).sum() / max(float((BR * BR).sum()), 1e-300))
    c = float((A * A).sum())
    E = A - s * BR
    g = np.zeros_like(F)
    g[1:len(T)] = (-2.0 * s / c) * (E @ R.T)
    return float((E * E).sum() / c), g


def project_positions(P: np.ndarray, F: np.ndarray, X: np.ndarray, L: int,
                      Y: np.ndarray | None = None, tol: float = 1e-6,
                      max_basis_bytes: float = 2.5e9, chunk: int = 2000) -> np.ndarray:
    """Real-SH coefficients ``(K, 3)`` of positions ``X`` sampled at sphere points
    ``P`` (connectivity ``F``): the mass-weighted Galerkin fit
    ``(YᵀMY) f̂ = YᵀMX``.

    ``YᵀMY`` is well conditioned (eigenvalues 0.38-1.61 on a head's eigen
    sphere at band 79), so it is solved by Jacobi-preconditioned CG through ``Y``
    rather than formed: 8 iterations to ``tol = 1e-6``, 0.55 s at 10242 points
    against 4 s for forming and solving the 6400² Gram. Coefficients are within
    1e-6 of the exact solve, and the spectral decode is unchanged to 1e-6. Pass
    ``Y = real_sh_basis(P, L)`` to share it with a later evaluation. If ``Y`` would
    exceed ``max_basis_bytes`` it is not held: the Gram is accumulated in chunks
    (``syrk``) and factorised.
    """
    import scipy.linalg as sla

    from .encode import real_sh_basis
    from .operators import vertex_areas

    m = vertex_areas(P, F)
    K = (L + 1) ** 2
    X = np.asarray(X, float)
    if Y is None and len(P) * K * 8 > max_basis_bytes:
        G, b = np.zeros((K, K)), np.zeros((K, X.shape[1]))
        for i in range(0, len(P), chunk):
            Yc = real_sh_basis(P[i:i + chunk], L)
            G = sla.blas.dsyrk(1.0, (Yc * np.sqrt(m[i:i + chunk])[:, None]).T,
                               beta=1.0, c=G, lower=True, overwrite_c=True)
            b += Yc.T @ (m[i:i + chunk, None] * X[i:i + chunk])
        return sla.cho_solve(sla.cho_factor(G, lower=True, check_finite=False), b,
                             check_finite=False)
    Y = real_sh_basis(P, L) if Y is None else Y

    def gram(p):
        return Y.T @ (m[:, None] * (Y @ p))

    b = Y.T @ (m[:, None] * X)
    dg = m @ (Y * Y)                                   # Jacobi preconditioner
    x = b / dg[:, None]
    r = b - gram(x)
    z = r / dg[:, None]
    p, rz, bn = z.copy(), (r * z).sum(0), np.linalg.norm(b)
    for _ in range(200):
        if np.linalg.norm(r) <= tol * bn:
            break
        Ap = gram(p)
        al = rz / (p * Ap).sum(0)
        x += al * p
        r -= al * Ap
        z = r / dg[:, None]
        rz, rz_old = (r * z).sum(0), rz
        p = z + (rz / rz_old) * p
    return x


class SpectralSpinor:
    """``ρ̂ → v`` (the kernel pair) with its adjoint, at band ``L`` and
    spinor band ``Ls``.

    ``forward(ρ̂, v0) -> (v, cache)`` is
    :func:`~shapegen.geom.dirac_spectral.solve_spinor` -- the decoder's own
    solve, not a copy -- and keeps its factorised operator.
    ``backward(g_v, cache)`` solves the reduced resolvent of stage 2 with
    ``n_neumann`` extra sweeps through that factorisation and returns
    ``dJ/dρ̂``."""

    def __init__(self, L: int, Ls: float, c: float = 1.0):
        from . import dirac_spectral as ds

        self.ds, self.L, self.Ls, self.c = ds, L, Ls, c
        self.tf = ds._coupling_transform(float(Ls), int(L))
        self.W = np.broadcast_to(ds.coupling_grid(Ls, L)[2],
                                 (len(self.tf.theta), self.tf.nph))
        self.n = self.tf.n

    def forward(self, rho_hat: np.ndarray, v0=None):
        res = self.ds.solve_spinor(rho_hat, self.L, c=self.c, Ls=self.Ls, v0=v0)
        return np.r_[res.psi1, res.psi2], {"res": res, "gamma": res.gamma,
                                            "niter": res.niter}

    def _synth(self, x):
        return self.tf.synth(+0.5, x[:self.n]), self.tf.synth(-0.5, x[self.n:])

    def backward(self, g_v: np.ndarray, cache, tol: float = 1e-12,
                 max_sweeps: int = 30) -> np.ndarray:
        """Neumann sweeps until the update is below ``tol`` (relative): the
        contraction is ``γ²/σ₃²``, so at L=32, γ ≈ 5e-4 one sweep is exact to
        5e-14 and larger γ simply takes more.

        **Trap: γ = 0 exactly** (the round sphere, or a code after
        :func:`~shapegen.geom.dirac_spectral.close_potential`) makes ``G``
        exactly singular and this gradient garbage. :func:`encode_spectral`
        holds γ off zero through ``ℓ = 0``, which the loss does not see."""
        res = cache["res"]
        op, V, g2 = res.op, res.pair, res.gamma ** 2
        v = V[:, 0]

        def P(x):
            return x - V @ (V.conj().T @ x)

        def Ainv(x):
            return op.solve(op.solve_h(x))

        y = P(g_v)
        z = P(Ainv(y))
        for _ in range(max_sweeps):
            z, z_old = P(Ainv(y + g2 * z)), z
            if np.linalg.norm(z - z_old) <= tol * np.linalg.norm(z):
                break
        (Y1, Y2), (Z1, Z2) = self._synth(op.apply(v)), self._synth(z)
        (Q1, Q2), (V1, V2) = self._synth(op.apply(z)), self._synth(v)
        gg = -self.W * np.real(np.conj(Y1) * Z1 - np.conj(Y2) * Z2
                               + np.conj(Q1) * V1 - np.conj(Q2) * V2)
        return self.c * (self.ds._rho_basis(float(self.Ls), int(self.L)).T @ gg.ravel())


def encode_spectral(target: np.ndarray, P: np.ndarray, F: np.ndarray, L: int,
                    c0: np.ndarray | None = None, maxiter: int = 300,
                    rtol: float = 1e-3, window: int = 5, gamma_offset: float = 1e-3,
                    alpha: float = 0.75,
                    lift: "SpectralLift | None" = None,
                    spinor: SpectralSpinor | None = None,
                    Ls: float | None = None, verbose: bool = False) -> dict:
    """Encode ``target`` (positions at the sphere points ``P``, connectivity
    ``F``) as ``(L+1)²`` real SH coefficients of ρ, starting from ``c0``
    (default ρ̂ = 0, the round sphere).

    The decoded surface (:class:`SpectralSpinor` then :class:`SpectralLift`)
    is compared by :func:`coef_procrustes` with the target's position
    coefficients, projected once by :func:`project_positions`; ``err`` is
    that coefficient-space Procrustes error. L-BFGS-B on band-preconditioned
    variables (:func:`shapegen.geom.encode.band_precond`); never returns worse
    than ``c0``. ``Ls`` defaults to ``spinor_band(L, potential_band_extra(L))``.

    **Stopping:** when ``err`` improved by less than ``rtol`` (relative) over
    the last ``window`` iterations. On FaceScape heads at L=32 that stops at
    35-45 evaluations with nothing measurable left (the old ``ftol=1e-18`` ran
    to ~100 for <0.1%). L-BFGS-B's own ``ftol`` cannot express this: it
    compares against ``max(|f|, 1)`` and the loss is ~1e-5.

    **The start does not matter** (direct measurement and ρ̂ = 0 reach the
    same code to 1e-6, the direct start saving ~7 evaluations). **ℓ = 0 is not
    optimised**: it shifts every eigenvalue of ``σ₃G`` and moves no surface
    (:func:`~shapegen.geom.dirac_spectral.signed_gamma`), so the loss cannot
    see it. During the fit it is re-set after each evaluation to hold the
    signed γ at ``gamma_offset``, because at γ = 0 ``G`` is singular and
    :meth:`SpectralSpinor.backward` breaks (from ρ̂ = 0 exactly, L-BFGS stalls
    at the round sphere). At the end
    :func:`~shapegen.geom.dirac_spectral.close_potential` sets it so the code
    is exactly realisable (``gamma`` ~1e-15).

    Pass ``lift``/``spinor`` when encoding many shapes at one band: they hold
    the per-band precomputation.
    """
    from scipy.optimize import minimize

    from . import dirac_spectral as ds
    from .encode import band_precond

    Ls = ds.spinor_band(L, ds.potential_band_extra(L)) if Ls is None else Ls
    lift = SpectralLift(Ls) if lift is None else lift
    spinor = SpectralSpinor(L, Ls) if spinor is None else spinor
    Tc = project_positions(P, F, np.asarray(target, float), lift.Lf)
    c0 = np.zeros((L + 1) ** 2) if c0 is None else np.asarray(c0, float)

    d = band_precond(L, alpha)
    y00 = np.sqrt(4.0 * np.pi) / spinor.c            # ρ̂₀ per unit of constant ρ
    # ℓ = 0 during the fit: held so that the signed γ sits at gamma_offset
    state = {"v": None, "nfev": 0, "t": 0.0, "J0": None, "c00": c0[0] + gamma_offset * y00}

    def fg(u):
        t0 = time.perf_counter()
        c = d * u
        c[0] = state["c00"]
        v, sc = spinor.forward(c, v0=state["v"])
        X, lc = lift.forward(v)
        J, gX = coef_procrustes(Tc, X)
        g = spinor.backward(lift.backward(gX, lc), sc)
        g[0] = 0.0
        mu = ds.signed_gamma(sc["res"])
        state.update(v=v, nfev=state["nfev"] + 1, c00=c[0] - (mu - gamma_offset) * y00,
                     t=state["t"] + time.perf_counter() - t0)
        if state["J0"] is None:
            state["J0"] = J
        if verbose:
            print(f"  eval {state['nfev']:4d}  err {np.sqrt(J):.5f}  "
                  f"gamma {sc['gamma']:.3e}  {time.perf_counter() - t0:.2f}s", flush=True)
        return J, d * g

    hist = []

    def stop(intermediate_result):
        hist.append(np.sqrt(max(float(intermediate_result.fun), 0.0)))
        if len(hist) > window and hist[-1 - window] - hist[-1] < rtol * hist[-1]:
            raise StopIteration

    u0 = c0 / d
    res = minimize(fg, u0, jac=True, method="L-BFGS-B", callback=stop,
                   options={"maxiter": maxiter, "ftol": 1e-18, "gtol": 1e-14,
                            "maxcor": 40})
    J0 = state["J0"]
    u, J = (res.x, float(res.fun)) if res.fun < J0 else (u0, J0)
    c = d * u
    c[0] = state["c00"]
    status = (f"CONVERGED: err improved < {rtol:g} over {window} iterations"
              if res.status == 99            # the callback's StopIteration
              else str(res.message))
    c = ds.close_potential(c, ds.solve_spinor(c, L, c=spinor.c, Ls=Ls))
    gamma = ds.solve_spinor(c, L, c=spinor.c, Ls=Ls).gamma
    return {"coeffs": c, "err": float(np.sqrt(max(J, 0.0))),
            "err0": float(np.sqrt(max(J0, 0.0))), "gamma": gamma,
            "nfev": state["nfev"], "niter": int(res.nit),
            "sec_per_eval": state["t"] / max(state["nfev"], 1),
            "status": status}

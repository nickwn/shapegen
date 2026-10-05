"""The quotient encoder: ρ̂ from the target's spinor by the Dirac equation read
pointwise, then corrected onto the decoder's own truncated equation. No dense
matrices and no optimiser.

1. **Spinor off the mesh** at band ``L + ½``
   (:func:`~shapegen.geom.dirac_direct.mesh_spinor`), zero-padded to the
   decoder's band ``Ls``. ``L + ½`` beats ``Ls`` itself (1.20x vs 1.32x off
   ``dirac`` before step 3). Every taper tried on the top bands is worse.
2. **Pointwise quotient.** ``Dψ = ρψ`` pointwise gives
   ``ρ − 1 = −Re⟨ψ, σ₃ G(1)ψ⟩ / |ψ|²``. It is projected to band ``L`` by
   quadrature with *no* weight: the real SH are orthonormal on
   :func:`~shapegen.geom.dirac_spectral.coupling_grid`, so there is no solve.
   Weights ``|ψ|^p`` get worse as ``p`` rises (p=2 is 1.7x worse). So does the
   Galerkin least squares ``min ‖G(ρ̂)v‖`` (0.0137 vs 0.0085 on head 339).
3. **Petrov-Galerkin correction.** The decoder realises ``P_Ls[ρψ] = Dψ``, not
   the pointwise equation, and it falls short at high ℓ (~7% at ``Ls`` = 39.5).
   Iterate ``ρ̂ ← ρ̂ − Q[a + Bρ̂]``, where ``a + Bρ̂ = G(ρ̂)v`` is the decoder's
   Galerkin residual (grid products and transforms) and ``Q`` is step 2's
   quotient. The first pass *is* step 2; three passes converge.
4. ``ℓ = 0`` by the σ₃-Rayleigh quotient of ``v``: it is a decode null
   direction and only picks the branch. :func:`encode_quotient` then closes it
   exactly (:func:`~shapegen.geom.dirac_direct.close_code`).

**Measured** (L=32; 11 FaceScape heads, ico5; ratio to ``method="dirac"``,
whole / face): 1.11x / 1.06x (worst head 1.13x) in ~80 ms, against ``direct``
at 3.41x / 3.62x in ~170 ms and ``dirac`` at ~1.0 s. The code moves 3.0%
between ico5 and ico6 meshings of a head, the same as ``dirac`` (``direct``:
4.4%). Cubes (after the codec's remesh) and demo bottles at L=32: 0.98-1.01x.

**Where it pays.** Heads are 1.0-1.1x off ``dirac`` at every L from 4 to 32.
The demo bottle is not: 1.6-1.75x at L=8 and 1.2x at L=16-24 on ico5. A
higher spinor band does not help. The gap is ``dirac``'s resolvent weighting
(a spinor-error objective), for which no cheap analogue was found: a
per-degree gain is shape-dependent, re-measurement passes gain <8%, and a
matrix-free resolvent CG does not converge. ``dirac``'s dense algebra is cheap
at low L (49 vs 41 ms at L=8, 94 vs 47 ms at L=16) and ~n³ above that (341 vs
57 ms at L=24). So use ``dirac`` at L ≤ 16 and the quotient at L ≥ 24, or
wherever ``direct`` was used for speed.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import dirac_spectral as ds

__all__ = ["rho_quotient", "encode_quotient"]


@lru_cache(maxsize=4)
def _band_basis(Ls: float, L: int, Lg: int) -> np.ndarray:
    """Real SH of band ``L`` on :func:`~shapegen.geom.dirac_spectral.coupling_grid` ``(Ls, Lg)``."""
    from .encode import real_sh_basis

    th, ph, _w = ds.coupling_grid(Ls, Lg)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    pts = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
    return real_sh_basis(pts, L)


def rho_quotient(v: np.ndarray, L: int, Ls: float, Lm: float | None = None,
                 iters: int = 3, c: float = 1.0, over: int = 1) -> np.ndarray:
    """ρ̂ (band ``L``) from the spinor ``v`` (band ``Lm``, default ``Ls``):
    steps 2-4 of the module docstring, with ``iters`` passes (1 = the plain
    pointwise quotient). ``ℓ = 0`` is the Rayleigh estimate, not closed exactly.

    ``over``: analyse on a grid exact to degree ``2·Ls + over·L``. The pointwise
    quotient (÷|ψ|²) is not band-limited, so on the minimal grid (``over = 1``) its
    tail aliases into band ``L``, by an amount that depends on the shape's orientation
    relative to the grid. Two encodings equal up to a rotation differed by 1e-3 at
    L=16 (code spectrum); ``over = 3`` brings that to ~1e-5 for < 0.1 s.
    ``method="polyhedral"`` uses 3.

    ``v`` is zero-padded to the decoder band ``Ls``, which prefixes each spin's
    modes because :func:`~shapegen.spectral.swsh.half_modes` orders by ℓ first.
    """
    Lm = float(Ls) if Lm is None else float(Lm)
    Lg = int(over * L)
    tf = ds._coupling_transform(float(Ls), Lg)
    _th, _ph, w = ds.coupling_grid(Ls, Lg)
    Y = ds._rho_basis(float(Ls), int(L)) if over == 1 else _band_basis(float(Ls), int(L), Lg)
    n, k = tf.n, len(v) // 2
    if k > n:
        raise ValueError(f"spinor band {Lm} exceeds the decoder band {Ls}")
    v = np.r_[v[:k], np.zeros(n - k), v[k:], np.zeros(n - k)]
    e = ds.sw.half_edth_factor(-0.5, Ls)
    eb = ds.sw.half_edthbar_factor(+0.5, Ls)
    a1, a2 = e * v[n:] + v[:n], eb * v[:n] - v[n:]                  # G(ρ = 1) v
    p1, p2 = tf.synth(+0.5, v[:n]), tf.synth(-0.5, v[n:])
    q = np.abs(p1) ** 2 + np.abs(p2) ** 2
    W = np.broadcast_to(w, q.shape)
    r1, r2 = tf.synth(+0.5, a1), tf.synth(-0.5, a2)
    rho = np.zeros(Y.shape[1])
    for it in range(iters):
        # Q[r]: the quotient of the residual, projected with no weight (YᵀWY = I)
        rho -= Y.T @ (W * np.real(np.conj(p1) * r1 - np.conj(p2) * r2) / q).ravel() / c
        if it == iters - 1:
            break
        g = c * W * (Y @ rho).reshape(q.shape)                      # G(ρ̂)v = a + Bρ̂ at Ls
        r1 = tf.synth(+0.5, a1 + tf.adjoint(+0.5, g * p1))
        r2 = tf.synth(-0.5, a2 - tf.adjoint(-0.5, g * p2))
    # ℓ = 0: μ = Re vᴴσ₃G(ρ̂)v / |v|², and σ₃-quadrature of Bρ̂ is ∫ρ̃|ψ|²
    rt = (Y @ rho).reshape(q.shape)
    mu = (np.real(np.vdot(v[:n], a1) - np.vdot(v[n:], a2)) + c * np.sum(W * rt * q)) \
        / np.real(np.vdot(v, v))
    rho[0] -= mu * np.sqrt(4.0 * np.pi) / c
    return rho


def encode_quotient(target: np.ndarray, P: np.ndarray, F: np.ndarray, L: int,
                    lift=None, Ls: float | None = None, iters: int = 3,
                    close: bool = True) -> dict:
    """``target`` (positions at sphere points ``P``, connectivity ``F``) → ρ̂ at
    band ``L``, by :func:`~shapegen.geom.dirac_direct.mesh_spinor` at band
    ``L + ½`` and :func:`rho_quotient`.

    Returns ``{"coeffs", "gamma", "f_hat"}`` like
    :func:`~shapegen.geom.dirac_direct.encode_dirac`. ``close=True`` spends one
    spinor solve (warm-started at the target's spinor) to close ``ℓ = 0``
    exactly, and reuses it for ``f_hat``, the decoded surface's real-SH
    positions (band ``lift.Lf``). ``close=False`` returns only ``coeffs``, with
    no solve at all: the ~80 ms encoder, whose ``ℓ = 0`` is the Rayleigh
    estimate (the decode does not depend on it).
    """
    from .dirac_direct import close_code, mesh_spinor
    from .spectral_adjoint import SpectralLift

    Ls = ds.spinor_band(L, ds.potential_band_extra(L)) if Ls is None else Ls
    Lm = L + 0.5
    v = mesh_spinor(np.asarray(P, float), np.asarray(target, float), F, Lm)
    rho = rho_quotient(v, L, Ls, Lm, iters=iters)
    if not close:
        return {"coeffs": rho, "gamma": float("nan"), "f_hat": None}
    n, k = len(ds.sw.half_modes(0.5, Ls)), len(v) // 2
    v0 = np.r_[v[:k], np.zeros(n - k), v[k:], np.zeros(n - k)]
    rho, res, gamma = close_code(rho, L, Ls, v0=v0)
    lift = SpectralLift(Ls) if lift is None else lift
    return {"coeffs": rho, "gamma": gamma,
            "f_hat": lift.forward(np.r_[res.psi1, res.psi2])[0]}

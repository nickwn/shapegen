"""The quotient encoder (``codec.encode(method="quotient")``): ρ̂ from the target's spinor
by the Dirac equation read pointwise, then corrected onto the decoder's own
truncated equation. No dense matrices and no optimiser. Also home of the pieces
the polyhedral encoders share: the target's spinor projected onto spin-weighted
harmonics (:func:`mesh_spinor`, :func:`_project_faces`) and the closing of a code
(:func:`close_code`).

1. **Spinor off the mesh** at band ``L + ½``
   (:func:`mesh_spinor`), zero-padded to the
   decoder's band ``Ls``. ``L + ½`` beats ``Ls`` itself (10% better before
   step 3). Every taper tried on the top bands is worse.
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
   exactly (:func:`close_code`).

**Measured** (L=32; 11 FaceScape heads, ico5): decode error ~0.0077 / 0.0065
(whole / face) in ~80 ms, 3x better than ``direct`` (~170 ms). The code moves 3.0%
between ico5 and ico6 meshings of a head (``direct``: 4.4%).

**Where it is weaker:** smooth shapes at low L. The removed ``dirac`` encoder (a
dense, resolvent-weighted least squares ``min ‖G(ρ̂)v‖``) matched it on heads at
every L but was 1.6-1.75x better on the demo bottle at L=8 and 1.2x at L=16-24;
no cheap analogue of its weighting was found (a per-degree gain is
shape-dependent, re-measurement passes gain <8%, a matrix-free resolvent CG does
not converge).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..spectral.quaternion import qconj, qmul
from . import dirac_spectral as ds

__all__ = ["mesh_spinor", "rho_quotient", "close_code", "encode_quotient"]


# ------------------------------------------------------------------ the target's spinor, closing


def mesh_spinor(S: np.ndarray, T: np.ndarray, F: np.ndarray, Ls: float) -> np.ndarray:
    """The spinor ``v = (ψ1, ψ2)`` (band ``Ls``) of the map from the sphere mesh
    ``(S, F)`` to the target positions ``T`` on the same faces.

    Per face, :func:`~shapegen.geom.spin.face_spinors` gives λ with a consistent
    sign: ``|λ|² = e^u``, and ``λ̄ e λ`` rotates the sphere triangle onto the target
    triangle, with shear discarded. The global sign is irrelevant to
    :func:`rho_quotient`. The decoder's dictionary is ``λ = R_frame (ψ1 + ψ̄2 j)``,
    so the frame is removed at each face centroid. Each half is then projected
    onto its spin-weighted basis by quadrature with the sphere-triangle areas
    (:func:`_project_faces`). Frame and basis must share the same ``φ`` branch:
    both are antiperiodic, and only their product is single-valued.
    """
    from ..quad.lift import frame_quaternion
    from .mesh import triangle_areas
    from .spin import face_spinors

    lam = face_spinors(S, T, F)
    c = S[F].mean(1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)                  # face centroid directions
    th = np.arccos(np.clip(c[:, 2], -1.0, 1.0))
    ph = np.arctan2(c[:, 1], c[:, 0])
    raw = qmul(qconj(frame_quaternion(th, ph)), lam)                # undo the frame
    w = triangle_areas(S, F)
    return _project_faces(float(Ls), th, ph, w * (raw[:, 0] + 1j * raw[:, 1]),
                          w * (raw[:, 2] - 1j * raw[:, 3]))


@lru_cache(maxsize=4)
def _theta_table(Ls: float, nt: int):
    """Spin ±½ Wigner-d on ``nt`` equispaced θ in ``[0, π]``."""
    return ds.SpinorTransform(Ls, np.linspace(0.0, np.pi, nt), 2)


def _project_faces(Ls: float, th, ph, g1, g2, nt: int | None = None) -> np.ndarray:
    """``half_eval_basis(±½, Ls, th, ph).conj().T @ g`` for both halves, i.e.
    :func:`~shapegen.spectral.swsh.half_eval_adjoint`, but separably: the φ
    factor ``e^{−imφ}`` is exact per point (one recurrence in m), and the
    Wigner-d factor is 4-point Lagrange interpolation from a cached θ table
    (``16 (Ls + 1)`` nodes). Measured 7e-7 relative to the exact per-point
    ladders (1e-5 at 256 nodes, 5e-8 at 1024), and 60 ms instead of 300 ms on
    20480 faces at ``Ls`` = 32.5.
    """
    import scipy.sparse as sp

    nt = int(16 * (Ls + 1)) if nt is None else int(nt)
    tf = _theta_table(Ls, nt)
    h = np.pi / (nt - 1)
    j0 = np.clip(np.floor(th / h).astype(int) - 1, 0, nt - 4)
    t = th / h - j0
    W = np.ones((len(th), 4))
    for i in range(4):
        for j in range(4):
            if i != j:
                W[:, i] *= (t - j) / (i - j)
    interp = sp.csr_matrix((W.ravel(), (np.repeat(np.arange(len(th)), 4),
                                        (j0[:, None] + np.arange(4)).ravel())),
                           shape=(len(th), nt)).T.tocsr()          # (nt, npts)
    ms = tf._half[0.5][2]                                          # the same m for both spins
    E = np.empty((len(ph), len(ms)), complex)
    E[:, 0] = np.exp(-1j * ms[0] * ph)
    z = np.exp(-1j * ph)
    for k in range(1, len(ms)):
        E[:, k] = E[:, k - 1] * z
    out = []
    for s, g in ((+0.5, g1), (-0.5, g2)):
        D, idx, _ms, _Ph = tf._half[s]
        X = (interp @ (E * g[:, None])).T                          # (nM, nt)
        gve = np.einsum("mtl,mt->ml", D, X).ravel()
        flat = idx.ravel()
        valid = flat < tf.n                                        # every mode once; pads -> n
        v = np.empty(tf.n, complex)
        v[flat[valid]] = gve[valid]
        out.append(v)
    return np.concatenate(out)


def close_code(rho: np.ndarray, L: int, Ls: float, c: float = 1.0, v0=None):
    """``(ρ̂ closed, solve, γ)``: ``ℓ = 0`` set so that γ = 0
    (:func:`~shapegen.geom.dirac_spectral.close_potential`), from one spinor
    solve warm-started at ``v0``. Closing shifts ρ by a constant, which leaves
    the eigenvectors unchanged, so the solve's spinor *is* the closed code's
    decode. ``γ`` is the closed code's residual ``‖G v − μσ₃v‖`` from that
    same solve."""
    res = ds.solve_spinor(rho, L, c=c, Ls=Ls, v0=v0)
    mu = ds.signed_gamma(res)
    rho = ds.close_potential(rho, res)
    n = len(res.psi1)
    x = np.r_[res.psi1, res.psi2]
    x = x / np.linalg.norm(x)
    sig = np.r_[np.ones(n), -np.ones(n)]
    return rho, res, float(np.linalg.norm(res.op.apply(x) - mu * sig * x))


# ------------------------------------------------------------------ the quotient
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
    band ``L``, by :func:`mesh_spinor` at band
    ``L + ½`` and :func:`rho_quotient`.

    Returns ``{"coeffs", "gamma", "f_hat"}``. ``close=True`` spends one
    spinor solve (warm-started at the target's spinor) to close ``ℓ = 0``
    exactly, and reuses it for ``f_hat``, the decoded surface's real-SH
    positions (band ``lift.Lf``). ``close=False`` returns only ``coeffs``, with
    no solve at all: the ~80 ms encoder, whose ``ℓ = 0`` is the Rayleigh
    estimate (the decode does not depend on it).
    """
    from ..quad.lift import SpectralLift

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

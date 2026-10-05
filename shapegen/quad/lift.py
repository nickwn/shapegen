"""Lifting the traced grid to ℝ³.

:func:`lift_spectral` is the lift (and what :func:`shapegen.codec.decode`
calls): the spectral spinor gives the surface's differential
``df̃ = λ̄ df₀ λ``, the least squares ``∫|∇f − ω|²`` is solved in spherical
harmonics (:class:`shapegen.geom.spectral_adjoint.SpectralLift`), and the
surface is read off at the requested directions. No mesh and no edge graph.

:func:`frame_quaternion` / :func:`spinor_quaternion` are the
spinor-to-quaternion dictionary it rests on.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = [ "frame_quaternion", "spinor_quaternion", "lift_spectral"]


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


@lru_cache(maxsize=2)
def _spectral_lift(Ls: float):
    from ..geom.spectral_adjoint import SpectralLift
    return SpectralLift(Ls)


def lift_spectral(dirs: np.ndarray, rho_hat: np.ndarray, L: int, res=None,
                  chunk: int = 4000):
    """Positions in ℝ³ at unit directions ``dirs`` ``(n,3)`` for the surface
    encoded by ``rho_hat`` (real-SH ρ coefficients at bandwidth ``L``).

    Solves ``∫_S² |∇f − ω|² dA`` exactly in spherical harmonics
    (:class:`~shapegen.geom.spectral_adjoint.SpectralLift`), ``ω`` the
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

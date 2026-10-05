"""The direct Dirac encoder: ρ̂ from linear solves, no optimiser, and no
positions in SH.

Instead of measuring the target's curvature (``mesh_to_rho_direct``, which the
decoder does not realise faithfully above ℓ ≈ 8), ask for the ρ̂ whose *decoder*
returns the target's own spinor:

1. **Spinor off the mesh.** A conformal map is a spin transform,
   ``df = λ̄ df₀ λ``, so λ is the similarity part of the map from each sphere
   triangle to its mesh triangle (:func:`~shapegen.geom.spin.face_spinors`). The
   decoder's frame is removed at each face centroid, and the two spin halves are
   projected straight onto spin ±½ SH by quadrature (:func:`mesh_spinor`). Its
   projection ``v_T`` to band ``Ls`` lifts to within 0.0035 / 0.0031 of the heads,
   below the fitted optimum.
2. **Make it the decoder's kernel.** ``G(ρ̂) v`` is affine in ρ̂, so
   ``min ‖G(ρ̂) v_T‖`` is one linear least squares (:func:`rho_for_spinor`).
3. **Weight by how residual moves the kernel.** To first order the decoded
   spinor moves by ``δv ≈ −G⁻¹ r`` off its pair, so residual along ``G``'s small
   singular directions matters most. One re-solve of
   ``min ‖(G₀ − iη σ₃)⁻¹ r‖`` with ``G₀`` from step 2 does that. ``σ₃G`` is
   Hermitian, so this is the regularised resolvent; ``η`` keeps the solve
   branch-safe. A hard projection off ``G₀``'s pair alternates with branch
   switches (decode error 0.65). The result is flat in ``η`` from 3e-3 to 0.1,
   and a second re-solve changes nothing.
4. :func:`~shapegen.geom.dirac_spectral.close_potential` for ``ℓ = 0``.

**Measured** (FaceScape heads 339/254, L=32, eigen sphere): decode 0.0069 / 0.0061
in ~1.05 s given the sphere (0.06 s spinor, 1.0 s ``rho_for_spinor``); 1.3 s in
:func:`shapegen.codec.encode`, which also evaluates the decode to score it. The fit gets 0.0050 / 0.0045 in 28 s; the
mesh measurement 0.0253 / 0.0212. The code sits 5% / 7% / 25% from the fitted one
in ℓ 1-8 / 9-16 / 17-32 (the mesh measurement: 12% / 14% / 37%). As a start for
:func:`~shapegen.geom.spectral_adjoint.encode_spectral` it reaches the same optimum
in 24-25 evaluations instead of 41.

The spinor used to be read off a band-79 SH fit of the positions: 0.00686 / 0.00598,
1-2% better, but 0.7 s instead of 0.4 s (CG projection plus a 6400-column basis).
The two spinors agree to 0.7%. A Galerkin fit of the face samples in place of plain
quadrature changes nothing.
Each ``rho_for_spinor`` step is a dense factorisation or product of 0.1-0.35 s,
the largest being the closing spinor solve.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..spectral.quaternion import qconj, qmul
from . import dirac_spectral as ds

__all__ = ["mesh_spinor", "rho_for_spinor", "close_code", "encode_dirac"]


def mesh_spinor(S: np.ndarray, T: np.ndarray, F: np.ndarray, Ls: float) -> np.ndarray:
    """The spinor ``v = (ψ1, ψ2)`` (band ``Ls``) of the map from the sphere mesh
    ``(S, F)`` to the target positions ``T`` on the same faces.

    Per face, :func:`~shapegen.geom.spin.face_spinors` gives λ with a consistent
    sign: ``|λ|² = e^u``, and ``λ̄ e λ`` rotates the sphere triangle onto the target
    triangle, with shear discarded. The global sign is irrelevant to
    :func:`rho_for_spinor`. The decoder's dictionary is ``λ = R_frame (ψ1 + ψ̄2 j)``,
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


def rho_for_spinor(v: np.ndarray, L: int, Ls: float, c: float = 1.0,
                   eta: float = 1e-2, reweight: int = 1, return_solve: bool = False):
    """ρ̂ (band ``L``) whose Dirac operator best annihilates ``v``, closed
    (γ = 0). Steps 2-4 of the module docstring.

    ``return_solve``: also return ``(res, gamma)``. ``res`` is the spinor solve the
    closing used. Closing shifts ρ by a constant, which leaves the eigenvectors
    unchanged, so ``res``'s spinor *is* the closed code's decode. ``gamma`` is
    the closed code's residual ``‖G v − μσ₃v‖``, from that same solve.

    ``G(ρ̂) v = a + B ρ̂``: with ``ρ = 1 + c·Σ ρ̂_k Y_k``, ``a`` is the round
    sphere's ``G v`` (the ρ = 1 coupling is the identity), and column ``k`` of
    ``B`` is ``c·(P₊[Y_k ψ1], −P₋[Y_k ψ2])``, the Galerkin projection of a
    pointwise product on :func:`~shapegen.geom.dirac_spectral.coupling_grid`.
    """
    tf = ds._coupling_transform(float(Ls), int(L))
    th, ph, w = ds.coupling_grid(Ls, L)
    Yr = ds._rho_basis(float(Ls), int(L)).reshape(len(th), len(ph), -1)
    n = tf.n
    p1, p2 = tf.synth(+0.5, v[:n]), tf.synth(-0.5, v[n:])
    B = c * np.r_[tf.adjoint(+0.5, (w * p1)[..., None] * Yr),
                  -tf.adjoint(-0.5, (w * p2)[..., None] * Yr)]
    e = ds.sw.half_edth_factor(-0.5, Ls)
    eb = ds.sw.half_edthbar_factor(+0.5, Ls)
    a = np.r_[e * v[n:] + v[:n], eb * v[:n] - v[n:]]

    def solve(B, a):
        # real ρ̂: Re(BᴴB) = BrᵀBr + BiᵀBi, one real syrk on the stacked parts
        # (half the flops of the complex product), then Cholesky
        import scipy.linalg as sla

        Bs = np.concatenate([B.real, B.imag])
        A = sla.blas.dsyrk(1.0, Bs.T, lower=True)
        rhs = -(Bs.T @ np.concatenate([a.real, a.imag]))
        return sla.cho_solve(sla.cho_factor(A, lower=True, check_finite=False), rhs,
                             check_finite=False)

    rho = solve(B, a)
    for _ in range(reweight):
        # only G₀'s couplings are needed, not its kernel.
        # (σ₃G₀ − iη)⁻¹ σ₃ = (G₀ − iη σ₃)⁻¹: the same block form, so the Schur solve applies
        grid = ds.rho_on_grid(rho, L, Ls, c)
        shift = 1j * eta * np.eye(n)
        W = ds.DiracOperator(ds.coupling_matrix(grid, +0.5, Ls, L) - shift,
                             ds.coupling_matrix(grid, -0.5, Ls, L) - shift, Ls)
        rho = solve(W.solve(B), W.solve(a))
    rho, res, gamma = close_code(rho, L, Ls, c, v0=v)       # v is near the kernel: warm start
    return (rho, res, gamma) if return_solve else rho


def encode_dirac(target: np.ndarray, P: np.ndarray, F: np.ndarray, L: int,
                 lift=None, Ls: float | None = None, eta: float = 1e-2,
                 reweight: int = 1) -> dict:
    """``target`` (positions at sphere points ``P``, connectivity ``F``) →
    closed ρ̂ at band ``L``, by :func:`mesh_spinor` and :func:`rho_for_spinor`.

    Returns ``{"coeffs", "gamma", "f_hat"}``. ``f_hat`` is the decoded surface's
    real-SH positions (band ``lift.Lf``), for scoring the decode at ``P``
    without another solve. Pass ``lift`` to reuse the per-band precomputation.
    """
    from .spectral_adjoint import SpectralLift

    Ls = ds.spinor_band(L, ds.potential_band_extra(L)) if Ls is None else Ls
    lift = SpectralLift(Ls) if lift is None else lift
    v = mesh_spinor(np.asarray(P, float), np.asarray(target, float), F, Ls)
    rho, res, gamma = rho_for_spinor(v, L, Ls, eta=eta, reweight=reweight,
                                     return_solve=True)
    return {"coeffs": rho, "gamma": gamma,
            "f_hat": lift.forward(np.r_[res.psi1, res.psi2])[0]}

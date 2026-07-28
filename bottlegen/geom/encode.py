"""The ρ encoder: ``mesh → ρ̂``, band-limited spherical-harmonic coefficients.

This is the inverse of the decode in :mod:`bottlegen.geom.dirac`, and it is built
by *optimisation* rather than by a local formula. The two local encoders —
:func:`~bottlegen.geom.spin.mesh_to_rho` (project a measured λ) and the direct
``ρ = H e^u − 1`` — both minimise a Dirac-consistency residual, which is not the
quantity anyone reports. Minimising decoded *shape error* directly, through
:func:`~bottlegen.geom.dirac_adjoint.decode_adjoint`, closes roughly half the gap
they leave: measured on a bottle, 0.1095 → 0.0569.

Encoding happens **in coefficient space at the target bandwidth**, not in vertex
space followed by a projection. That matters: an encoder that knows it will be
truncated to ``L`` spends its budget differently from one that is truncated
afterwards, and the difference grows as ``L`` falls.

The objective is the Procrustes-aligned squared shape error, which is invariant
under rotation, translation *and* scale. That is not a convenience — it is
exactly the gauge-invariance :func:`stage_a_adjoint` requires of a loss, since
the decode only determines the immersion up to a similarity. By the envelope
theorem the alignment is held fixed when differentiating, so the SVD inside
:func:`procrustes` is never differentiated through.

**Two traps, both measured, both silent.**

*Optimiser tolerances.* ``J`` is a normalised squared error — about 3e-5 at a
0.6% shape error — so its gradient is numerically tiny. L-BFGS-B's default
``pgtol`` of 1e-5 is already satisfied at the starting point and it returns with
``nit=0``, having taken no step, which is indistinguishable from convergence
unless the iteration count is inspected. :func:`encode_sh` pins both tolerances
far below the scale of the problem and reports ``niter`` and ``gnorm`` so this
cannot hide.

*Warm starts across bandwidths.* Seeding band ``L`` with the solution at ``L−1``
is tempting and usually helps, but it is not safe on its own: a band that stalls
hands its bad basin to every band after it. Measured on the bottle, a warm chain
stalled at 0.0944 for L=4…10 while a cold start at L=10 reached 0.0065. The
starting objective does not predict which will win — the stalled point has the
*lower* ``J`` — so a sweep should run both and keep whichever *ends* better.
``scripts/eval_sh_roundtrip.py`` does this.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from . import dirac, dirac_adjoint as da
from . import spin

__all__ = ["real_sh_basis", "n_coeffs", "band_index", "procrustes",
           "encode_sh", "decode_sh", "sh_project"]


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
    from scipy.special import sph_harm_y
    r = np.linalg.norm(V, axis=1)
    theta = np.arccos(np.clip(V[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(V[:, 1], V[:, 0])
    root2 = np.sqrt(2.0)
    cols = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            if m == 0:
                cols.append(np.real(sph_harm_y(l, 0, theta, phi)))
            elif m > 0:
                cols.append(root2 * (-1) ** m * np.real(sph_harm_y(l, m, theta, phi)))
            else:
                cols.append(root2 * (-1) ** m * np.imag(sph_harm_y(l, -m, theta, phi)))
    return np.column_stack(cols)


def sh_project(rho: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least-squares SH coefficients of a per-vertex field. ``Y`` is the basis."""
    return np.linalg.lstsq(Y, rho, rcond=None)[0]


# --------------------------------------------------------------------------- #
# The objective                                                               #
# --------------------------------------------------------------------------- #

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
    A = target - target.mean(0)
    B = X - X.mean(0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    BR = B @ R
    denom = float((BR * BR).sum())
    s = float((A * BR).sum() / max(denom, 1e-300))
    c = float((A * A).sum())
    E = A - s * BR
    J = float((E * E).sum() / c)
    dB = (-2.0 * s / c) * (E @ R.T)
    return J, dB - dB.mean(0)


# --------------------------------------------------------------------------- #
# Encode / decode                                                             #
# --------------------------------------------------------------------------- #

def decode_sh(coeffs: np.ndarray, sph: np.ndarray, F: np.ndarray,
              Y: np.ndarray | None = None, L: int | None = None):
    """``ρ̂ → mesh``. Synthesise ρ on the base sphere and run the spin transform."""
    if Y is None:
        Y = real_sh_basis(sph, L if L is not None else
                          int(round(np.sqrt(len(coeffs)))) - 1)
    rho = Y[:, :len(coeffs)] @ coeffs
    lam, _ = dirac.spin_transform(sph, F, rho)
    return dirac.integrate_edges(sph, F, lam)


def encode_sh(mesh, L: int, sph=None, Y=None, c0=None, maxiter: int = 60,
              rho0=None, cmcf_iters: int = 300, verbose: bool = False):
    """Encode ``mesh`` as ``(L+1)²`` real SH coefficients of ρ.

    ``sph`` is the cMCF sphere parameterisation (computed if absent — it is the
    expensive step, so pass it when sweeping ``L``). ``Y`` is a basis of at least
    ``(L+1)²`` columns; ``c0`` an initial guess, e.g. the solution at a lower
    bandwidth zero-padded, which is worth a large fraction of the iteration
    budget.

    Returns a dict with ``coeffs``, ``rho``, ``sph``, ``err`` (the *relative
    Procrustes error*, √J), ``err0`` for the initialiser, and ``nfev``.
    """
    F = mesh.faces
    target = mesh.vertices

    if sph is None:
        from . import conformal as cf
        sph = cf.mobius_normalize(mesh, cf.spherical_conformal_map(
            mesh, iters=cmcf_iters))
    if Y is None:
        Y = real_sh_basis(sph, L)
    Yl = Y[:, :n_coeffs(L)]

    if c0 is None:
        if rho0 is None:
            rho0 = spin.mesh_to_rho(mesh, sph=sph)[0]
        c0 = sh_project(rho0, Yl)
    c0 = np.asarray(c0, dtype=float)

    def fg(c):
        rho = Yl @ c
        X, cache = da.decode_forward(sph, F, rho)
        J, dJdX = procrustes(target, X)
        return J, Yl.T @ da.decode_adjoint(dJdX, cache)

    J0, g0 = fg(c0)
    # ``J`` is a *normalised* squared error, so it and its gradient are numerically
    # tiny (J ~ 3e-5 at a 0.6% shape error). L-BFGS-B's default pgtol of 1e-5 is
    # then already satisfied at the starting point and it returns with nit=0
    # without taking a step — a silent no-op that looks exactly like convergence.
    # Both tolerances are pinned well below the scale of the problem instead.
    res = minimize(fg, c0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-18, "gtol": 1e-14})
    # L-BFGS can end on a rejected step; keep whichever point is actually better.
    c, J = (res.x, float(res.fun)) if res.fun < J0 else (c0, J0)
    return {"coeffs": c, "rho": Yl @ c, "sph": sph, "L": L,
            "err": float(np.sqrt(max(J, 0.0))),
            "err0": float(np.sqrt(max(J0, 0.0))),
            "nfev": int(res.nfev), "niter": int(res.nit),
            "gnorm0": float(np.abs(g0).max()),
            "gnorm": float(np.abs(res.jac).max()) if res.jac is not None else np.nan,
            "status": str(res.message)}

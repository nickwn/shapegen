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
is tempting, and before :func:`band_precond` existed it was actively dangerous: a
band that stalled handed its bad basin to every band after it, and a warm chain
sat at 0.0944 for L=4…10 where a cold start at L=10 reached 0.0065. Note the
starting objective does *not* predict this — the stalled point has the **lower**
``J`` — so no cheap heuristic picks the right start.

Preconditioning removed the stall at its source. Measured afterwards across
L=2…24, warm and cold reach the same optimum to 0.00% wherever the optimiser
converges, and cost the same in total (800 iterations either way; warm is ~2.7×
*worse* below L≈8 and ~20% better above it, and the two cancel). So sweeps just
encode each band independently from its own projection. Warm starting is not
wrong here, it is simply not worth the code.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from . import dirac, dirac_adjoint as da
from . import spin

__all__ = ["real_sh_basis", "n_coeffs", "band_index", "procrustes",
           "band_precond", "encode_sh", "decode_sh", "sh_project"]


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
    # Only the direction matters, so divide out the radius rather than assuming
    # the caller normalised: `sph` comes from cMCF and is unit only up to the
    # flow's own tolerance.  The clip guards acos against |z/r| = 1 + 1e-16.
    r = np.linalg.norm(V, axis=1)
    theta = np.arccos(np.clip(V[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(V[:, 1], V[:, 0])

    # Real orthonormal SH from the complex ones.  For m > 0 the real and
    # imaginary parts of Y_l^m give the cos(m phi) and sin(m phi) pair; the
    # sqrt(2) restores unit norm after taking one of the two, and the (-1)^m
    # cancels the Condon-Shortley phase that scipy carries so that the resulting
    # basis is the standard real one.  (The exact phase convention is immaterial
    # here -- we only ever fit and re-synthesise in the *same* basis -- but
    # matching the standard keeps the coefficients comparable to other tools.)
    root2 = np.sqrt(2.0)
    cols = []
    # Loop order defines the coefficient layout, and that layout is load-bearing:
    # degree ascending, m from -l to +l, so the first (L'+1)^2 columns form the
    # bandwidth-L' basis and callers can slice instead of rebuilding.
    for l in range(L + 1):
        for m in range(-l, l + 1):
            if m == 0:
                # m = 0 is already real; no sqrt(2), no phase.
                cols.append(np.real(sph_harm_y(l, 0, theta, phi)))
            elif m > 0:
                cols.append(root2 * (-1) ** m * np.real(sph_harm_y(l, m, theta, phi)))
            else:
                # Note the +|m| passed to scipy: the sin partner comes from the
                # imaginary part of the *positive*-order harmonic.
                cols.append(root2 * (-1) ** m * np.imag(sph_harm_y(l, -m, theta, phi)))
    return np.column_stack(cols)


def sh_project(rho: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least-squares SH coefficients of a per-vertex field. ``Y`` is the basis.

    Least squares rather than quadrature (``Yᵀ diag(area) rho``) on purpose: the
    mesh's lumped-area quadrature is only ~4e-3 accurate on an icosphere, which
    would put a systematic error into the initialiser and, worse, into the
    *baseline* curve this is also used to build.
    """
    # rcond=None keeps numpy's future-proof default (machine-eps cutoff).  It
    # matters at high L, where Y's columns approach the mesh's resolution limit
    # and the system becomes genuinely ill-conditioned -- the small singular
    # values there are aliasing, not signal, and are better dropped.
    return np.linalg.lstsq(Y, rho, rcond=None)[0]


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

    Measured, cold start, bottle at n=2562, L=40, 220 function evaluations::

        alpha=0     (plain)      final error 0.0045
        alpha=0.75               final error 0.0019, and reaches 0.0045 in 26

    an 8.5x reduction in work for the same answer. The optimum in ``alpha`` is
    flat over 0.5-0.75 and reproduces across meshes; above ~0.875 it degrades
    again.

    A *probed* per-band Jacobi preconditioner was tried and rejected: it beat
    this at L=12 but lost to no preconditioning at all at L=16, because one
    representative coefficient stops describing a band as the m-multiplicity
    grows, and probing every coefficient costs more than it saves.
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
    # exactly why encode_sh has to override L-BFGS's default tolerances.
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


# --------------------------------------------------------------------------- #
# Encode / decode                                                             #
# --------------------------------------------------------------------------- #

def decode_sh(coeffs: np.ndarray, sph: np.ndarray, F: np.ndarray,
              Y: np.ndarray | None = None, L: int | None = None):
    """``ρ̂ → mesh``. Synthesise ρ on the base sphere and run the spin transform.

    The exact inverse of :func:`encode_sh`, and the function that makes the
    round trip a round trip: whatever ``encode_sh`` reports as its error is what
    you get back through here (pinned by
    ``tests/test_encode.py::test_decode_sh_inverts_encode_sh``).
    """
    if Y is None:
        # No basis supplied: rebuild one. L is recoverable from the coefficient
        # count since K = (L+1)^2, so an explicit L is only needed when passing a
        # coefficient vector shorter than its own bandwidth.
        Y = real_sh_basis(sph, L if L is not None else
                          int(round(np.sqrt(len(coeffs)))) - 1)
    # Slice rather than require an exactly-sized basis, so a single large Y can
    # serve every bandwidth in a sweep.
    rho = Y[:, :len(coeffs)] @ coeffs
    # Stage A: the Dirac eigensolve. gamma (discarded here) is the squared Dirac
    # residual -- worth capturing if you want to know how realisable this rho is.
    lam, _ = dirac.spin_transform(sph, F, rho)
    # Stages B and C: the CPS edge rule, then the cotan-Poisson integration.
    # Result is mean-centred; it is defined only up to a similarity, which is
    # exactly why the error metric has to be Procrustes-aligned.
    return dirac.integrate_edges(sph, F, lam)


def encode_sh(mesh, L: int, sph=None, Y=None, c0=None, maxiter: int = 60,
              rho0=None, cmcf_iters: int = 300, verbose: bool = False,
              alpha: float = 0.75, maxcor: int = 40, precond=None):
    """Encode ``mesh`` as ``(L+1)²`` real SH coefficients of ρ.

    ``sph`` is the cMCF sphere parameterisation (computed if absent — it is the
    expensive step, so pass it when sweeping ``L``). ``Y`` is a basis of at least
    ``(L+1)²`` columns. ``c0`` overrides the default initialiser (the λ-route
    encoder projected to this bandwidth), which is a poor answer but a good
    starting point — measured ``‖ρ_opt − ρ₀‖/‖ρ₀‖ = 0.19``.

    Returns a dict with ``coeffs``, ``rho``, ``sph``, ``err`` (the *relative
    Procrustes error*, √J), ``err0`` for the initialiser, and ``nfev``.
    """
    F = mesh.faces
    target = mesh.vertices

    # The base sphere. Every stage below lives on *this* geometry with the
    # mesh's own connectivity, so vertex i of sph corresponds to vertex i of the
    # target -- that correspondence is what makes a per-vertex shape error
    # meaningful at all. cMCF is by far the most expensive step here, hence the
    # option to pass it in when sweeping bandwidths.
    if sph is None:
        from . import conformal as cf
        sph = cf.mobius_normalize(mesh, cf.spherical_conformal_map(
            mesh, iters=cmcf_iters))
    if Y is None:
        Y = real_sh_basis(sph, L)
    Yl = Y[:, :n_coeffs(L)]          # accept an over-wide basis and slice

    # Initialiser. The local lambda-route encoder is a poor *answer* (it
    # minimises Dirac consistency, not shape error) but a good *starting point*:
    # measured ||rho_opt - rho_0||/||rho_0|| = 0.19.
    if c0 is None:
        if rho0 is None:
            rho0 = spin.mesh_to_rho(mesh, sph=sph)[0]
        c0 = sh_project(rho0, Yl)
    c0 = np.asarray(c0, dtype=float)

    # Optimise in preconditioned variables u, with c = d * u. Pure change of
    # variables; see band_precond for why it is worth ~8x.
    d = band_precond(L, alpha) if precond is None else np.asarray(precond, float)

    def fg(u):
        """Objective and gradient at preconditioned coefficients ``u``.

        One call = one decode + one adjoint, which is the unit of cost for the
        whole encoder (~640 ms at n=2562), so nfev is the honest measure of work.
        """
        # u -> c -> rho: two linear maps, both trivially differentiable.
        rho = Yl @ (d * u)
        # Forward: rho -> X. The cache carries lambda and gamma so the backward
        # pass does not repeat the eigensolve.
        X, cache = da.decode_forward(sph, F, rho)
        # J must be gauge-invariant (rotation AND scale) or the adjoint below is
        # not valid -- see stage_a_adjoint. Procrustes is exactly that.
        J, dJdX = procrustes(target, X)
        # Reverse: dJ/dX -> dJ/drho through the three stage adjoints, then Ylᵀ
        # for the basis and d for the preconditioner. Both are the adjoints of
        # the two linear maps at the top of this function, applied in reverse.
        return J, d * (Yl.T @ da.decode_adjoint(dJdX, cache))

    u0 = c0 / d
    # Evaluate the start point explicitly: needed both to report err0 and to
    # guarantee below that we never return something worse than we were given.
    J0, g0 = fg(u0)
    # ``J`` is a *normalised* squared error, so it and its gradient are numerically
    # tiny (J ~ 3e-5 at a 0.6% shape error). L-BFGS-B's default pgtol of 1e-5 is
    # then already satisfied at the starting point and it returns with nit=0
    # without taking a step — a silent no-op that looks exactly like convergence.
    # Both tolerances are pinned well below the scale of the problem instead.
    res = minimize(fg, u0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-18, "gtol": 1e-14,
                            "maxcor": maxcor})
    # L-BFGS can end on a rejected step; keep whichever point is actually better.
    # Makes encode_sh monotone by construction: it can never return worse than
    # its initialiser, so a caller sweeping bandwidths cannot go backwards.
    u, J = (res.x, float(res.fun)) if res.fun < J0 else (u0, J0)
    # Undo the preconditioner. Forgetting this is the one silent bug this
    # reparameterisation can cause -- u is a plausible-looking coefficient vector
    # that decodes to the wrong mesh -- so test_decode_sh_inverts_encode_sh is
    # parametrised over alpha specifically to catch it.
    c = d * u
    # Gradient norms are reported in *coefficient* space, not the preconditioned
    # space, so they stay comparable across different alpha.
    gc = (res.jac / d) if res.jac is not None else None
    return {"coeffs": c, "rho": Yl @ c, "sph": sph, "L": L, "alpha": alpha,
            "err": float(np.sqrt(max(J, 0.0))),
            "err0": float(np.sqrt(max(J0, 0.0))),
            "nfev": int(res.nfev), "niter": int(res.nit),
            "gnorm0": float(np.abs(g0 / d).max()),
            "gnorm": float(np.abs(gc).max()) if gc is not None else np.nan,
            "status": str(res.message)}

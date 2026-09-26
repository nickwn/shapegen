"""The quaternionic Dirac solve, spectrally (PLAN.md §P9).

PLAN.md's last ``[unbuilt]`` item, which it called the largest modelling
error in the chain: the connection in :mod:`shapegen.quad.field` used the
linearisation ``mu = 1 + 2rho`` instead of solving for the spinor. This module
solves it, and since 2026-09-26 it is the only potential (the linearisation
was removed) and the spinor the §P7 route 2 lift integrates.

**The system.** Splitting ``lambda = a + b j`` gives the Kamberov-Pedit-Pinkall
/ Konopelchenko-Taimanov pair

    ethbar psi1 =  rho psi2         psi1 spin +1/2
    eth    psi2 = -rho psi1         psi2 spin -1/2

``eth`` and ``ethbar`` are diagonal in the spin-weighted basis, and
multiplication by ``rho`` is built **by transform** rather than from 3j
symbols (PLAN.md's instruction): synthesise a mode, multiply on the grid,
analyse back. That needs half-integer spin, which is
:func:`shapegen.spectral.swsh.half_basis` (§P8.2), which in turn needs a
Wigner-d that survives degree 32, which is
:func:`~shapegen.spectral.swsh.wigner_d_ladder` (§P8.1). The three steps are a
chain, in that order.

**The operator, and why a singular value rather than an eigenvalue.** Write

    G(psi1, psi2) = ( eth psi2 + rho psi1 ,  ethbar psi1 - rho psi2 )

whose kernel is exactly the solution set. ``G`` is *not* Hermitian and its
eigenvalues go imaginary as soon as ``l + 1/2`` exceeds ``rho`` -- the natural
object is its smallest **singular** value, i.e. the smallest eigenvalue of the
Hermitian PSD Gram operator ``G* G``. That also lines the two solvers up:
:mod:`shapegen.geom.dirac` builds ``A = (D - rho)* M_F (D - rho)`` and reports
a *squared* residual, so PLAN.md §2.5's prediction reads
``gamma_mesh ~ gamma_spec**2`` with no further conversion.

**The rho convention, which is not this repository's.** PLAN.md §2.4:
``rho_bundle = c * rho_repo + 1``. The round sphere is ``rho_bundle = 1``
here, not 0 -- and that is what makes the whole thing work, because the
kernel condition on a constant ``rho`` is ``(l + 1/2)**2 = rho**2``, which at
``rho = 1`` selects ``l = 1/2`` and nothing else: the constant spinor. Apply
that offset exactly once (:func:`rho_on_grid` is the one place).

**Gate status** (``tests/test_dirac_spectral.py``):

* **P9.A** round sphere -> ``gamma = 0`` (1.3e-16) with the spinor entirely at
  ``l = 1/2``, kernel multiplicity exactly 2. Passes.
* **P9.B** the coupling is Hermitian to 2.8e-16. Passes.
* **P9.C** branch points: :func:`min_spinor_modulus` locates where
  ``|lambda| -> 0``, i.e. where no surface exists for that rho.
* **P9.D** spectral convergence in band.
* **P9.E** agreement with :mod:`shapegen.geom.dirac`, via the conformal
  factor -- which is the row that does not need a frame, see
  :func:`conformal_factor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..spectral import swsh as sw

__all__ = ["DiracSpectral", "spinor_band", "potential_band_extra",
           "coupling_grid", "rho_on_grid", "coupling_matrix",
           "dirac_matrix", "solve_spinor", "conformal_factor",
           "log_conformal_coeffs",
           "min_spinor_modulus"]


#: Degrees of head-room :func:`log_conformal_coeffs` needs above its output
#: band. See :func:`spinor_band`; 4 is measured **at L = 12/16 on a smooth
#: fixture**, and 1 already recovers most of it there. It is **not enough at
#: L = 32 on a real face**, where 8 is needed and 4 costs 1.28-1.50x -- see
#: :func:`spinor_band` for the sweep and the proposed ``max(4, L // 4)`` rule.
def potential_band_extra(L: int) -> int:
    """How much to pad the spinor band beyond ``L`` before taking
    :func:`log_conformal_coeffs` -- ``max(4, L//4)``.

    A fixed 4 was tuned at L=12/16 and does **not** carry: the padding has to
    scale with the band, because the Galerkin truncation it compensates for
    is a ``rho`` coupling that moves degree ``l`` to ``l +/- L``. Measured at
    L=32 on two FaceScape heads, score against a converged referee at
    ``extra = 2 / 4 / 8 / 12``::

        head A   0.1183  0.1109  0.0738  0.0746
        head B   0.0991  0.1157  0.0900  0.0917

    8 and 12 agree to 1-2%; 4 costs 1.28-1.50x. ``L//4`` gives 8 at L=32 and
    reproduces the old value at L<=16, so nothing below that band moves.

    **That sweep ran on the aliased coupling grid** (before
    :func:`coupling_grid`, 2026-09-26), and part of what the padding bought
    was pushing the aliasing away from ``l <= L``. On the exact grid, the
    potential's A-error against the decoded surface (same two heads) is::

        extra      4      6      8      10     12     16
        aliased  0.25    --    0.09    --    0.036   --
        exact    0.10   0.07   0.04   0.035  0.033  0.035

    The ~0.033 floor is the referee's own error. The rule is unchanged;
    ``extra = 12`` would reach the floor at 1.8x the solve time.
    """
    return max(4, L // 4)


def spinor_band(L: int, extra: int = 0) -> float:
    """Half-integer spinor band for an integer ``rho`` band ``L``.

    ``L - 1/2``, which at ``L = 32`` gives 1056 modes at each of the two
    spins -- PLAN.md's "2112 complex unknowns at L=32", exactly.

    **``extra`` is not optional if you are going to take
    :func:`log_conformal_coeffs` at band ``L``.** The spinor itself is
    resolved at ``L - 1/2``: measured on the ``ell^-2`` fixture at ``L = 12``,
    the power above ``l = 11.5`` is 2.8e-7 of the total at amplitude 1 and
    5.7e-6 at amplitude 4. What is *not* resolved is the Galerkin system's
    top rows: the ``rho`` coupling moves degree ``l`` to ``l +/- L``, so the
    equations for the highest degrees are missing their partners and the
    truncation residual lands there. ``|lambda|^2`` then beats that corrupted
    top mode against the dominant ``l = 1/2`` one and deposits it at degree
    ``Ls + 1/2 = L`` -- the *top* of the band the connection uses, and the one
    ``edth`` weights most heavily (``sqrt(L(L+1))``).

    Measured, band-``L`` potential against the conformal factor of the
    surface ``codec.decode`` builds from the same ``rho`` (``ell^-2``
    fixture, relative error of ``A = i edth phi``)::

        L=12 amp=1   Ls=11.5: 0.0869   12.5: 0.0048  14.5: 0.0046  23.5: 0.0046
        L=12 amp=4   Ls=11.5: 0.0960   12.5: 0.0095  14.5: 0.0062  23.5: 0.0060
        L=16 amp=1   Ls=15.5: 0.0627   16.5: 0.0056  18.5: 0.0054  31.5: 0.0055
        L=16 amp=4   Ls=15.5: 0.0621   16.5: 0.0086  18.5: 0.0069  31.5: 0.0069

    One extra degree removes 95% of the error and four are converged, at
    every amplitude and both bands. The cost is ``(1 + extra/L)^2`` in modes.

    **Those rows are L=12/16 on a smooth fixture, and ``extra = 4`` does not
    carry to L=32 on a real face (measured 2026-09-21).** Sweeping ``extra``
    on two FaceScape heads at L=32, scoring the resulting potential through
    the anneal (:func:`~shapegen.quad.field.area_distortion`, geometric mean
    over three schedules, after refinement) and against the conformal factor
    of ``codec.decode``'s own ``icosphere(6)`` surface::

        face #339   extra= 0  1  2      4       8       12
                    score    0.1738  0.1183  0.1109  0.0738  0.0746
                    vs truth 0.3534  0.3279  0.3187  0.3030  0.3041
        face #254   score    0.2243  0.0991  0.1157  0.0900  0.0917
                    vs truth 0.3337  0.3112  0.3066  0.2897  0.2915

    ``extra = 8`` and ``extra = 12`` agree to 1-2% on both heads and on both
    measures; ``extra = 4`` does not, and is worth **1.28-1.50x** of score.
    The shipped constant is therefore under-padded at L=32 and the right rule
    looks like ``max(4, L // 4)`` -- which is a no-op at every band this
    repository tests at (L <= 16) and would only cost at L >= 20, where the
    solve is opt-in anyway. It has not been changed here because the new
    value is characterised at one bandlimit on two faces; measure it across
    bands before moving the default.
    """
    return float(L + extra) - 0.5


def coupling_grid(Ls: float, L: int):
    """``(theta, phi, weights)`` exact for the ``rho`` coupling at spinor band
    ``Ls`` and ``rho`` band ``L``.

    :func:`~shapegen.spectral.swsh.half_grid` is exact for a product of *two*
    spinor modes, degree ``2 Ls``. The coupling integrand
    ``conj(Y_l') rho Y_l`` has degree ``2 Ls + L`` in ``cos theta`` (the
    half-angle factors pair up into integer powers because
    ``|m-s| + |m'-s| + |m-m'|`` is even), so it needs
    ``ceil((2 Ls + L + 1) / 2)`` Gauss-Legendre nodes and more than
    ``2 Ls + L`` azimuthal samples. On ``half_grid`` at ``L = 32``,
    ``Ls = 39.5`` the top ~30 degrees of the coupling alias.
    """
    return _coupling_grid(float(Ls), int(L))


@lru_cache(maxsize=None)
def _coupling_grid(Ls: float, L: int):
    nth = int(np.ceil((2.0 * Ls + L + 1.0) / 2.0))
    nph = int(np.floor(2.0 * Ls)) + L + 2
    x, w = np.polynomial.legendre.leggauss(nth)
    theta = np.arccos(x)
    phi = 2.0 * np.pi * np.arange(nph) / nph
    return theta, phi, w[:, None] * (2.0 * np.pi / nph)


@lru_cache(maxsize=4)
def _coupling_basis(s: float, Ls: float, L: int) -> np.ndarray:
    """``(nmodes, Ntheta*Nphi)`` spin-``s`` basis on :func:`coupling_grid` --
    :func:`~shapegen.spectral.swsh.half_basis`'s layout, on the padded grid.
    ~170 MB per spin at ``L = 32``, hence the small cache."""
    th, ph, _w = coupling_grid(Ls, L)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    return np.ascontiguousarray(sw.half_eval_basis(s, Ls, TH.ravel(), PH.ravel()).T)


def rho_on_grid(rho_hat: np.ndarray, L: int, Ls: float, c: float = 1.0):
    """This repository's real-SH ``rho_hat`` -> the bundle's ``rho``, sampled
    on :func:`coupling_grid` ``(Ls, L)``.

    ``rho_bundle = c * rho_repo + 1`` (PLAN.md §2.4). **The one place that
    offset is applied** -- twice, or again inside a re-measurement pass,
    shifts the realised curvature by a constant and the failure reads as a
    scale error in the output rather than as a bug.
    """
    from ..quad.field import real_to_complex

    th, ph, _w = coupling_grid(Ls, L)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    C = real_to_complex(np.asarray(rho_hat, float), L)
    Y = sw.eval_basis(0, L, TH.ravel(), PH.ravel())
    flat = np.array([C[l, l + m] for l, m in sw.mode_list(0, L)])
    field = (Y @ flat).reshape(TH.shape)
    assert np.abs(field.imag).max() < 1e-8, "rho must be real"
    return c * field.real + 1.0


def coupling_matrix(rho_grid: np.ndarray, s: float, Ls: float,
                    L: int) -> np.ndarray:
    """Multiplication by ``rho`` as a matrix in the spin-``s`` basis.

    Built **by transform** (PLAN.md: "build the coupling by transform, not by
    3j symbols"): a coefficient vector is synthesised, multiplied pointwise,
    and analysed back, which in matrix form is
    ``B* diag(rho . w) B^T``. Hermitian for real ``rho`` -- gate P9.B, and
    the cheapest catch for a conjugation error.

    ``rho_grid`` is :func:`rho_on_grid`'s output for the same ``(L, Ls)``;
    the quadrature is :func:`coupling_grid`, exact for the triple product.
    Until 2026-09-26 this ran on ``half_grid(Ls)``, which is exact only to
    degree ``2 Ls`` and aliased the top couplings at ``L = 32``.
    """
    B = _coupling_basis(float(s), float(Ls), int(L))
    _th, _ph, w = coupling_grid(Ls, L)
    return B.conj() @ ((rho_grid * w).ravel()[:, None] * B.T)


def dirac_matrix(rho_grid: np.ndarray, Ls: float, L: int) -> np.ndarray:
    """The operator ``G`` above, as a dense ``(2n, 2n)`` complex matrix with
    the spin ``+1/2`` block first.

    Left dense deliberately: 2112 unknowns at ``L = 32`` is 71 MB and a few
    seconds, which is what PLAN.md budgeted, and a matrix-free route buys
    nothing until the band grows.
    """
    Mp = coupling_matrix(rho_grid, +0.5, Ls, L)
    Mm = coupling_matrix(rho_grid, -0.5, Ls, L)
    n = Mp.shape[0]
    G = np.zeros((2 * n, 2 * n), dtype=complex)
    G[:n, :n] = Mp                                        # rho . psi1
    G[:n, n:] = np.diag(sw.half_edth_factor(-0.5, Ls))    # eth : -1/2 -> +1/2
    G[n:, :n] = np.diag(sw.half_edthbar_factor(+0.5, Ls))  # ethbar: +1/2 -> -1/2
    G[n:, n:] = -Mm                                       # -rho . psi2
    return G


@dataclass
class DiracSpectral:
    """The solve's result, with the diagnostics its gates are stated in."""
    psi1: np.ndarray        # spin +1/2 coefficients, half_modes order
    psi2: np.ndarray        # spin -1/2 coefficients
    gamma: float            # smallest singular value of G
    gap: float              # the next one -- the kernel is simple iff this is big
    multiplicity: int       # singular values within 1e-8 of gamma
    Ls: float
    L: int
    c: float

    @property
    def gamma_squared(self) -> float:
        """PLAN.md §2.5: this is what ``geom/dirac.py``'s ``gamma_mesh`` is
        predicted to equal, because that one is a Gram residual."""
        return self.gamma ** 2


def solve_spinor(rho_hat: np.ndarray, L: int, c: float = 1.0,
                 Ls: float | None = None) -> DiracSpectral:
    """Solve ``(D - rho) lambda = gamma lambda`` for the smallest ``|gamma|``.

    Returns the spinor as its two spin-weighted halves, plus ``gamma``. The
    realised curvature is ``rho - gamma`` (PLAN.md), so a large ``gamma`` is a
    first-class statement about the output rather than a convergence
    complaint -- on real face codes it is not small.
    """
    Ls = spinor_band(L) if Ls is None else Ls
    G = dirac_matrix(rho_on_grid(rho_hat, L, Ls, c), Ls, L)
    _U, S, Vh = np.linalg.svd(G)
    n = G.shape[0] // 2
    v = Vh[-1].conj()
    mult = int((S < S[-1] + 1e-8).sum())
    return DiracSpectral(psi1=v[:n], psi2=v[n:], gamma=float(S[-1]),
                         gap=float(S[-mult - 1]) if mult < len(S) else float("inf"),
                         multiplicity=mult, Ls=Ls, L=L, c=c)


def conformal_factor(res: DiracSpectral, theta=None, phi=None) -> np.ndarray:
    """``|lambda|^2 = |psi1|^2 + |psi2|^2`` -- the conformal factor of the
    immersion, on the solve's own grid or at given ``(theta, phi)``.

    **This is the frame-free half of the spinor, and the only part that can be
    compared with the mesh solver without further work.** ``psi1`` and
    ``psi2`` are spin ±1/2, so they are not functions on the sphere at all: a
    rotation of the tangent frame by ``alpha`` multiplies them by
    ``e^{-i alpha/2}`` and ``e^{+i alpha/2}``. Their *moduli* are untouched,
    so ``|lambda|^2`` is an honest scalar field -- which is why PLAN.md §P9.E's
    conformal-factor row is the one this module can answer, and why it is what
    the connection in :mod:`shapegen.quad.field` is fed.
    """
    f1 = sw.half_synth(res.psi1, +0.5, res.Ls)
    f2 = sw.half_synth(res.psi2, -0.5, res.Ls)
    return np.abs(f1) ** 2 + np.abs(f2) ** 2


def log_conformal_coeffs(res: DiracSpectral, L: int | None = None) -> np.ndarray:
    """``log|lambda|^2`` as spin-0 coefficients in :mod:`swsh`'s dense
    ``C[l, l+m]`` layout -- the exact conformal potential, for
    :func:`shapegen.quad.field.curvature_potential`.

    The ``l = 0`` mode is dropped, because the immersion is only defined up
    to scale and the connection only sees ``eth phi`` anyway.

    **``res`` must have been solved with head-room above ``L``** --
    ``spinor_band(L, potential_band_extra(L))``, not ``spinor_band(L)``. At
    ``Ls = L - 1/2`` the degree-``L`` coefficients come back 6-9x too large
    and the resulting connection is 9% off instead of 0.5%; see
    :func:`spinor_band` for the mechanism and the numbers. It is not a
    quadrature problem: padding the grid by 3x changes the spike by 0.5%,
    raising ``Ls`` by one degree removes 95% of it.

    The quadrature itself is the solve's own grid, which carries this
    comfortably: at ``Ls`` it has ``⌊Ls⌋+2`` Gauss-Legendre nodes in theta and
    ``⌊2Ls⌋+2`` in phi, against the ``L+1`` / ``2L+1`` a band-``L`` spin-0
    projection needs. Measured against a 6x-padded grid, the whole
    coefficient vector moves by 6e-4 relative at ``Ls = L - 1/2`` and not at
    all once ``Ls > L``.
    """
    L = int(res.L) if L is None else int(L)
    if res.Ls < L + 0.5:
        import warnings
        warnings.warn(
            f"log_conformal_coeffs at band {L} from a spinor solved at "
            f"Ls={res.Ls}: the top degrees will be truncation residue, not "
            f"signal. Solve with Ls=spinor_band({L}, potential_band_extra({L})).",
            RuntimeWarning, stacklevel=2)
    th, ph, w = sw.half_grid(res.Ls)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    f = np.log(np.maximum(conformal_factor(res), 1e-300))
    Y = sw.eval_basis(0, L, TH.ravel(), PH.ravel())
    # w is a COLUMN (Nθ,1) that broadcasts over φ; it must be broadcast before
    # ravelling, never ravelled on its own
    flat = Y.conj().T @ (f * w).ravel()
    C = sw.empty_coeffs(L)
    for i, (l, m) in enumerate(sw.mode_list(0, L)):
        C[l, l + m] = flat[i]
    C[0, 0] = 0.0
    return C


def min_spinor_modulus(res: DiracSpectral) -> float:
    """**Gate P9.C.** ``min |lambda|`` over the grid, normalised by its mean.

    Approaching zero means the spinor vanishes somewhere, and where
    ``lambda = 0`` the immersion degenerates -- no surface exists for that
    ``rho``. PLAN.md expects this to bite on real scanned faces (F3) before it
    bites on the synthetic fixtures.
    """
    cf = conformal_factor(res)
    _th, _ph, w = sw.half_grid(res.Ls)
    W = np.broadcast_to(w, cf.shape)
    return float(np.sqrt(cf.min() / ((cf * W).sum() / W.sum())))

"""Mesh-free cone placement: a spin-4 cross field found by spectral
Ginzburg-Landau annealing, cones read off its zeros, and the metrics that
score a cone set (ported from ``03_spectral/solver.py``
and ``gl_spec.py``).

The field ``ψ`` is spin-4 (a cross raised to the 4th power, so single-valued).
Its connection, for an immersion conformal to the round sphere with conformal
potential ``φ`` (:func:`curvature_potential`), is ``∇ = ∇₀ - i·s·A`` with
``A = i·ð(φ)``:

    C ψ = -Δ₀ψ + s²|A|²ψ + 2is·(A·∇₀ψ),   A·∇₀ψ = ½(Ā·ðψ + A·ð̄ψ)

``div A = 0`` (``A`` is a rotated gradient), so there is no extra term.

Not included: ``blend.py``'s smoothness/alignment control; this module
only anneals ``ψ`` towards unit modulus.
"""

from __future__ import annotations

import functools

import numpy as np

from ..spectral import swsh as _sw

__all__ = [
    "real_to_complex", "curvature_potential", "solve_potential_spinor",
    "connection_field",
    "apply_C", "assemble_C", "linear_start", "dealias", "epsilon_ceiling",
    "anneal_schedule", "ginzburg_landau", "DEFAULT_LEVELS",
    "cones_from_field", "winding_total", "rho_rms",
    "decoded_surface", "area_distortion",
]


# --------------------------------------------------------------------------- #
# The real/complex coefficient boundary (the "one conversion pair")          #
# --------------------------------------------------------------------------- #

def real_to_complex(c: np.ndarray, L: int) -> np.ndarray:
    """This repo's real-SH coefficients (layout: ``l`` ascending, ``m`` from
    ``-l`` to ``l``, :func:`~shapegen.geom.encode.real_sh_basis`'s convention)
    -> ``swsh``'s dense complex ``C[l, l+m]``, for a real-valued field.

    For ``m > 0``, real_sh_basis's columns are ``√2·(-1)^m·Re(Y_lm)`` (index
    ``+m``) and ``√2·(-1)^m·Im(Y_lm)`` (index ``-m``). Writing the field as
    ``Σ c_m·Yreal_m`` and collecting the ``Y_{l,±m}`` terms (using
    ``conj(Y_lm) = (-1)^m Y_{l,-m}``) gives, for ``m = 1..l``::

        C[l, l+m] = (-1)^m/√2 · (c[+m] - i·c[-m])
        C[l, l-m] = 1/√2      · (c[+m] + i·c[-m])

    and ``C[l, l] = c[0]`` for ``m = 0``. Checked against ``real_sh_basis``
    synthesis to 1.7e-14 (``test_real_to_complex_matches_real_sh_basis``).
    """
    C = _sw.empty_coeffs(L)
    idx = 0
    for l in range(L + 1):
        C[l, l] = c[idx + l]
        for m in range(1, l + 1):
            cp, cm = c[idx + l + m], c[idx + l - m]
            C[l, l + m] = ((-1) ** m / np.sqrt(2.0)) * (cp - 1j * cm)
            C[l, l - m] = (1.0 / np.sqrt(2.0)) * (cp + 1j * cm)
        idx += 2 * l + 1
    return C


# --------------------------------------------------------------------------- #
# P8.3 -- the connection operator                                             #
# --------------------------------------------------------------------------- #

def rho_rms(rho_hat: np.ndarray) -> float:
    """``sqrt(∫ρ² dA / 4π)`` straight from the coefficients.

    The real SH basis is orthonormal, so ``∫ρ² dA = Σ c²`` and no synthesis
    is needed. Matches a synthesised ``ρ.std()`` to better than 0.3%.
    """
    c = np.asarray(rho_hat, dtype=float)
    return float(np.sqrt((c ** 2).sum() / (4.0 * np.pi)))


def curvature_potential(rho_hat: np.ndarray, L: int, spinor=None) -> np.ndarray:
    """``ρ̂ -> φ``, the conformal potential the connection is a gradient of:
    ``-log|λ|²`` of the immersion's spinor, via
    :func:`shapegen.geom.dirac_spectral.log_conformal_coeffs` (complex swsh
    coefficients, spin 0).

    Pass ``spinor`` (a :class:`~shapegen.geom.dirac_spectral.DiracSpectral`
    from :func:`solve_potential_spinor`) to reuse one solve -- it is the
    expensive step at L=32, and :func:`shapegen.quad.lift.lift_spectral`
    needs the same one. Trap: a spinor solved at a different band (e.g. plain
    ``spinor_band(L)``) is accepted silently but gets ``φ``'s top degree 6-9x
    wrong (see :func:`~shapegen.geom.dirac_spectral.log_conformal_coeffs`).

    The sign is this module's convention for ``A = i·ðφ``. It replaced the
    linear potential ``μ = 1 + 2ρ``.
    """
    if spinor is None:
        spinor = solve_potential_spinor(rho_hat, L)
    from ..geom.dirac_spectral import log_conformal_coeffs
    return -log_conformal_coeffs(spinor, L)


def solve_potential_spinor(rho_hat: np.ndarray, L: int):
    """The spinor :func:`curvature_potential` needs, at the padded band
    ``spinor_band(L, potential_band_extra(L))``. Solve once and pass it to
    both :func:`curvature_potential` and
    :func:`shapegen.quad.lift.lift_spectral`."""
    from ..geom.dirac_spectral import (potential_band_extra, solve_spinor,
                                       spinor_band)
    return solve_spinor(rho_hat, L, Ls=spinor_band(L, potential_band_extra(L)))


def connection_field(phi: np.ndarray) -> np.ndarray:
    """``A = i·ð(φ)``: the spin-1 connection correction, a rotated gradient of
    the curvature potential."""
    return 1j * _sw.edth(phi, 0)


def apply_C(psi_c: np.ndarray, S: int, L: int, A: np.ndarray) -> np.ndarray:
    """``C ψ`` for coefficients ``psi_c`` at spin ``S`` (see module docstring).

    ``A`` is spin-1 coefficients at the same bandlimit ``L`` (``swsh``'s grid
    depends only on ``L``). Reference form only: :func:`assemble_C` is
    cross-checked against it and does not call it.
    """
    eig = _sw.laplacian_eig(L, S)          # -[l(l+1) - S^2]
    neg_lap0 = -eig[:, None] * psi_c       # -Lap_0 psi = [l(l+1)-S^2] psi
    dp = _sw.edth(psi_c, S)                # spin S -> S+1
    dm = _sw.edthbar(psi_c, S)             # spin S -> S-1
    A_g = _sw.synth(A, 1, L)
    dp_g = _sw.synth(dp, S + 1, L)
    dm_g = _sw.synth(dm, S - 1, L)
    psi_g = _sw.synth(psi_c, S, L)
    adv = 0.5 * (np.conj(A_g) * dp_g + A_g * dm_g)
    extra = (S * S) * (np.abs(A_g) ** 2) * psi_g + 2j * S * adv
    return neg_lap0 + _sw.analy(extra, S, L)


def assemble_C(A: np.ndarray, S: int, L: int):
    """Dense assembly of ``C`` at spin ``S`` as three matrix products (far
    less memory traffic than one :func:`apply_C` per column).

    With ``Y_s`` = ``basis_matrix(s, L)`` (so ``synth(c, s) = Y_s.T @ c`` and
    ``analy(f, s) = Y_s.conj() @ (weights * f)``):

        M_B    = Y_S.conj() @ diag(w·|A|²) @ Y_S.T                (n_S x n_S)
        M_up   = Y_S.conj() @ diag(w·Ā) @ Y_{S+1}.T @ diag(ð-factors)
        M_down = Y_S.conj() @ diag(w·A) @ Y_{S-1,last n_S rows}.T @ diag(ð̄-factors)
        C      = diag(l(l+1)-S²) + S²·M_B + iS·(M_up + M_down)

    ``M_up`` is zero on the ``l=S`` block (``ð`` annihilates it); ``M_down``
    uses ``Y_{S-1}``'s last ``n_S`` rows, since ``mode_list(S-1, L)`` is
    ``[l=S-1 block] + mode_list(S, L)``. Matches the ``apply_C``-per-column
    assembly to 1.2e-16 at L=8.

    ``C`` is self-adjoint analytically, but quadrature aliasing of the
    ``|A|²`` and ``A·∇`` products leaves ~1% asymmetry; the result is
    symmetrised so ``eigh`` is safe.

    Returns ``(Cmat, modes)``, ``modes = swsh.mode_list(S, L)`` giving each
    row/column's ``(l, m)``.
    """
    from ..spectral.grids import gl_grid, solid_angle_weights  # noqa: PLC0415

    grid = gl_grid(L)
    weights = np.broadcast_to(solid_angle_weights(grid),
                              (grid.ntheta, grid.nphi)).ravel()
    A_g = _sw.synth(A, 1, L).ravel()
    absA2 = np.abs(A_g) ** 2

    Y_S, rows_S, _cols_S = _sw.basis_matrix(S, L)
    Y_Sp1, _rows_p1, _cols_p1 = _sw.basis_matrix(S + 1, L)
    Y_Sm1, _rows_m1, _cols_m1 = _sw.basis_matrix(S - 1, L)
    l_arr = rows_S.astype(float)
    n = len(l_arr)

    M_B = (Y_S.conj() * (weights * absA2)[None, :]) @ Y_S.T

    n_p1 = Y_Sp1.shape[0]
    up_factors = np.sqrt(np.clip((l_arr[-n_p1:] - S) * (l_arr[-n_p1:] + S + 1),
                                 0.0, None))
    Mid_up = (Y_S.conj() * (weights * np.conj(A_g))[None, :]) @ Y_Sp1.T
    M_up = np.zeros((n, n), dtype=complex)
    M_up[:, -n_p1:] = Mid_up * up_factors[None, :]

    Y_Sm1_last = Y_Sm1[-n:, :]      # mode_list(S-1,L)'s last n rows == mode_list(S,L)
    down_factors = -np.sqrt(np.clip((l_arr + S) * (l_arr - S + 1), 0.0, None))
    M_down = ((Y_S.conj() * (weights * A_g)[None, :]) @ Y_Sm1_last.T
             * down_factors[None, :])

    diagL = (l_arr * (l_arr + 1) - S * S).astype(complex)
    Cmat = np.diag(diagL) + (S * S) * M_B + 1j * S * (M_up + M_down)
    Cmat = 0.5 * (Cmat + Cmat.conj().T)
    return Cmat, _sw.mode_list(S, L)


def linear_start(Cmat: np.ndarray, modes, S: int, L: int) -> np.ndarray:
    """Smallest eigenvector of ``C`` (assumed already Hermitian, as
    :func:`assemble_C` returns it) -- the linear-theory starting field.

    Returns the ``modes``-ordered coefficient vector, rescaled by
    ``1/√(peak modulus)`` -- the same soft rescaling :func:`ginzburg_landau`
    applies per level (peak ``m -> √m``, not unit peak).
    """
    _w, V = np.linalg.eigh(Cmat)
    psi = V[:, 0].astype(complex)
    pg = _sw.synth(_vec_to_full(psi, modes, L), S, L)
    return psi / np.sqrt(np.abs(pg).max())


# --------------------------------------------------------------------------- #
# P8.4 -- Ginzburg-Landau                                                     #
# --------------------------------------------------------------------------- #

def dealias(vec: np.ndarray, modes, L: int, frac: float = 0.62) -> np.ndarray:
    """Zero out modes above ``frac * L`` -- the nonlinear term's cubic product
    aliases high degrees back down, so its own spectrum must be truncated
    before it re-enters the flow."""
    out = vec.copy()
    for i, (l, _m) in enumerate(modes):
        if l > frac * L:
            out[i] = 0.0
    return out


def _vec_to_full(vec: np.ndarray, modes, L: int) -> np.ndarray:
    full = _sw.empty_coeffs(L)
    for i, (l, m) in enumerate(modes):
        full[l, l + m] = vec[i]
    return full


def _full_to_vec(full: np.ndarray, modes) -> np.ndarray:
    return np.array([full[l, l + m] for (l, m) in modes], dtype=complex)


TAU_COEFF = 0.4
"""``τ = TAU_COEFF · ε²``, the flow's step size. Note :func:`epsilon_ceiling`
does *not* depend on it — see there."""

EPS_SAFETY = 0.85
"""How close to :func:`epsilon_ceiling` a schedule is allowed to run.
Measured: the collapse boundary is sharp, and 0.85 of it still anneals."""

DEFAULT_LEVELS = (2.0, 1.74, 1.52, 1.32, 1.15, 1.0)
"""Geometric ``2.0 -> 1.0`` in units of ``π/L``, six levels. Across 13
(L, amplitude) cases at L = 12/16/24: geometric-mean :func:`area_distortion`
0.0735, worst 0.081, no failures; other in-window schedules land within 6%,
so what matters is staying under :func:`epsilon_ceiling`."""


def epsilon_ceiling(Cmat: np.ndarray) -> float:
    """The largest ``ε`` at which :func:`ginzburg_landau` can hold a field up
    at all: ``1/√(λ_min(C))``. **Above it the flow is not an anneal.**

    Linearise the semi-implicit step about ``ψ = 0``. The nonlinear term
    ``(1/ε²)(1 - |ψ|²)ψ`` becomes ``(1/ε²)ψ``, so with ``τ = c·ε²`` one step
    multiplies each eigenmode of ``C`` by

        (1 + τ/ε²) / (1 + τλ) = (1 + c) / (1 + c·ε²λ),

    which exceeds 1 exactly when ``ε²λ < 1``, independently of ``c``. If even
    ``λ_min`` fails that, every mode decays.

    Trap (silent): the per-level rescale keeps the field alive, so above the
    ceiling the flow does not fail -- it returns the slowest-decaying
    eigendirection, i.e. :func:`linear_start` (overlap 0.9999+), and scores
    like it. Raising ``iters`` only underflows it to exactly 0.

    ``λ_min`` is nearly bandlimit-independent (spin-4 floor ``l(l+1)-S² = 4``
    at ``l = S``, lowered by curvature), so the ceiling is an absolute ``ε``
    (0.5 on the round sphere, 0.575 on F1 amp 1, 0.653 at amp 4). In ``π/L``
    units it therefore grows with L (1.91 / 2.55 / 3.82 / 5.09 at L = 12 /
    16 / 24 / 32, round sphere): a fixed ``π/L`` schedule is most at risk at
    *small* L. Returns ``inf`` if ``λ_min <= 0``.
    """
    try:
        from scipy.linalg import eigvalsh  # noqa: PLC0415
        lam = float(eigvalsh(Cmat, subset_by_index=[0, 0])[0])
    except Exception:                      # pragma: no cover - scipy fallback
        lam = float(np.linalg.eigvalsh(Cmat)[0])
    if lam <= 0.0:
        return float("inf")
    return 1.0 / np.sqrt(lam)


def anneal_schedule(Cmat: np.ndarray, L: int, levels=DEFAULT_LEVELS,
                    safety: float = EPS_SAFETY) -> tuple:
    """``levels`` (in units of ``π/L``) clamped to ``safety ×``
    :func:`epsilon_ceiling`.

    The optimum scales with ``π/L`` (1-1.5 at L=12 and 16) but the ceiling is
    an absolute ``ε``, so at small L the top of a fixed schedule can cross it
    (``DEFAULT_LEVELS``' 2.0 is 0.524 at L=12 vs a round-sphere ceiling of
    0.5). Clamping keeps one schedule valid at every L.
    """
    cap = safety * epsilon_ceiling(Cmat) * L / np.pi
    return tuple(min(float(ef), cap) for ef in levels)


def ginzburg_landau(psi0: np.ndarray, Cmat: np.ndarray, modes, S: int, L: int,
                     levels=None, iters: int = 160,
                     check: bool = True) -> np.ndarray:
    """Semi-implicit Ginzburg-Landau flow at spin ``S``, one factorisation per
    ``ε`` level in ``levels``.

    ``psi0`` is a ``modes``-ordered coefficient vector (e.g.
    :func:`linear_start`); returns the annealed vector, same layout.

    ``levels`` are in units of the **field's own resolution** ``π/L``, not of
    the finer grid :func:`cones_from_field` reads zeros from -- scaling to the
    output grid leaves the cones essentially unmoved (1.5x worse than a plain
    cube). They default to :func:`anneal_schedule` and are bounded on both
    sides:

    * above by :func:`epsilon_ceiling`, where the result silently collapses
      to the linear start; ``check=True`` (default) raises instead, and
      ``check=False`` exists only to reproduce the collapse in tests;
    * below by the grid: under ~0.5 the vortex core is narrower than the
      field's resolution and quality degrades (one level at L=12 F1: 0.075
      at 1.2, 0.084 at 0.5, 0.175 at 0.2). Do not drive ``ε`` to 0.

    Raises ``FloatingPointError`` if the field collapses to zero anyway.
    """
    if levels is None:
        levels = anneal_schedule(Cmat, L)
    levels = tuple(float(ef) for ef in levels)
    n = len(modes)
    hf = np.pi / L
    if check:
        ceil = epsilon_ceiling(Cmat)
        bad = [ef for ef in levels if ef * hf >= ceil]
        if bad:
            raise ValueError(
                f"anneal levels {bad} are at or above the epsilon ceiling "
                f"{ceil / hf:.3f} (= {ceil:.4f} absolute, 1/sqrt(lambda_min(C))): "
                "every mode decays there and the flow returns the linear "
                "start, not an annealed field. Use field.anneal_schedule(), "
                "or pass check=False to reproduce the collapse deliberately.")
    psi = psi0.astype(complex).copy()
    for ef in levels:
        eps = ef * hf
        tau = TAU_COEFF * eps ** 2
        Ainv = np.linalg.inv(np.eye(n) + tau * Cmat)
        for _ in range(iters):
            pg = _sw.synth(_vec_to_full(psi, modes, L), S, L)
            nl = (1.0 / eps ** 2) * (1.0 - np.abs(pg) ** 2) * pg
            nl_vec = _full_to_vec(_sw.analy(nl, S, L), modes)
            rhs = psi + tau * dealias(nl_vec, modes, L)
            psi = Ainv @ rhs
        pg = _sw.synth(_vec_to_full(psi, modes, L), S, L)
        peak = np.abs(pg).max()
        if not np.isfinite(peak) or peak == 0.0:
            raise FloatingPointError(
                f"psi collapsed to {peak} at eps = {ef:.3f}*pi/L; the "
                "schedule is above the epsilon ceiling (see "
                "field.epsilon_ceiling)")
        psi = psi / np.sqrt(peak)
    return psi


# --------------------------------------------------------------------------- #
# Reading cones off the field                                                 #
# --------------------------------------------------------------------------- #

def _fine_grid(L: int, over: float = 7.0):
    """A finer Gauss-Legendre x equiangular grid than ``gl_grid(L)``, used
    only to read zeros off a bandlimit-``L`` field -- never to integrate."""
    nth = int(over * L) + 8
    nph = 2 * nth
    x, _w = np.polynomial.legendre.leggauss(nth)
    th = np.arccos(x)
    ph = np.arange(nph) * 2.0 * np.pi / nph
    return th, ph


@functools.lru_cache(maxsize=8)
def _fine_grid_basis(S: int, L: int, over: float):
    """``(th, ph, Y)`` for the fine grid, ``Y`` the ``eval_basis`` design
    matrix. Cached: callers evaluate the same grid repeatedly (linear start,
    then GL result) and ``eval_basis`` cannot use ``swsh``'s own cache.
    """
    th, ph = _fine_grid(L, over)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    Y = _sw.eval_basis(S, L, TH.ravel(), PH.ravel())
    return th, ph, Y


def _field_on_fine_grid(vec: np.ndarray, S: int, L: int, th, ph):
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    Y = _sw.eval_basis(S, L, TH.ravel(), PH.ravel())
    return (Y @ vec).reshape(TH.shape)


def cones_from_field(vec: np.ndarray, S: int, L: int, over: float = 7.0):
    """Cones as cell windings of the spin-``S`` field on a fine grid
    (``gl_spec.py``'s ``cones_of``): the plaquette phase sum around each cell
    gives an integer winding ``k``; a nonzero ``k`` is a cone of index
    ``+k/S`` at that cell's center.

    ``vec`` is a ``mode_list(S, L)``-ordered coefficient vector; ``over`` sets
    the fine grid's oversampling. Returns a list of ``(unit position (3,),
    index)``.

    Sign convention: indices sum to ``+χ = +2`` (a round-sphere cross field
    gives 8 cones of ``+1/4``). Trap: a sign flip is not cosmetic -- the flat
    form ``prod (z - C_k)^(-idx_k) dz`` is regular at the chart's far pole
    only when the sum is ``+2``, and at ``-2`` :func:`shapegen.quad.grid.quad_grid`
    produces garbage (Euler characteristic -3240 instead of 2).
    """
    th, ph, Y = _fine_grid_basis(S, L, over)
    v = (Y @ vec).reshape(len(th), len(ph))
    nth, nph = v.shape
    cones = []
    for i in range(nth - 1):
        for j in range(nph):
            j2 = (j + 1) % nph
            a, b, c, d = v[i, j], v[i, j2], v[i + 1, j2], v[i + 1, j]
            w = (np.angle(b / a) + np.angle(c / b) + np.angle(d / c)
                 + np.angle(a / d))
            k = int(round(w / (2.0 * np.pi)))
            if k != 0:
                thc = 0.5 * (th[i] + th[i + 1])
                phc = ph[j] + np.pi / nph
                pos = np.array([np.sin(thc) * np.cos(phc),
                                np.sin(thc) * np.sin(phc), np.cos(thc)])
                cones.append((pos, k / float(S)))
    return cones


def winding_total(field_grid: np.ndarray) -> float:
    """Total winding of a synthesised field over its whole (periodic-in-φ)
    grid (shape ``(ntheta, nphi)``), in units of ``2π`` -- a cheap check that
    the field is a genuine section. Poincaré-Hopf fixes it at ``+2S`` on a
    genus-0 surface (``+8`` at ``S=4``), i.e. ``S`` times
    :func:`cones_from_field`'s index sum."""
    v = field_grid
    tot = 0.0
    for i in range(v.shape[0] - 1):
        a = v[i]
        b = v[i, np.r_[1:v.shape[1], 0]]
        c = v[i + 1, np.r_[1:v.shape[1], 0]]
        d = v[i + 1]
        tot += (np.angle(b / a) + np.angle(c / b) + np.angle(d / c)
                + np.angle(a / d)).sum()
    return tot / (2.0 * np.pi)


# --------------------------------------------------------------------------- #
# The quality metric Gate P8.4 is scored on                                   #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The quality metric a change to the potential can actually be judged on      #
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=4)
def _decoded_surface_cached(rho_bytes: bytes, n: int, subdiv: int):
    from ..geom.mesh import icosphere
    from ..geom.operators import vertex_areas
    from .lift import lift_spectral

    rho_hat = np.frombuffer(rho_bytes, dtype=float, count=n)
    m = icosphere(subdiv)
    S = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    X, _ = lift_spectral(S, rho_hat, int(round(np.sqrt(n))) - 1)
    return {"S": S, "faces": m.faces,
            "dA_sphere": vertex_areas(S, m.faces),
            "dA_surface": vertex_areas(X, m.faces)}


def decoded_surface(rho_hat: np.ndarray, subdiv: int = 5) -> dict:
    """The cone-independent half of :func:`area_distortion`: the band-limited
    surface (:func:`shapegen.quad.lift.lift_spectral`) sampled at
    ``icosphere(subdiv)``'s vertices, with per-vertex area elements of the
    sphere and of the surface on that triangulation.

    It dominates :func:`area_distortion`'s cost and depends only on ``ρ̂``
    (``L`` is inferred from its length), so sweeps over cone positions should
    build it once and pass ``surface=``. The last four ``(ρ̂, subdiv)`` are
    LRU-cached.

    Returns ``{"S", "faces", "dA_sphere", "dA_surface"}``.
    """
    rho = np.ascontiguousarray(np.asarray(rho_hat, dtype=float))
    return _decoded_surface_cached(rho.tobytes(), rho.size, subdiv)


def area_distortion(cones, rho_hat: np.ndarray, L: int, subdiv: int = 5,
                    n_samples: int = 4000, seed: int = 0,
                    surface: dict | None = None,
                    pole: np.ndarray | None = None) -> dict:
    """How much a cone set stretches area, measured on the surface itself.

    A quad mesh lays down cells that are exactly ``s x s`` in the developed
    plane; this measures how unequal they become on the surface. It never
    forms a curvature potential, so it can referee changes to one. 0 is
    unreachable, since a conformal map
    cannot preserve area.

    * surface area element: :func:`decoded_surface` (``lift_spectral``);
    * flat area element: ``|W|^2`` from the cones alone
      (:func:`shapegen.quad.grid.form_log_abs`);
    * ``sigma`` = their ratio, integrated against the flat measure (every
      cell has equal flat area, so this is what a per-face average over the
      traced mesh converges to).

    Returns ``{"energy", "interdecile", "sigma", "weights"}``: ``energy`` is
    the flat-measure variance of ``log sigma`` (scale-free), ``interdecile``
    the p90/p10 ratio of ``sigma``. Raises ``ValueError`` unless the index
    sum is 2.

    Trap -- resolution on scanned faces. The surface is exact, but both area
    elements are triangle areas on ``icosphere(subdiv)``. With the former
    mesh decoder, ``subdiv=5`` was converged on ``ℓ⁻²`` fixtures (<3% at
    L=16, ~7% at L=32 vs subdiv 6) but not on FaceScape faces (20-35%, and
    cone-set comparisons changed sign). How much of that was the decoder and
    how much the sampling has not been re-measured; check convergence in
    ``subdiv`` before quoting a face number.

    ``surface`` takes a prebuilt :func:`decoded_surface`. ``pole`` freezes
    the chart (forwarded to :func:`~shapegen.quad.cones.project_cones`),
    skipping the 40 ms random pole search; the energy is chart-free (agrees
    to 2e-15 across poles), so this buys cost and determinism only.
    """
    from . import cones as _qc
    from . import grid as _qg

    idx = np.array([k for _p, k in cones], dtype=float)
    if abs(idx.sum() - 2.0) > 1e-6:
        raise ValueError(f"cone index sum is {idx.sum()}, not 2 -- the flat "
                         "structure is not regular at the chart's far pole "
                         "and |W| is meaningless (see quad/grid.py)")

    if surface is None:
        surface = decoded_surface(rho_hat, subdiv)
    S = surface["S"]
    dA_sphere = surface["dA_sphere"]
    dA_surface = surface["dA_surface"]

    chart = _qc.project_cones(cones, n_samples=n_samples, seed=seed,
                              pole=pole)
    Xr = S @ chart["Rm"].T
    c = _qg.chart_of(Xr)
    z = _qg.to_chart(Xr, c)
    lm = _qg.form_log_abs(z, c, chart["z"], -chart["idx"])
    # dA_sphere / dA_flat, the chart cancelling out of the ratio
    conf = (2.0 / (1.0 + np.abs(z) ** 2)) ** 2 * np.exp(-2.0 * lm)

    dA_flat = dA_sphere / conf
    sigma = dA_surface / dA_flat
    ls = np.log(sigma)
    w = dA_flat / dA_flat.sum()
    energy = float((w * (ls - (w * ls).sum()) ** 2).sum())
    return {"energy": energy,
            "interdecile": float(np.percentile(sigma, 90)
                                 / np.percentile(sigma, 10)),
            "sigma": sigma, "weights": w}

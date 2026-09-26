"""Mesh-free cone placement: a spin-4 cross field found by spectral
Ginzburg-Landau annealing (PLAN.md §P8.3/§P8.4, ported from
``03_spectral/solver.py`` and ``gl_spec.py``).

**Deliberately not included**: ``03_spectral/blend.py``'s smoothness/alignment
control ``t`` (§P8.6). That term biases the flow towards the Hopf differential
of ρ; this module only anneals ``ψ`` towards unit modulus, which is what
``gl_spec.py`` itself does before ``blend.py`` is layered on top.

The field ``ψ`` is spin-4 (a "cross" — 4-fold rotational symmetry — raised to
the 4th power, so it is single-valued). Its connection, for an immersion
conformal to the round sphere, is ``∇ = ∇₀ - i·s·A`` with ``A = i·ð(φ)``, ``φ``
the potential of the curvature-density perturbation ``μ - 1``:

    C ψ = -Δ₀ψ + s²|A|²ψ + 2is·(A·∇₀ψ),   A·∇₀ψ = ½(Ā·ðψ + A·ð̄ψ)

``div A = 0`` because ``A`` is a rotated gradient, so there is no extra term.

**The linearisation this inherits, unfixed.** ``φ`` solves ``Δ₀φ = μ - 1`` with
``μ = 1 + 2ρ`` — the bundle's own linearisation, not this repo's exact spin
transform. PLAN.md §2.1: "It does not fix the mesh-free path... replacing
*that* is §P9." So ``curvature_potential`` takes ``ρ̂`` (this repo's real-SH
convention) and applies ``μ = 1 + 2ρ`` directly, with no correction.
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
    "cones_from_field", "winding_total", "area_distortion_energy",
    "rho_rms", "band_taper", "DEFAULT_TAPER",
    "decoded_surface", "area_distortion",
]


# --------------------------------------------------------------------------- #
# The real/complex coefficient boundary (§2.3/§P8.1's "one conversion pair")   #
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

    and ``C[l, l] = c[0]`` for ``m = 0`` (already real, no phase). Verified
    against a direct ``real_sh_basis`` synthesis to 1.7e-14 — see
    ``test_quad_field.py::test_real_to_complex_matches_real_sh_basis``.
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


DEFAULT_TAPER = 1e-3
"""Heat-kernel time :func:`band_taper` uses by default on scanned input."""


def band_taper(rho_hat: np.ndarray, L: int, t: float = DEFAULT_TAPER):
    """Smooth ``rho_hat`` by a heat kernel, ``c_lm *= exp(-t l(l+1))``.

    **Why a band-limited code still needs this.** ``rho_hat`` is band-limited
    by construction, so it is tempting to assume the decode can resolve it.
    It cannot, on scanned input, because what matters is the *spectrum* inside
    the band, not the band. Measured at L=24, share of the coefficient norm in
    the top quartile of degrees:

        FaceScape head   0.665        l^-2 fixture   0.071

    A face's per-degree power is flat-to-rising (1.37 at l=2, 1.73 at l=24);
    the fixture's falls as ``l^-2`` (5.83 -> 0.18). The projection of a field
    with large content *above* ``L`` piles that energy into the top modes, so
    the decoded surface carries features exactly at the resolution limit and
    :func:`codec.decode` stops being mesh-convergent. That is what made
    :func:`area_distortion` unusable on faces -- its verdict changed sign
    between referee subdivisions.

    The taper restores it, monotonically (head 254, L=24, referee drift
    between subdivision 5 and 6)::

        t = 0     5e-4    1e-3    2e-3    4e-3
        31.0%   20.4%   13.7%    6.4%    0.0%      drift
        1.87    1.63    1.45    1.18    0.90       rms(rho)

    **Both columns matter, and they pull against each other.** Smoothing far
    enough to converge the referee (``t = 4e-3``) also takes the amplitude
    below ``rms`` 1.0, i.e. it smooths away the very regime
    that makes the exact potential worth solving for. :data:`DEFAULT_TAPER`
    is ``1e-3``: drift 13.7%, comfortably under the effect sizes being
    resolved (the potential difference is 15-29% on these heads), while
    leaving ``rms`` at 1.45-1.51.

    This is a *measurement* aid, not a pipeline stage -- it changes the shape
    being meshed. Use it to make a scanned code scoreable; do not silently
    apply it to something you are going to ship.
    """
    c = np.array(rho_hat, dtype=float, copy=True)
    i = 0
    for l in range(L + 1):
        n = 2 * l + 1
        c[i:i + n] *= np.exp(-t * l * (l + 1))
        i += n
    return c


def curvature_potential(rho_hat: np.ndarray, L: int, spinor=None) -> np.ndarray:
    """``ρ̂ -> φ``, the conformal potential the connection is a gradient of:
    ``-log|λ|²`` of the immersion's actual spinor (PLAN.md §P9), via
    :func:`shapegen.geom.dirac_spectral.log_conformal_coeffs`.

    Pass ``spinor`` (a :class:`~shapegen.geom.dirac_spectral.DiracSpectral`
    solved at ``spinor_band(L, potential_band_extra(L))``) to reuse one
    instead of solving again -- at L=32 the solve is the expensive step, and
    §P7 route 2 (:func:`shapegen.quad.lift.lift_route2`) needs the same one.
    See :func:`solve_potential_spinor`.

    The sign is this module's convention for ``A = i·ðφ``. It converges to the
    decoded surface's own conformal factor at ``O(h²)`` (0.0436, 0.0109,
    0.0027, 0.00068 at subdivision 3/4/5/6).

    **The bundle's linearisation ``μ = 1 + 2ρ`` was removed 2026-09-26**,
    along with the ``method="auto"`` switch between the two. Real FaceScape
    codes sit at ``rho_rms`` 1.87-1.93, where the linearisation is not an
    approximation of anything; the evidence that retired it is recorded in
    PLAN.md §P9 ("Linear potential removed").
    """
    if spinor is None:
        spinor = solve_potential_spinor(rho_hat, L)
    from ..geom.dirac_spectral import log_conformal_coeffs
    return -log_conformal_coeffs(spinor, L)


def solve_potential_spinor(rho_hat: np.ndarray, L: int):
    """The §P9 spinor :func:`curvature_potential` needs, at the padded band
    ``spinor_band(L, potential_band_extra(L))`` -- solve it once with this and
    hand it to both :func:`curvature_potential` and
    :func:`shapegen.quad.lift.lift_route2`."""
    from ..geom.dirac_spectral import (potential_band_extra, solve_spinor,
                                       spinor_band)
    return solve_spinor(rho_hat, L, Ls=spinor_band(L, potential_band_extra(L)))


def connection_field(phi: np.ndarray) -> np.ndarray:
    """``A = i·ð(φ)``: the spin-1 connection correction, a rotated gradient of
    the curvature potential."""
    return 1j * _sw.edth(phi, 0)


def apply_C(psi_c: np.ndarray, S: int, L: int, A: np.ndarray) -> np.ndarray:
    """``C ψ`` for coefficients ``psi_c`` at spin ``S`` (see module docstring).

    ``A`` is spin-1, synthesised at the same bandlimit ``L`` as everything
    else here (one shared grid per ``L``, unlike the bundle's per-spin
    ``Grid`` dict -- ``swsh``'s grid depends only on ``L``, not spin).

    This is the direct, one-column-at-a-time reference form -- kept for
    :func:`assemble_C` to be cross-checked against (see
    ``test_quad_field.py::test_assemble_C_matches_apply_C_reference``), not
    used by ``assemble_C`` itself any more (see its docstring for why).
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
    """Dense assembly of ``C`` at spin ``S``, as three matrix products
    instead of one ``apply_C`` call per column.

    ``apply_C`` calls ``synth``/``analy`` four times, and each of those reads
    the *entire* basis matrix for its ``(s, L)`` (``nmodes x Npix``, tens of
    MB at ``L=32``) from memory. Column-by-column assembly therefore streams
    that whole matrix once per column -- ``O(n)`` reads of an ``O(n)``-sized
    matrix -- which is exactly the redundant memory traffic a matrix
    *product* avoids by blocking. Concretely, writing ``c`` for the
    coefficient vector (``modes = mode_list(S, L)`` order) and ``Y_s`` for
    ``basis_matrix(s, L)``'s stacked basis (so ``synth(c, s) = Y_s.T @ c``
    and ``analy(f, s) = Y_s.conj() @ (weights * f)``), ``apply_C``'s three
    terms are each linear or bilinear in ``c`` and reduce to:

        M_B    = Y_S.conj() @ diag(w·|A|²) @ Y_S.T                (n_S x n_S)
        M_up   = Y_S.conj() @ diag(w·Ā) @ Y_{S+1}.T @ diag(ð-factors)
        M_down = Y_S.conj() @ diag(w·A) @ Y_{S-1,last n_S rows}.T @ diag(ð̄-factors)
        C      = diag(l(l+1)-S²) + S²·M_B + iS·(M_up + M_down)

    ``M_up``'s columns beyond ``mode_list(S+1, L)`` (the ``l=S`` block, which
    ``ð`` annihilates) are zero; ``M_down`` needs only ``Y_{S-1}``'s *last*
    ``n_S`` rows because ``mode_list(S-1, L) = [l=S-1 block] + mode_list(S, L)``
    exactly (``swsh.mode_list``'s ordering makes this a slice, not a search).

    Cross-checked against the direct ``apply_C``-per-column assembly at
    ``L=8`` to 1.2e-16 relative -- see
    ``test_quad_field.py::test_assemble_C_matches_apply_C_reference``.

    ``C`` is analytically self-adjoint (``div A = 0``), but the discrete
    ``synth``/``analy`` round trip through the nonlinear ``|A|²`` and
    ``A·∇`` terms aliases slightly above ``L``, leaving a relative asymmetry
    of a percent or so; symmetrised here (as the bundle's own scripts always
    do right after assembly) so every caller gets a matrix ``eigh`` can use
    safely.

    Returns ``(Cmat, modes)`` where ``modes = swsh.mode_list(S, L)`` gives the
    ``(l, m)`` each row/column corresponds to.
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

    Rescaled by ``1/√(peak modulus)``, exactly as the bundle's own
    ``gl()``/linear-stage code does after every stage. This is *not* a unit-
    peak normalisation (it moves the peak from ``m`` to ``√m``, not to 1) --
    it is the same soft rescaling :func:`ginzburg_landau` applies after every
    annealing level, kept here only so the two stages start from a
    comparable scale.
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
"""Geometric ``2.0 -> 1.0`` in units of ``π/L``, six levels — the measured
default (see :func:`ginzburg_landau`; it replaces ``(6.0, 3.0, 2.0)``, which
sat above :func:`epsilon_ceiling` at every ``L <= 16`` tested and therefore
returned the *linear start* rather than an annealed field)."""


def epsilon_ceiling(Cmat: np.ndarray) -> float:
    """The largest ``ε`` at which :func:`ginzburg_landau` can hold a field up
    at all: ``1/√(λ_min(C))``. **Above it the flow is not an anneal.**

    Linearise the semi-implicit step about ``ψ = 0``. The nonlinear term
    ``(1/ε²)(1 - |ψ|²)ψ`` becomes ``(1/ε²)ψ``, so with ``τ = c·ε²`` one step
    multiplies each eigenmode of ``C`` by

        (1 + τ/ε²) / (1 + τλ) = (1 + c) / (1 + c·ε²λ),

    which exceeds 1 exactly when ``ε²λ < 1`` — *independently of* ``c``, so
    this is a property of the flow, not of the step size. A mode can only
    grow out of zero if ``ε < 1/√λ``; if even the smallest eigenvalue fails
    that, **every** mode decays and ``ψ -> 0``.

    What that does in practice is worse than "the penalty does nothing"
    (PLAN.md §P8.4's wording). ``ψ`` decays geometrically, the per-level
    rescale ``ψ/√(peak|ψ|)`` cannot keep up, and what survives is the
    slowest-decaying eigendirection — i.e. :func:`linear_start`. Measured on
    the F1 fixture with the old ``(6.0, 3.0, 2.0)`` schedule, the returned
    field overlaps ``linear_start`` at ``|⟨ψ_GL, ψ_lin⟩| = 0.99999`` (L=12)
    and ``0.99964`` (L=16), and :func:`area_distortion` scores it at the
    linear start's own number to five digits. Pushing ``iters`` higher does
    not rescue it, it *underflows*: at ``iters=2560`` the field is exactly 0
    and :func:`cones_from_field` fails on ``NaN``.

    ``λ_min`` is nearly bandlimit-independent (the spin-4 Laplacian's
    ``l(l+1) - S²`` floor at ``l = S`` is 4, pulled down by the curvature
    term), so the *ceiling is essentially an absolute ``ε``* — measured 0.575
    at both L=12 and L=16 on F1 amp 1, 0.5 exactly on the round sphere, 0.653
    at amp 4 — while the schedule is written in units of ``π/L``. The two
    therefore cross: in ``π/L`` units the ceiling is ``1.91`` at L=12, ``2.55``
    at L=16, ``3.82`` at L=24 (round sphere). That is why the old schedule
    failed hardest at *small* L, and why :func:`anneal_schedule` clamps.
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

    The schedule is written in ``π/L`` — the resolution scaling that fixes
    where the *optimum* sits (measured 1-1.5 at both L=12 and L=16, PLAN.md
    §P8.4's own number) — while the ceiling is an absolute ``ε``. At small
    ``L`` the top of a fixed ``π/L`` schedule can poke through it:
    ``DEFAULT_LEVELS``' 2.0 is ``0.524`` at L=12, against a round-sphere
    ceiling of ``0.5``. Clamping keeps one schedule usable across bandlimits
    instead of making the caller re-derive it per ``L``.
    """
    cap = safety * epsilon_ceiling(Cmat) * L / np.pi
    return tuple(min(float(ef), cap) for ef in levels)


def ginzburg_landau(psi0: np.ndarray, Cmat: np.ndarray, modes, S: int, L: int,
                     levels=None, iters: int = 160,
                     check: bool = True) -> np.ndarray:
    """Semi-implicit Ginzburg-Landau flow at spin ``S``, one factorisation per
    ``ε`` level in ``levels``.

    ``ε`` scales to the **field's own resolution** ``π/L`` (``tune.py``'s
    fix), not to the finer grid :func:`cones_from_field` reads zeros off —
    PLAN.md's "the pitfall that cost the most": scaling to the output grid
    left the cones essentially unmoved (worse than a plain cube on the
    area-distortion energy; measured here at 0.115 against the cube's 0.078
    at L=12, a 1.47x loss, and 1.53x at L=16).

    ``levels`` defaults to :func:`anneal_schedule` — ``DEFAULT_LEVELS``
    clamped to :func:`epsilon_ceiling` — and is bounded on **both** sides:

    * **above** by :func:`epsilon_ceiling`, where the field collapses to the
      linear start (``check=True`` refuses such a schedule; see that
      function);
    * **below** by the grid: once the vortex core is narrower than the field's
      own resolution the solution degrades. Measured at one level, L=12 F1:
      ``ε/(π/L) = 1.2 -> 0.075``, ``0.8 -> 0.081``, ``0.5 -> 0.084``,
      ``0.4 -> 0.112``, ``0.3 -> 0.124``, ``0.2 -> 0.175``; L=16 the same,
      turning up below 0.5. PLAN.md §P8.4's "do not drive ε to 0", measured.

    **Why the default changed from ``(6.0, 3.0, 2.0)``.** That schedule is
    entirely above the ceiling at every bandlimit this was tested at, so it
    did not anneal — it returned :func:`linear_start` (overlap 0.99999) and
    failed §P8.4's own "GL beats the linear eigenproblem by >2x" gate at
    1.02x. Across 13 (L, amplitude) cases at L = 12/16/24 the old schedule
    scored a geometric-mean :func:`area_distortion` energy of **0.189** with
    two outright failures (0 cones on the round sphere) and a worst case of
    0.647; ``DEFAULT_LEVELS`` scores **0.0735** with no failures and a worst
    case of 0.081 — a 2.6x improvement with 8x less spread. Four other
    in-window schedules land within 6% of that, so the choice among them is
    inside the noise; what matters is staying under the ceiling.

    **It was a small-L failure, which is why it survived.** The ceiling in
    ``pi/L`` units grows with ``L`` (1.91 at L=12, 2.55 at L=16, 3.82 at
    L=24, 5.09 at L=32, round sphere), so ``(6.0, 3.0, 2.0)`` is entirely
    above it at L=12-16 -- where every test in this repo runs -- and mostly
    *below* it at L=32, where ``scripts/quad_demo.py`` runs. Measured at
    L=32: old 0.0881 / 0.0783 against this default's 0.0756 / 0.0855 at
    amplitude 0 / 1, i.e. a wash. The demo was never broken; the tests were.

    ``check=False`` runs an above-ceiling schedule anyway, which is only
    useful for reproducing the collapse (the tests do).
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
    matrix -- cached because a typical caller (:func:`cones_from_field`)
    evaluates the same ``(S, L, over)`` grid twice per sample (linear start,
    then the GL result), and ``eval_basis`` recomputes ``wigner_d`` from
    scratch (it takes arbitrary scattered points, so it cannot reuse
    ``swsh``'s own ``gl_grid``-keyed basis cache) -- one of the two calls
    was pure repeated work.
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

    Returns a list of ``(position (3,), index)``.

    **The sign, which used to be negated here.** Poincare-Hopf on a genus-0
    surface puts the total index at ``+chi = +2`` for a field of any spin --
    a cross field on a sphere is 8 cones of ``+1/4``, the cube -- and
    :func:`winding_total` measures the raw plaquette sum on these fields as
    ``+8 = +S.chi``, so ``k/S`` already has the right sign and the extra
    negation made every index come back at ``-1/4``, summing to ``-2``.

    That was not cosmetic downstream. The flat structure the whole quad
    pipeline develops is ``omega = prod_k (z - C_k)^(-idx_k) dz``, which is
    regular at the chart's far pole exactly when ``sum_k idx_k = +2``; at
    ``-2`` it acquires a pole of order 4 there instead, and no square grid
    closes up over it. Measured directly on the ``ell^-2`` fixture with
    :func:`shapegen.quad.grid.quad_grid`: the corrected sign gives Euler
    characteristic 2 and coverage 1.000000, the negated one gives -3240.
    The P0 reference corpus agrees -- its ``cone_idx.npy`` is ``+0.25``
    eight times over -- which is why nothing caught this until the mesh-free
    cones and the extraction were run end to end.
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
    grid, in units of ``2π`` -- a cheap diagnostic that the field is a
    genuine section. Poincaré-Hopf fixes it at ``S·χ = +2S`` for a spin-``S``
    field on a genus-0 surface, and it measures ``+8.000000`` at ``S=4`` on
    both the round sphere and F1; it is ``+S`` times
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

def area_distortion_energy(cones, rho_hat: np.ndarray, L: int, LS: int = 12,
                            power: int = 2, n_samples: int = 30000,
                            seed: int = 0) -> float:
    """``tune.py``'s ``energy``: how well a discrete set of quarter cones
    reproduces the curvature-induced target moments ``K_lm``, in an
    ``H⁻power`` norm (``power=2`` matches PLAN.md's gate).

    ``K_lm`` are the degree ``1..LS`` moments of ``μ - 1 = 2ρ`` on a Monte
    Carlo sample of the sphere; a set of point charges ``k_i`` at ``P_i``
    contributes ``2π Σ_i k_i · conj(Y_lm(P_i))`` (the ``2π`` matches the
    charges' own normalisation in the bundle). Lower is better.
    """
    from scipy.special import sph_harm_y

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    th = np.arccos(np.clip(X[:, 2], -1.0, 1.0))
    ph = np.arctan2(X[:, 1], X[:, 0])

    rho_c = real_to_complex(np.asarray(rho_hat, dtype=float), L)
    modes0 = _sw.mode_list(0, L)
    mu_grid = np.zeros(n_samples)
    for i, (l, m) in enumerate(modes0):
        if 1 <= l <= LS:
            mu_grid += np.real(2.0 * rho_c[l, l + m] * sph_harm_y(l, m, th, ph))

    llw = np.array([l * (l + 1) for l in range(1, LS + 1) for _m in range(-l, l + 1)],
                    dtype=float)
    Klm = np.array([4.0 * np.pi * np.mean(mu_grid * np.conj(sph_harm_y(l, m, th, ph)))
                    for l in range(1, LS + 1) for m in range(-l, l + 1)])

    if len(cones) == 0:
        d = -Klm
    else:
        P = np.array([p for p, _k in cones])
        k = np.array([kk for _p, kk in cones])
        tc = np.arccos(np.clip(P[:, 2], -1.0, 1.0))
        pc = np.arctan2(P[:, 1], P[:, 0])
        Y = np.array([sph_harm_y(l, m, tc, pc)
                      for l in range(1, LS + 1) for m in range(-l, l + 1)])
        d = 2.0 * np.pi * (Y.conj() @ k) - Klm
    return float(np.sum(np.abs(d) ** 2 / llw ** power))


# --------------------------------------------------------------------------- #
# The quality metric a change to the potential can actually be judged on      #
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=4)
def _decoded_surface_cached(rho_bytes: bytes, n: int, subdiv: int):
    from ..geom.mesh import icosphere
    from ..geom.operators import vertex_areas
    from .. import codec

    rho_hat = np.frombuffer(rho_bytes, dtype=float, count=n)
    m = icosphere(subdiv)
    S = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    X = codec.decode(rho_hat, S, m.faces)
    return {"S": S, "faces": m.faces,
            "dA_sphere": vertex_areas(S, m.faces),
            "dA_surface": vertex_areas(X, m.faces)}


def decoded_surface(rho_hat: np.ndarray, subdiv: int = 5) -> dict:
    """The **cone-independent** half of :func:`area_distortion`: decode ``ρ̂``
    onto ``icosphere(subdiv)`` and take both area elements.

    :func:`area_distortion` used to redo this on every call, and it is 97% of
    that call's cost (2.63 s of 2.72 s at ``L=12, subdiv=5``) — while
    depending on nothing but ``ρ̂``. Any sweep or optimisation over *cone
    positions* at fixed ``ρ̂`` should build this once and pass it as
    ``surface=``; an LRU of the last four ``(ρ̂, subdiv)`` makes plain repeated
    calls fast too.

    Returns ``{"S", "faces", "dA_sphere", "dA_surface"}``.
    """
    rho = np.ascontiguousarray(np.asarray(rho_hat, dtype=float))
    return _decoded_surface_cached(rho.tobytes(), rho.size, subdiv)


def area_distortion(cones, rho_hat: np.ndarray, L: int, subdiv: int = 5,
                    n_samples: int = 4000, seed: int = 0,
                    surface: dict | None = None,
                    pole: np.ndarray | None = None) -> dict:
    """How much a cone set stretches area, measured on the surface itself.

    **Why this exists next to :func:`area_distortion_energy`.** That one scores
    a placement against ``K_lm``, the spherical-harmonic moments of
    ``mu - 1 = 2 rho`` -- which *is* the linearisation
    :func:`curvature_potential`'s Dirac solve replaced (the linearisation
    itself was removed 2026-09-26). A referee that
    shares the assumption under test cannot settle the question, and measured
    it did not: the exact potential scored 1.17x, 0.48x, 1.37x, 1.10x and
    3.09x the linearisation's number across five amplitudes, with no trend.

    This one never forms ``mu``. It asks the question cones exist to answer:
    a quad mesh lays down cells that are *exactly* ``s x s`` in the developed
    plane, so how unequal are they once they are on the surface? Lower is
    better, and 0 would mean the flat structure the cones define is
    isometric to the surface -- unreachable, since a conformal map cannot
    preserve area (PLAN.md §10), which is exactly why where the cones go is a
    real choice.

    Three quantities, none of which come from a curvature potential:

    * the **surface** area element, from :func:`shapegen.codec.decode` -- the
      *mesh* Dirac solve, independent of both candidates;
    * the **flat** area element, ``|W|^2`` from the cones alone
      (:func:`shapegen.quad.grid.form_log_abs`);
    * their ratio ``sigma``, integrated against the flat measure -- which is
      the right measure here precisely because every cell has the same flat
      area, so this is what a per-face average over the traced mesh converges
      to.

    Returns ``{"energy", "interdecile", "sigma", "weights"}`` with ``energy``
    the flat-measure variance of ``log sigma`` (scale-free, so the immersion's
    undetermined global scale drops out) and ``interdecile`` the p90/p10
    ratio of ``sigma`` for a reading in plain units.

    **Resolution warning: ``subdiv=5`` does not referee a real scanned face.**
    The surface half of this metric is ``codec.decode`` on ``icosphere(subdiv)``,
    and that decode is only as converged as the mesh. Measured, the energy of
    one fixed cone set against ``subdiv`` 4 / 5 / 6:

    ====================  =======  =======  =======  =========  =========
    field                 sub 4    sub 5    sub 6    |5-6|/6    |4-6|/6
    ====================  =======  =======  =======  =========  =========
    ``ℓ⁻²`` L=16 amp 1    0.07237  0.07130  0.07196  **0.9%**   0.6%
    ``ℓ⁻²`` L=16 amp 8    0.21563  0.22854  0.23543  **2.9%**   8.4%
    ``ℓ⁻²`` L=32 amp 1    0.08395  0.08549  0.07991  **7.0%**   5.1%
    ``ℓ⁻²`` L=32 amp 8    0.23074  0.25638  0.25938  **1.2%**   11.0%
    FaceScape L=24        0.05603  0.06656  0.08499  **21.7%**  34.1%
    FaceScape L=24 (2nd)  0.05426  0.06870  0.08999  **23.7%**  39.7%
    FaceScape L=32        0.15787  0.05688  0.08093  **29.7%**  95.1%
    FaceScape L=32 (2nd)  0.16633  0.05468  0.08420  **35.1%**  97.5%
    ====================  =======  =======  =======  =========  =========

    On a smooth band-limited field the default is fine (<3% at L=16, ~7% at
    L=32). On a real face it is **not**: the value moves by a fifth to a
    third between ``icosphere(5)`` and ``icosphere(6)``, and it is not even
    monotone in the mesh (0.158 -> 0.057 -> 0.081 at L=32).

    **And the bias is not common mode, so ratios do not survive it either.**
    The obvious hope is that two cone sets scored on the *same* surface
    cancel the referee's error. Measured, on a smooth field they do and on a
    face they do not. Ratio of the Dirac potential to the (since removed) linearisation
    (geometric mean over four schedules, after refinement), the *same* cone
    sets scored against ``subdiv=5`` and ``subdiv=6``:

    ====================  ==========  ==========
    field                 referee 5   referee 6
    ====================  ==========  ==========
    ``ℓ⁻²`` L=16 amp 8    0.432x      0.453x
    ``ℓ⁻²`` L=32 amp 1    0.983x      0.984x
    FaceScape L=24        0.983x      **0.831x**
    FaceScape L=24 (2nd)  1.016x      **0.711x**
    FaceScape L=32        2.247x      **0.946x**
    FaceScape L=32 (2nd)  2.271x      **0.737x**
    ====================  ==========  ==========

    On the fixture the ratio is stable to 5%. On faces it **changes sign** at
    L=24 and swings by 2.4x at L=32 -- "dirac loses 2.3x" at ``subdiv=5``
    becomes "dirac wins 1.4x" at ``subdiv=6``, on identical cone sets. So at
    the default this metric cannot referee a scanned face *at all*: not its
    absolute value, and not the comparisons it is used for. Raise ``subdiv``
    until the answer stops moving, or measure on the traced mesh, before
    quoting any face number. Nothing here weakens it on the band-limited
    fixtures the tests use, where it is converged and was validated against
    both the traced mesh and the cube's known optimum.

    ``surface`` takes a prebuilt :func:`decoded_surface` so a sweep over cone
    placements at fixed ``ρ̂`` does not pay for a Dirac solve per evaluation.

    ``pole`` **freezes the chart** (forwarded to
    :func:`~shapegen.quad.cones.project_cones`), skipping the 4000-sample
    random pole search — 40 ms against the 1.1 ms the rest of a hoisted
    evaluation costs. The energy is chart-free and measured it really is:
    against a searched pole, and against a second arbitrary fixed pole, the
    number agrees to 2e-15 relative even where the search's own answer jumps
    by 179 degrees. So this is for cost and determinism, not correctness.
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

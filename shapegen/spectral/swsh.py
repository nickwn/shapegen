"""Spin-weighted spherical-harmonic (SWSH) transforms and the ð ladder.

Self-contained NumPy/SciPy implementation (float64, zero heavy deps). The basis
is built from the Wigner small-d function, by a **stable three-term recurrence
in ℓ** (:func:`wigner_d_ladder`, PLAN.md §P8.1); the convention is verified in
the tests against the ``spherical`` package (independent oracle) at every spin
and scipy ``sph_harm_y`` at spin 0, namely

    ₍s₎Y_{ℓm}(θ,φ) = √((2ℓ+1)/4π) d^ℓ_{m,−s}(θ) e^{imφ}   (Condon-Shortley).

Coefficients live in a dense complex array ``C[ℓ, ℓ+m]`` of shape
``(L+1, 2L+1)`` (column index ``ℓ+m`` ∈ [0, 2ℓ]); entries with ℓ < |s| or
|m| > ℓ are zero.

Transforms use a Gauss-Legendre (θ) x equiangular (φ) grid with ``Nθ = L+1`` and
``Nφ = 2L+1``, which makes ``analy∘synth = id`` exactly for band-limited spin
fields (the relevant products are degree-≤2L polynomials in cosθ).

**Accuracy (PLAN.md §P8.1's gate: 1e-12 round trip at band 32).** Met with three
decades to spare, at every spin tried::

    L=16  1.0e-14    L=24  4.0e-14    L=32  4.3e-14    L=48  1.3e-13

This module used to build ``d`` from Wigner's sum formula, which alternates in
sign over O(ℓ) terms with a ``(2ℓ)!`` prefactor: that was ~1e-12 at L=16 but
degraded to **5.3e-8 at L=32** -- the band this project targets -- and overflowed
outright at L=64. PLAN.md flagged it as a decision to take before anything else
in §P8/§P9 and offered a ``ducc0``/``shtns`` dependency as the alternative; the
recurrence keeps the module dependency-free and is flat in ℓ instead.

**Half-integer degrees (§P8.2)** fall out of the same recurrence with no special
casing -- see :func:`wigner_d_ladder` and the ``half_*`` functions below, which
spin-±1/2 fields (and so :mod:`shapegen.geom.dirac_spectral`) need.
"""

from functools import lru_cache
from math import sqrt

import numpy as np
from scipy.special import gammaln

from .grids import gl_grid, solid_angle_weights

__all__ = [
    "wigner_d", "wigner_d_ladder", "swsh", "basis_matrix", "empty_coeffs", "synth", "analy",
    "edth", "edthbar", "edth_factor", "edthbar_factor",
    "spin_laplacian", "laplacian_eig", "hopf_scaling", "q_from_sigma",
    "mode_list", "eval_basis", "fit_coeffs",
    "euler_zyz", "wigner_D", "rotate_coeffs",
    "half_modes", "half_grid", "half_basis", "half_synth", "half_analy",
    "half_edth_factor", "half_edthbar_factor", "half_eval_basis",
]


# ---------------------------------------------------------------------------
# Wigner small-d
# ---------------------------------------------------------------------------

def _d_seed(l0: float, a: float, b: float, theta: np.ndarray) -> np.ndarray:
    """``d^{l0}_{a,b}`` at ``l0 = max(|a|,|b|)``, where Wigner's sum collapses
    to a **single** term -- no alternating signs, so nothing cancels -- and
    the factorial prefactor goes through ``gammaln`` so nothing overflows
    either. This is the only place a closed form is needed; every higher
    degree comes from the recurrence."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    k = max(0.0, b - a)
    lg = 0.5 * (gammaln(l0 + a + 1) + gammaln(l0 - a + 1)
                + gammaln(l0 + b + 1) + gammaln(l0 - b + 1))
    lg -= (gammaln(l0 + b - k + 1) + gammaln(k + 1)
           + gammaln(a - b + k + 1) + gammaln(l0 - a - k + 1))
    return ((-1.0) ** (a - b + k)) * np.exp(lg) * c ** (2 * l0 + b - a - 2 * k) \
        * s ** (a - b + 2 * k)


def wigner_d_ladder(lmax: float, a: float, b: float, theta: np.ndarray):
    """``(degrees, d)`` with ``d[i] = d^{degrees[i]}_{a,b}(θ)`` for every degree
    from ``max(|a|,|b|)`` to ``lmax`` in unit steps -- by a stable three-term
    recurrence in ℓ (PLAN.md §P8.1).

    **Why this replaced Wigner's sum formula.** That sum alternates in sign
    over ``O(ℓ)`` terms whose prefactor grows like ``(2ℓ)!``, so it loses
    precision catastrophically with degree. Measured on ``Σ_b (d^ℓ_{a,b})² = 1``,
    which any correct implementation satisfies exactly:

    | ℓ  | sum formula | this recurrence |
    |----|-------------|-----------------|
    | 8  | 6.2e-15     | 2.0e-15         |
    | 16 | 4.5e-13     | 9.1e-15         |
    | 24 | 6.1e-11     | 1.8e-14         |
    | 32 | 3.7e-08     | 1.9e-14         |
    | 48 | 1.1e-03     | 2.1e-14         |
    | 64 | OverflowError | 4.4e-14       |

    PLAN.md §P8.1 called this decision out as one to take early ("upstream of
    everything in P8 and P9"), with the alternative being a dependency on
    ``ducc0``/``shtns``. The recurrence keeps the module dependency-free and
    is flat in ℓ, so it wins on both counts.

    The recurrence is

        d^{ℓ+1} = [ (2ℓ+1)(ℓ(ℓ+1)cosθ − ab)·d^ℓ
                    − (ℓ+1)√((ℓ²−a²)(ℓ²−b²))·d^{ℓ−1} ]
                  / [ ℓ·√(((ℓ+1)²−a²)((ℓ+1)²−b²)) ]

    which at ``a = b = 0`` reduces to Legendre's -- the cheapest check that it
    is transcribed correctly.

    **``a`` and ``b`` may be half-integers** (PLAN.md §P8.2), with no special
    casing: ``a−b`` stays an integer, so every factorial argument above stays
    a non-negative integer and the recurrence coefficients do not care. That
    is the whole of the half-integer implementation -- ``halfspin.py``'s
    ``fac(x) = Γ(x+1)`` trick applied to a formula that is stable to begin
    with.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    l0 = max(abs(a), abs(b))
    ls = np.arange(l0, lmax + 1e-9, 1.0)
    out = np.zeros((len(ls), theta.size))
    if len(ls) == 0:
        return ls, out
    x = np.cos(theta)
    if l0 == 0:
        # a = b = 0 is the one case the recurrence cannot start itself (its
        # denominator carries a factor of ℓ); seed two Legendre terms instead.
        out[0] = 1.0
        if len(ls) > 1:
            out[1] = x
        start = 1
    else:
        out[0] = _d_seed(l0, a, b, theta)
        start = 0
    for i in range(start, len(ls) - 1):
        l = ls[i]
        prev = out[i - 1] if i > 0 else 0.0
        out[i + 1] = ((2 * l + 1) * (l * (l + 1) * x - a * b) * out[i]
                      - (l + 1) * np.sqrt((l * l - a * a) * (l * l - b * b)) * prev
                      ) / (l * np.sqrt(((l + 1) ** 2 - a * a)
                                       * ((l + 1) ** 2 - b * b)))
    return ls, out


def wigner_d(l, a, b, theta: np.ndarray) -> np.ndarray:
    """Wigner small-d ``d^l_{a,b}(θ)``, by :func:`wigner_d_ladder`.

    ``theta`` may be an array; the result has the same shape. ``l``, ``a`` and
    ``b`` may be integers or all half-integers. Computing one degree at a time
    throws away the ladder, so callers that need a whole range of ℓ (the basis
    builders below) should call :func:`wigner_d_ladder` directly.
    """
    theta_a = np.asarray(theta, dtype=float)
    if abs(a) > l or abs(b) > l:
        return np.zeros_like(theta_a)
    _ls, d = wigner_d_ladder(l, a, b, theta_a.ravel())
    return d[-1].reshape(theta_a.shape)


# ---------------------------------------------------------------------------
# SWSH basis on the canonical GL grid (cached per (s, L))
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _basis(s: int, L: int):
    """Dict ``(ℓ,m) -> ₍s₎Y_{ℓm}`` sampled on ``gl_grid(L)`` (complex, (Nθ,Nφ))."""
    grid = gl_grid(L)
    th, ph = grid.theta, grid.phi
    Y = {}
    # loop over m and walk the whole ℓ ladder in one recurrence, rather than
    # one independent sum per (ℓ, m) -- both stabler (§P8.1) and cheaper
    for m in range(-L, L + 1):
        ls, d = wigner_d_ladder(L, m, -s, th)
        eim = np.exp(1j * m * ph)
        for i, l in enumerate(ls):
            l = int(round(l))
            # Condon-Shortley phase sits inside d^ℓ_{m,-s}; with this
            # normalization spin 0 matches scipy.sph_harm_y exactly.
            Y[(l, m)] = sqrt((2 * l + 1) / (4.0 * np.pi)) * d[i][:, None] * eim[None, :]
    return Y


def swsh(s: int, l: int, m: int, L: int) -> np.ndarray:
    """Single spin-weighted harmonic ``₍s₎Y_{ℓm}`` on ``gl_grid(L)``."""
    return _basis(s, L)[(l, m)]


@lru_cache(maxsize=None)
def basis_matrix(s: int, L: int):
    """Stacked form of :func:`_basis`: ``(Yflat, rows, cols)`` with
    ``Yflat`` shape ``(nmodes, Nθ·Nφ)`` (row ``i`` is ``₍s₎Y_{ℓm}.ravel()`` for
    ``mode_list(s, L)[i] = (l, m)``) and ``rows[i], cols[i] = l, l+m`` -- the
    ``C[l, l+m]`` index each row corresponds to.

    This is what makes :func:`synth`/:func:`analy` a single matrix-vector
    product instead of a Python loop over every ``(l, m)`` -- pulled out once
    per ``(s, L)`` and cached, same as ``_basis`` itself. Also exposed
    directly (not just via ``synth``/``analy``) for callers building their
    own operators out of several spins at once, e.g.
    ``shapegen.quad.field.assemble_C``, where going through ``synth``/
    ``analy`` one column at a time would mean re-reading this same matrix
    from memory once per column instead of once per assembly.
    """
    Y = _basis(s, L)
    modes = mode_list(s, L)
    rows = np.array([l for l, _m in modes], dtype=np.intp)
    cols = np.array([l + m for l, m in modes], dtype=np.intp)
    Yflat = np.array([Y[(l, m)].ravel() for (l, m) in modes])
    return Yflat, rows, cols


# ---------------------------------------------------------------------------
# Forward / inverse transforms
# ---------------------------------------------------------------------------

def empty_coeffs(L: int) -> np.ndarray:
    """Zeroed coefficient array of shape ``(L+1, 2L+1)`` (layout ``C[ℓ, ℓ+m]``)."""
    return np.zeros((L + 1, 2 * L + 1), dtype=complex)


def synth(C: np.ndarray, s: int, L: int | None = None) -> np.ndarray:
    """Inverse transform: coefficients ``C`` -> spin-``s`` field on ``gl_grid(L)``."""
    if L is None:
        L = C.shape[0] - 1
    grid = gl_grid(L)
    Yflat, rows, cols = basis_matrix(s, L)
    field = C[rows, cols] @ Yflat            # (nmodes,) @ (nmodes, Npix) -> (Npix,)
    return field.reshape(grid.ntheta, grid.nphi)


def analy(field: np.ndarray, s: int, L: int) -> np.ndarray:
    """Forward transform: spin-``s`` field on ``gl_grid(L)`` -> coefficients ``C``."""
    grid = gl_grid(L)
    Yflat, rows, cols = basis_matrix(s, L)
    fw = (field * solid_angle_weights(grid)).ravel()
    C = empty_coeffs(L)
    C[rows, cols] = Yflat.conj() @ fw         # (nmodes, Npix) @ (Npix,) -> (nmodes,)
    return C


# ---------------------------------------------------------------------------
# Ladder operators, Laplacian, and the Hopf scaling (pure coefficient ops)
# ---------------------------------------------------------------------------

def edth(C: np.ndarray, s: int) -> np.ndarray:
    """Raising operator ð: spin ``s`` -> ``s+1``. Multiplies degree ℓ by
    ``+√((ℓ−s)(ℓ+s+1))``."""
    out = np.zeros_like(C)
    for l in range(C.shape[0]):
        out[l] = sqrt(max((l - s) * (l + s + 1), 0)) * C[l]
    return out


def edthbar(C: np.ndarray, s: int) -> np.ndarray:
    """Lowering operator ð̄: spin ``s`` -> ``s−1``. Multiplies degree ℓ by
    ``−√((ℓ+s)(ℓ−s+1))``."""
    out = np.zeros_like(C)
    for l in range(C.shape[0]):
        out[l] = -sqrt(max((l + s) * (l - s + 1), 0)) * C[l]
    return out


def edth_factor(L: int, s: int) -> np.ndarray:
    """Per-degree ð factors ``√((ℓ−s)(ℓ+s+1))`` for ℓ ∈ [0, L]."""
    l = np.arange(L + 1)
    return np.sqrt(np.clip((l - s) * (l + s + 1), 0, None))


def edthbar_factor(L: int, s: int) -> np.ndarray:
    """Per-degree ð̄ factors ``−√((ℓ+s)(ℓ−s+1))`` for ℓ ∈ [0, L]."""
    l = np.arange(L + 1)
    return -np.sqrt(np.clip((l + s) * (l - s + 1), 0, None))


def spin_laplacian(C: np.ndarray, s: int) -> np.ndarray:
    """Spin-``s`` Laplacian ``Δ_s = ½(ðð̄ + ð̄ð)`` applied to ``C``."""
    a = edthbar(edth(C, s), s + 1)   # ð then ð̄: spin s -> s+1 -> s
    b = edth(edthbar(C, s), s - 1)   # ð̄ then ð: spin s -> s-1 -> s
    return 0.5 * (a + b)


def laplacian_eig(L: int, s: int) -> np.ndarray:
    """Spin-``s`` Laplacian eigenvalues ``−[ℓ(ℓ+1)−s²]`` for ℓ ∈ [0, L]."""
    l = np.arange(L + 1)
    return -(l * (l + 1) - s ** 2)


def hopf_scaling(L: int) -> np.ndarray:
    """Per-degree Hopf scaling ``√((ℓ−1)ℓ(ℓ+1)(ℓ+2))`` (q̂ = scaling · σ̂)."""
    l = np.arange(L + 1)
    return np.sqrt(np.clip((l - 1) * l * (l + 1) * (l + 2), 0, None))


def q_from_sigma(sigma_hat: np.ndarray) -> np.ndarray:
    """Hopf differential ``q = ð²σ`` (spin-0 σ̂ -> spin-2 q̂)."""
    return edth(edth(sigma_hat, 0), 1)


# ---------------------------------------------------------------------------
# Scattered-point evaluation and least-squares fit (off the GL grid)
# ---------------------------------------------------------------------------

def mode_list(s: int, L: int):
    """Ordered ``[(ℓ, m), ...]`` for ℓ ∈ [|s|, L], m ∈ [−ℓ, ℓ]."""
    return [(l, m) for l in range(abs(s), L + 1) for m in range(-l, l + 1)]


def eval_basis(s: int, L: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Spin-``s`` SWSH basis sampled at arbitrary ``(θ, φ)``.

    Returns a complex design matrix of shape ``(npts, nmodes)`` whose columns
    follow ``mode_list(s, L)``:
    ``₍s₎Y_{ℓm}(θ,φ) = √((2ℓ+1)/4π) d^ℓ_{m,−s}(θ) e^{imφ}``.
    """
    theta = np.asarray(theta, float).ravel()
    phi = np.asarray(phi, float).ravel()
    modes = mode_list(s, L)
    col_of = {lm: i for i, lm in enumerate(modes)}
    Y = np.empty((theta.size, len(modes)), dtype=complex)
    for m in range(-L, L + 1):
        ls, d = wigner_d_ladder(L, m, -s, theta)
        eim = np.exp(1j * m * phi)
        for i, l in enumerate(ls):
            l = int(round(l))
            Y[:, col_of[(l, m)]] = (sqrt((2 * l + 1) / (4.0 * np.pi))
                                    * d[i] * eim)
    return Y


def fit_coeffs(s: int, L: int, theta, phi, values, weights=None,
               rcond=1e-8) -> np.ndarray:
    """Least-squares spin-``s`` coefficients from scattered samples.

    Minimizes ``Σ_i w_i |Σ_{ℓm} C_{ℓm} ₍s₎Y_{ℓm}(θ_i,φ_i) − values_i|²`` and
    returns ``C`` in the dense ``C[ℓ, ℓ+m]`` layout. Use for fields known only
    on an irregular spherical mesh (e.g. the conformal Hopf field).
    """
    theta = np.asarray(theta, float).ravel()
    phi = np.asarray(phi, float).ravel()
    v = np.asarray(values, complex).ravel()
    Y = eval_basis(s, L, theta, phi)
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, float).ravel())
        Y = Y * sw[:, None]
        v = v * sw
    sol, *_ = np.linalg.lstsq(Y, v, rcond=rcond)
    C = empty_coeffs(L)
    for col, (l, m) in enumerate(mode_list(s, L)):
        C[l, l + m] = sol[col]
    return C


# ---------------------------------------------------------------------------
# Wigner-D rotation of coefficients (rotation augmentation in σ-space)
# ---------------------------------------------------------------------------

def euler_zyz(R: np.ndarray):
    """ZYZ Euler angles ``(α, β, γ)`` of an active rotation
    ``R = R_z(α) R_y(β) R_z(γ)``."""
    R = np.asarray(R, float)
    beta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
    if np.sin(beta) > 1e-9:
        alpha = np.arctan2(R[1, 2], R[0, 2])
        gamma = np.arctan2(R[2, 1], -R[2, 0])
    else:                                   # gimbal: fold γ into α
        alpha = np.arctan2(R[1, 0], R[0, 0])
        gamma = 0.0
    return alpha, beta, gamma


def wigner_D(l: int, R: np.ndarray) -> np.ndarray:
    """Wigner-D matrix ``D^ℓ_{m'm}(R) = e^{-im'α} d^ℓ_{m'm}(β) e^{-imγ}`` (active
    convention), shape ``(2ℓ+1, 2ℓ+1)`` indexed by ``m', m ∈ [-ℓ, ℓ]``.

    Self-contained (reuses :func:`wigner_d`); verified in tests against the
    grid-rotation ground truth and the ``spherical`` oracle.
    """
    a, b, g = euler_zyz(R)
    bt = np.array([b])
    D = np.empty((2 * l + 1, 2 * l + 1), dtype=complex)
    for mp in range(-l, l + 1):
        for mm in range(-l, l + 1):
            D[mp + l, mm + l] = (np.exp(-1j * mp * a)
                                 * wigner_d(l, mp, mm, bt)[0]
                                 * np.exp(-1j * mm * g))
    return D


def rotate_coeffs(C: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate coefficients by ``R`` (per degree ``a'_{ℓm'} = Σ_m D^ℓ_{m'm} a_{ℓm}``)
    so the implied field becomes ``f(R^{-1}·)``. Works for any spin (the D^ℓ are
    spin-independent)."""
    L = C.shape[0] - 1
    out = empty_coeffs(L)
    for l in range(L + 1):
        if l == 0:
            out[0, 0] = C[0, 0]
            continue
        out[l, 0:2 * l + 1] = wigner_D(l, R) @ C[l, 0:2 * l + 1]
    return out


# ---------------------------------------------------------------------------
# Half-integer degrees and spins (PLAN.md §P8.2)
# ---------------------------------------------------------------------------
#
# A spin-±1/2 field is what the quaternionic Dirac solve (§P9,
# :mod:`shapegen.geom.dirac_spectral`) is written in, and it needs half-integer
# ℓ and m. The recurrence above already handles those, so all that is missing
# is a container and a quadrature.
#
# These deliberately do NOT reuse the integer path's dense ``C[ℓ, ℓ+m]``
# array. Half-integer ℓ would need a second indexing convention inside every
# one of ``synth``/``analy``/``edth``/``edthbar``, and those are pinned by the
# oracle tests against the ``spherical`` package. A flat mode-list vector is
# both simpler and exactly what a dense eigensolver wants anyway.
#
# The fields themselves are antiperiodic in φ (period 4π, not 2π), so they are
# not single-valued on the sphere. The transform is unaffected: every
# *difference* m − m' is an integer, so discrete orthogonality in φ still
# holds on [0, 2π).

def _is_half(x) -> bool:
    return abs((float(x) * 2.0) % 2.0 - 1.0) < 1e-12


def half_modes(s: float, L: float):
    """Ordered ``[(ℓ, m), ...]`` for half-integer spin ``s`` and band ``L``:
    ℓ from ``|s|`` to ``L`` in unit steps, m from −ℓ to ℓ in unit steps."""
    if not (_is_half(s) and _is_half(L)):
        raise ValueError("half_modes needs half-integer s and L")
    out = []
    l = abs(s)
    while l <= L + 1e-9:
        m = -l
        while m <= l + 1e-9:
            out.append((l, m))
            m += 1.0
        l += 1.0
    return out


@lru_cache(maxsize=None)
def half_grid(L: float):
    """``(theta, phi, weights)`` for half-integer band ``L``.

    ``Nθ = ⌊L⌋ + 2`` Gauss-Legendre nodes and ``Nφ = ⌊2L⌋ + 2`` azimuthal
    samples. The θ count is what exactness needs: ``d^ℓ_{m,−s} d^{ℓ'}_{m,−s}``
    is a polynomial of degree ``ℓ+ℓ' ≤ 2L`` in cos θ (the half-integer parts
    cancel because ``|m±s|`` are integers), and GL with ``N`` nodes is exact
    through degree ``2N−1``. The φ count only has to exceed ``2L`` so that the
    integer differences ``m − m'`` never alias.
    """
    nth = int(np.floor(L)) + 2
    nph = int(np.floor(2.0 * L)) + 2
    x, w = np.polynomial.legendre.leggauss(nth)
    theta = np.arccos(x)
    phi = 2.0 * np.pi * np.arange(nph) / nph
    return theta, phi, w[:, None] * (2.0 * np.pi / nph)


@lru_cache(maxsize=None)
def half_basis(s: float, L: float):
    """``(nmodes, Nθ·Nφ)`` complex basis matrix for half-integer spin ``s``,
    rows following :func:`half_modes` -- the same
    ``√((2ℓ+1)/4π) d^ℓ_{m,−s}(θ) e^{imφ}`` convention as the integer path."""
    theta, phi, _w = half_grid(L)
    modes = half_modes(s, L)
    row_of = {lm: i for i, lm in enumerate(modes)}
    Y = np.empty((len(modes), theta.size * phi.size), dtype=complex)
    m = -L
    while m <= L + 1e-9:
        ls, d = wigner_d_ladder(L, m, -s, theta)
        eim = np.exp(1j * m * phi)
        for i, l in enumerate(ls):
            Y[row_of[(l, m)]] = (sqrt((2 * l + 1) / (4.0 * np.pi))
                                 * d[i][:, None] * eim[None, :]).ravel()
        m += 1.0
    return Y


def half_synth(vec: np.ndarray, s: float, L: float) -> np.ndarray:
    """Coefficients (flat, :func:`half_modes` order) -> field on
    :func:`half_grid`."""
    theta, phi, _w = half_grid(L)
    return (np.asarray(vec) @ half_basis(s, L)).reshape(theta.size, phi.size)


def half_analy(field: np.ndarray, s: float, L: float) -> np.ndarray:
    """Field on :func:`half_grid` -> coefficients (flat,
    :func:`half_modes` order)."""
    _theta, _phi, w = half_grid(L)
    return half_basis(s, L).conj() @ (np.asarray(field) * w).ravel()


def half_edth_factor(s: float, L: float) -> np.ndarray:
    """Per-mode ð factors ``+√((ℓ−s)(ℓ+s+1))``, spin ``s`` -> ``s+1``.

    At ``s = −1/2`` this is exactly ``ℓ + 1/2``, and at ``s = +1/2``
    :func:`half_edthbar_factor` is ``−(ℓ + 1/2)`` -- the two magnitudes the
    Dirac system in §P9 is built from."""
    return np.array([sqrt(max((l - s) * (l + s + 1.0), 0.0))
                     for l, _m in half_modes(s, L)])


def half_edthbar_factor(s: float, L: float) -> np.ndarray:
    """Per-mode ð̄ factors ``−√((ℓ+s)(ℓ−s+1))``, spin ``s`` -> ``s−1``."""
    return np.array([-sqrt(max((l + s) * (l - s + 1.0), 0.0))
                     for l, _m in half_modes(s, L)])


def half_eval_basis(s: float, L: float, theta: np.ndarray,
                    phi: np.ndarray) -> np.ndarray:
    """Half-integer spin-``s`` basis at arbitrary ``(theta, phi)`` -- the
    off-grid counterpart of :func:`half_basis`.

    ``(npts, nmodes)``, columns following :func:`half_modes`. Needed wherever
    a spinor has to be evaluated somewhere other than the solve's own
    quadrature grid -- e.g. at the vertices of a traced quad mesh, which is
    what §P7 route 2 lifts.

    The values are **not** single-valued functions on the sphere: ``m`` is a
    half-integer, so ``e^{i m phi}`` is antiperiodic and the sign depends on
    the branch of ``phi``. Anything built from these has to be quadratic in
    the spinor (as ``|lambda|^2`` and ``lambda-bar df lambda`` both are) for
    that sign to cancel.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    modes = half_modes(s, L)
    col = {lm: i for i, lm in enumerate(modes)}
    Y = np.empty((theta.size, len(modes)), dtype=complex)
    m = -L
    while m <= L + 1e-9:
        ls, d = wigner_d_ladder(L, m, -s, theta)
        eim = np.exp(1j * m * phi)
        for i, l in enumerate(ls):
            Y[:, col[(l, m)]] = sqrt((2 * l + 1) / (4.0 * np.pi)) * d[i] * eim
        m += 1.0
    return Y

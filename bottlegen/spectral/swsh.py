"""Spin-weighted spherical-harmonic (SWSH) transforms and the ð ladder.

Self-contained NumPy/SciPy implementation (float64, zero heavy deps). The basis
is built from the Wigner small-d function (Wigner sum formula); the convention is
verified in the tests against the ``spherical`` package (independent oracle) at
every spin and scipy ``sph_harm_y`` at spin 0, namely

    ₍s₎Y_{ℓm}(θ,φ) = √((2ℓ+1)/4π) d^ℓ_{m,−s}(θ) e^{imφ}   (Condon-Shortley).

Coefficients live in a dense complex array ``C[ℓ, ℓ+m]`` of shape
``(L+1, 2L+1)`` (column index ``ℓ+m`` ∈ [0, 2ℓ]); entries with ℓ < |s| or
|m| > ℓ are zero.

Transforms use a Gauss-Legendre (θ) x equiangular (φ) grid with ``Nθ = L+1`` and
``Nφ = 2L+1``, which makes ``analy∘synth = id`` exactly for band-limited spin
fields (the relevant products are degree-≤2L polynomials in cosθ). The sum
formula is accurate to ~1e-12 at the project's default L=16 (and stays under the
test tolerances through L≈24, beyond which cancellation in the factorial sum
starts to matter).
"""

from functools import lru_cache
from math import factorial as _fac, sqrt

import numpy as np

from .grids import gl_grid, solid_angle_weights

__all__ = [
    "wigner_d", "swsh", "empty_coeffs", "synth", "analy",
    "edth", "edthbar", "edth_factor", "edthbar_factor",
    "spin_laplacian", "laplacian_eig", "hopf_scaling", "q_from_sigma",
]


# ---------------------------------------------------------------------------
# Wigner small-d
# ---------------------------------------------------------------------------

def wigner_d(l: int, a: int, b: int, theta: np.ndarray) -> np.ndarray:
    """Wigner small-d ``d^l_{a,b}(θ)`` via Wigner's sum formula (exact signs).

    ``theta`` may be an array; the result has the same shape. Valid for integer
    ``l ≥ 0`` and ``|a|, |b| ≤ l``.
    """
    theta = np.asarray(theta, dtype=float)
    if abs(a) > l or abs(b) > l:
        return np.zeros_like(theta)
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    pref = sqrt(_fac(l + a) * _fac(l - a) * _fac(l + b) * _fac(l - b))
    kmin = max(0, b - a)
    kmax = min(l + b, l - a)
    out = np.zeros_like(theta)
    for k in range(kmin, kmax + 1):
        denom = _fac(l + b - k) * _fac(k) * _fac(a - b + k) * _fac(l - a - k)
        cpow = 2 * l + b - a - 2 * k
        spow = a - b + 2 * k
        out += ((-1) ** (a - b + k)) * (pref / denom) * (c ** cpow) * (s ** spow)
    return out


# ---------------------------------------------------------------------------
# SWSH basis on the canonical GL grid (cached per (s, L))
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _basis(s: int, L: int):
    """Dict ``(ℓ,m) -> ₍s₎Y_{ℓm}`` sampled on ``gl_grid(L)`` (complex, (Nθ,Nφ))."""
    grid = gl_grid(L)
    th, ph = grid.theta, grid.phi
    eim = {m: np.exp(1j * m * ph) for m in range(-L, L + 1)}
    Y = {}
    for l in range(abs(s), L + 1):
        Nl = sqrt((2 * l + 1) / (4.0 * np.pi))
        for m in range(-l, l + 1):
            # Condon-Shortley phase sits inside d^ℓ_{m,-s}; with this
            # normalization spin 0 matches scipy.sph_harm_y exactly.
            d = wigner_d(l, m, -s, th)
            Y[(l, m)] = Nl * d[:, None] * eim[m][None, :]
    return Y


def swsh(s: int, l: int, m: int, L: int) -> np.ndarray:
    """Single spin-weighted harmonic ``₍s₎Y_{ℓm}`` on ``gl_grid(L)``."""
    return _basis(s, L)[(l, m)]


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
    Y = _basis(s, L)
    grid = gl_grid(L)
    field = np.zeros((grid.ntheta, grid.nphi), dtype=complex)
    for l in range(abs(s), L + 1):
        for m in range(-l, l + 1):
            field += C[l, l + m] * Y[(l, m)]
    return field


def analy(field: np.ndarray, s: int, L: int) -> np.ndarray:
    """Forward transform: spin-``s`` field on ``gl_grid(L)`` -> coefficients ``C``."""
    Y = _basis(s, L)
    grid = gl_grid(L)
    fw = field * solid_angle_weights(grid)
    C = empty_coeffs(L)
    for l in range(abs(s), L + 1):
        for m in range(-l, l + 1):
            C[l, l + m] = np.sum(fw * np.conj(Y[(l, m)]))
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

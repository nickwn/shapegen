"""The analytic forward blur, its reverse velocity, and the terminal prior.

``κ_ℓ = (ℓ−1)ℓ(ℓ+1)(ℓ+2)`` is the per-degree Willmore-Hessian rate; high
degrees damp fastest, so the blur leaves only low-ℓ structure at large ``t``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kappa", "blur", "reverse_velocity", "TerminalPrior",
           "tmax_for_band"]


def kappa(L: int) -> np.ndarray:
    """Per-degree blur rate ``κ_ℓ = (ℓ−1)ℓ(ℓ+1)(ℓ+2)`` for ℓ ∈ [0, L]
    (``κ_0 = κ_1 = 0``)."""
    l = np.arange(L + 1)
    return (l - 1) * l * (l + 1) * (l + 2)


def blur(sigma_hat: np.ndarray, t: float, kap: np.ndarray) -> np.ndarray:
    """Forward blur ``σ̂(t) = σ̂ · e^{-κ_ℓ t}`` (per degree)."""
    return sigma_hat * np.exp(-kap[:, None] * t)


def reverse_velocity(sigma_hat: np.ndarray, t: float, kap: np.ndarray) -> np.ndarray:
    """Analytic deblur velocity ``+κ_ℓ σ̂`` (``d/dt`` of the reverse process; a
    forward Euler step ``σ̂ + dt·κ σ̂`` recovers a less-blurred state to first
    order)."""
    return kap[:, None] * sigma_hat


def tmax_for_band(L: int, top_band_damp: float = 1e-3) -> float:
    """``t_max`` such that the top degree ``L`` damps to ``top_band_damp``:
    ``e^{-κ_L t_max} = top_band_damp``."""
    kL = float(kappa(L)[L])
    return -np.log(top_band_damp) / max(kL, 1.0)


class TerminalPrior:
    """Diagonal Gaussian over the surviving (low-ℓ) modes at ``t_max``.

    Cold-diffusion sampling seeds from this prior, so it is the *only* source of
    sample diversity — keep enough low-ℓ bands alive at ``t_max``.
    """

    def __init__(self):
        self.mean = None      # complex (L+1, 2L+1)
        self.std = None       # real    (L+1, 2L+1)
        self.mask = None      # bool, which modes are "alive"

    def fit(self, blurred_coeffs: np.ndarray, alive_tol: float = 1e-6):
        """Fit per-mode mean/std from a stack of blurred training coeffs
        ``(N, L+1, 2L+1)``."""
        C = np.asarray(blurred_coeffs)
        self.mean = C.mean(0)
        self.std = C.std(0)
        self.mask = (np.abs(self.mean) + self.std) > alive_tol
        return self

    def sample(self, n: int, rng=None) -> np.ndarray:
        """Draw ``n`` seeds ``(n, L+1, 2L+1)`` (complex), Hermitian-symmetrized
        so the implied σ field is real."""
        rng = np.random.default_rng() if rng is None else rng
        shape = (n,) + self.mean.shape
        re = rng.standard_normal(shape)
        im = rng.standard_normal(shape)
        out = self.mean[None] + self.std[None] * (re + 1j * im) / np.sqrt(2)
        out *= self.mask[None]
        return _hermitian_symmetrize(out)


def _hermitian_symmetrize(C: np.ndarray) -> np.ndarray:
    """Enforce ``C[ℓ,−m] = (−1)^m conj(C[ℓ,m])`` (real σ field), batched over a
    leading axis if present."""
    C = C.copy()
    Lp1 = C.shape[-2]
    L = Lp1 - 1
    for l in range(Lp1):
        for m in range(1, l + 1):
            avg_pos = 0.5 * (C[..., l, l + m]
                             + (-1) ** m * np.conj(C[..., l, l - m]))
            C[..., l, l + m] = avg_pos
            C[..., l, l - m] = (-1) ** m * np.conj(avg_pos)
        C[..., l, l] = np.real(C[..., l, l])    # m=0 real
    return C

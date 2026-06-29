"""Quadrature grid for spin-weighted spherical-harmonic transforms.

Gauss-Legendre in the polar angle (nodes in ``x = cos θ``) and equiangular in
the azimuth. With ``Nθ = L+1`` GL nodes and ``Nφ = 2L+1`` azimuthal samples the
analysis/synthesis pair is exact (machine precision) for band-limited spin
fields up to degree ``L``: the relevant products ``₍s₎Y_{ℓm} ₍s₎Ȳ_{ℓ'm'}`` are
polynomials of degree ≤ 2L in ``cos θ``, which GL integrates exactly.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GLGrid:
    """Gauss-Legendre x equiangular grid for bandlimit ``L`` (max degree)."""

    L: int
    theta: np.ndarray   # (Nθ,) polar angles in (0, π)
    phi: np.ndarray     # (Nφ,) azimuthal angles in [0, 2π)
    wtheta: np.ndarray  # (Nθ,) GL weights in x = cos θ (carry the sinθ dθ measure)

    @property
    def ntheta(self) -> int:
        return self.theta.size

    @property
    def nphi(self) -> int:
        return self.phi.size


def gl_grid(L: int) -> GLGrid:
    """Build the canonical GL grid for bandlimit ``L``."""
    ntheta = L + 1
    nphi = 2 * L + 1
    x, w = np.polynomial.legendre.leggauss(ntheta)  # x in (-1, 1), ascending
    theta = np.arccos(x)
    phi = 2.0 * np.pi * np.arange(nphi) / nphi
    return GLGrid(L=L, theta=theta, phi=phi, wtheta=w)


def solid_angle_weights(grid: GLGrid) -> np.ndarray:
    """Weights ``w[j,k]`` for ``∫ f dΩ ≈ Σ w[j,k] f(θ_j, φ_k)``.

    ``dΩ = sinθ dθ dφ``; the GL weights carry ``sinθ dθ`` (via x=cosθ), and the
    azimuth contributes the uniform factor ``2π/Nφ``.
    """
    return grid.wtheta[:, None] * (2.0 * np.pi / grid.nphi)

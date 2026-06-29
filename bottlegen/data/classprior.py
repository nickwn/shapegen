"""The empirical class prior — the dataset's σ̂ distribution.

The "bottle class" is defined empirically as the distribution of cached σ̂. This
module summarizes it (per-degree RMS spectrum, mean, optionally a rotation-
averaged isotropic version) for diagnostics, whitening, and as a sanity baseline
for generated samples.
"""

from __future__ import annotations

import numpy as np

from ..flow import state
from .dataset import random_so3, wigner_rotate_sigma

__all__ = ["ClassPrior"]


class ClassPrior:
    """Summary statistics of a σ̂ dataset."""

    def __init__(self, sigma_stack: np.ndarray):
        self.C = np.asarray(sigma_stack)        # (N, L+1, 2L+1) complex
        self.L = self.C.shape[-2] - 1
        self.mean = self.C.mean(0)
        self.perdeg_rms = state.perdeg_std(self.C)      # (L+1,)

    @property
    def power_spectrum(self) -> np.ndarray:
        """Mean per-degree power ``Σ_m E|σ̂_{ℓm}|²`` (ℓ ∈ [0, L])."""
        return np.array([np.mean(np.sum(np.abs(self.C[:, l, 0:2 * l + 1]) ** 2,
                                        axis=1))
                         for l in range(self.L + 1)])

    def rotation_augmented_spectrum(self, n_rot=8, rng=None) -> np.ndarray:
        """Power spectrum after averaging random rotations — should match
        :attr:`power_spectrum` (rotation only mixes orders within a degree, so
        per-degree power is invariant)."""
        rng = np.random.default_rng() if rng is None else rng
        acc = np.zeros(self.L + 1)
        for s in self.C:
            for _ in range(n_rot):
                sr = wigner_rotate_sigma(s, random_so3(rng))
                acc += np.array([np.sum(np.abs(sr[l, 0:2 * l + 1]) ** 2)
                                 for l in range(self.L + 1)])
        return acc / (len(self.C) * n_rot)

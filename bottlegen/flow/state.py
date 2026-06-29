"""State packing for the SWSH potential σ̂.

The state lives in ℓ ∈ [2, L] (ℓ≤1 gauge-fixed to zero). Two real
representations are provided:

* **flat** — a single real vector (Re then Im of every active mode), for prior
  fitting / whitening.
* **per-degree** — a list of real tensors, one per degree ℓ, each of length
  ``2(2ℓ+1)`` (Re and Im of the ``2ℓ+1`` orders). This is the degree-tokenized
  layout the DiT consumes.

All packs are exact round trips with ``unpack``. ``gauge_fix`` (zero ℓ≤1) is
idempotent.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gauge_fix", "active_degrees", "real_pack", "real_unpack",
    "perdeg_pack", "perdeg_unpack", "flat_dim", "perdeg_std", "sobolev_weights",
]


def gauge_fix(coeffs: np.ndarray) -> np.ndarray:
    """Zero the ℓ≤1 modes (ℓ=0 scale, ℓ=1 translation/Möbius). Idempotent."""
    out = coeffs.copy()
    out[0] = 0.0
    if out.shape[0] > 1:
        out[1] = 0.0
    return out


def active_degrees(L: int):
    """Degrees carrying state: ``range(2, L+1)``."""
    return list(range(2, L + 1))


def real_pack(coeffs: np.ndarray) -> np.ndarray:
    """Flatten ℓ∈[2,L] modes to a real vector ``[Re…, Im…]`` (length
    ``2·Σ(2ℓ+1)``)."""
    L = coeffs.shape[0] - 1
    parts = []
    for l in active_degrees(L):
        row = coeffs[l, l - l: l + l + 1]      # m ∈ [-ℓ, ℓ]
        parts.append(np.concatenate([row.real, row.imag]))
    return np.concatenate(parts) if parts else np.zeros(0)


def real_unpack(vec: np.ndarray, L: int) -> np.ndarray:
    """Inverse of :func:`real_pack` into a ``(L+1, 2L+1)`` complex array."""
    out = np.zeros((L + 1, 2 * L + 1), dtype=complex)
    i = 0
    for l in active_degrees(L):
        w = 2 * l + 1
        re = vec[i:i + w]
        im = vec[i + w:i + 2 * w]
        out[l, 0:w] = re + 1j * im
        i += 2 * w
    return out


def perdeg_pack(coeffs: np.ndarray):
    """List of per-degree real tensors (length ``2(2ℓ+1)`` each)."""
    L = coeffs.shape[0] - 1
    out = []
    for l in active_degrees(L):
        row = coeffs[l, 0:2 * l + 1]
        out.append(np.concatenate([row.real, row.imag]))
    return out


def perdeg_unpack(perdeg, L: int) -> np.ndarray:
    """Inverse of :func:`perdeg_pack`."""
    out = np.zeros((L + 1, 2 * L + 1), dtype=complex)
    for k, l in enumerate(active_degrees(L)):
        w = 2 * l + 1
        v = np.asarray(perdeg[k])
        out[l, 0:w] = v[:w] + 1j * v[w:2 * w]
    return out


def flat_dim(L: int) -> int:
    """Length of the flat real vector for bandlimit ``L``."""
    return int(sum(2 * (2 * l + 1) for l in active_degrees(L)))


def perdeg_std(coeffs_stack: np.ndarray) -> np.ndarray:
    """Per-degree RMS amplitude over a dataset ``(N, L+1, 2L+1)`` (for Sobolev
    weighting / whitening). Returns shape ``(L+1,)`` (ℓ≤1 = 0)."""
    C = np.asarray(coeffs_stack)
    L = C.shape[-2] - 1
    std = np.zeros(L + 1)
    for l in range(2, L + 1):
        std[l] = np.sqrt(np.mean(np.abs(C[:, l, 0:2 * l + 1]) ** 2)) + 1e-12
    return std


def sobolev_weights(L: int, beta: float = 1.0) -> np.ndarray:
    """Per-degree Sobolev weights ``(1 + ℓ(ℓ+1))^β`` for ℓ ∈ [0, L] (loss
    weighting that emphasizes smoothness)."""
    l = np.arange(L + 1)
    return (1.0 + l * (l + 1)) ** beta

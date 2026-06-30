"""Robust σ̂ → mesh via an amplitude trust region around the Dirac decode.

The linearized decode drives the curvature change by ``ρ̇ = −(ℓ(ℓ+1))² σ̂``;
that ``(ℓ(ℓ+1))²`` gain means high-ℓ content (sharp features) pushes the spin
transform into the nonlinear regime where the surface self-folds. Rather than
return folded garbage, :func:`decode` takes the **largest amplitude scale**
``s ∈ (0,1]`` whose linear-init decode is a valid, non-degenerate genus-0 mesh
(no flipped faces, bounded Willmore). Where ``s = 1`` is stable (typically
``L ≲ 16``) it is returned unchanged; where it would fold, a flatter but valid
shape is returned. Optional damped encode-residual refinement is available but
off by default (it does not improve spectral fidelity over the linear init).
"""

from __future__ import annotations

import numpy as np

from ..eval.metrics import flip_fraction, willmore_energy
from ..geom import dirac

__all__ = ["decode", "DEFAULT_AMPLITUDES"]

DEFAULT_AMPLITUDES = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12)


def decode(base_V, base_F, sigma_hat, *, max_flip=0.02, w_cap=None,
           amplitudes=DEFAULT_AMPLITUDES, refine_iters=1, encode=None,
           sigma=1e-6, return_info=False):
    """Robust σ̂ → ``(V, F)`` on the base-sphere connectivity.

    Tries amplitudes high→low and returns the first whose decode is valid
    (finite, ``flip ≤ max_flip``, ``W ≤ w_cap`` if given). If none qualify, the
    lowest-flip attempt is returned. With ``return_info`` also returns a dict
    ``{amplitude, flip, W, valid}``.
    """
    fallback = None
    for s in amplitudes:
        V, F = dirac.reconstruct(base_V, base_F, s * sigma_hat,
                                 iters=refine_iters, encode=encode, sigma=sigma)
        finite = np.isfinite(V).all()
        ff = flip_fraction(V, F) if finite else 1.0
        W = willmore_energy(V, F) if finite else np.inf
        valid = finite and ff <= max_flip and (w_cap is None or W <= w_cap)
        info = {"amplitude": s, "flip": ff, "W": float(W), "valid": valid}
        if valid:
            return (V, F, info) if return_info else (V, F)
        if fallback is None or ff < fallback[2]["flip"]:
            fallback = (V, F, info)
    V, F, info = fallback
    return (V, F, info) if return_info else (V, F)

"""The representation layer: meshes <-> coefficient vectors.

The generative model in :mod:`bottlegen.models.flow` never sees a mesh — it sees
an ``(N, K)`` matrix of codes. Everything that decides *what those codes mean*
lives here, so swapping the representation is swapping one object.

Two implementations of the same interface:

``PCABasis``
    Positions in a data-adapted basis. For registered data this wins by a wide
    margin — measured on 200 FaceScape heads (train 160, held-out 40, scored
    against the smoothed target), **k=32 gives 0.0082** against the ρ codec's
    0.0209 at L=32, i.e. 2.6x better accuracy from 34x fewer numbers. It needs a
    solved correspondence, which registered datasets have by construction.

``RhoBasis``
    ρ̂, the spin-transform code, via :mod:`bottlegen.codec`. Needs no
    correspondence, works on unregistered genus-0 shapes, and always decodes to
    *some* surface. Far more expensive (the adjoint encoder is ~19 s/mesh at
    subdiv 3 and ~350 s at 26k vertices) and, for faces, less accurate. This is
    the right choice for the bottle-style data the codec was built for.

Both expose ``fit`` / ``encode`` / ``decode`` on **whitened** codes. Whitening is
not cosmetic: the flow-matching loss is an unweighted MSE in code space, so
without it the model fits whichever coordinates happen to be largest and ignores
the rest. In whitened PCA coordinates the data is close to isotropic, which also
makes ``N(0, I)`` a meaningful baseline to beat.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PCABasis", "RhoBasis"]


class PCABasis:
    """Linear basis over vertex positions, fitted by SVD.

    Codes are **whitened** principal-component coordinates: unit variance per
    component on the training set, so a well-fit generative model should produce
    something close to ``N(0, I)`` and the interesting question is what
    non-Gaussian structure it adds. That is the point of the whitening — it makes
    the trivial baseline explicit rather than hiding it.
    """

    def __init__(self, mean, comps, scale):
        self.mean = mean            # (3n,)
        self.comps = comps          # (k, 3n) orthonormal rows
        self.scale = scale          # (k,) per-component std on the training set

    @property
    def k(self) -> int:
        return len(self.comps)

    @classmethod
    def fit(cls, X: np.ndarray, k: int) -> "PCABasis":
        """Fit on ``X`` ``(N, n, 3)``. ``k`` is capped at ``N-1``.

        Capped because that is the rank of a centred ``N``-sample covariance —
        asking for more returns components fitted to nothing, which look like
        signal and are noise.
        """
        A = np.asarray(X, dtype=float).reshape(len(X), -1)
        mean = A.mean(0)
        U, s, Vt = np.linalg.svd(A - mean, full_matrices=False)
        k = min(k, len(A) - 1)
        scale = s[:k] / np.sqrt(len(A) - 1)
        return cls(mean, Vt[:k], np.maximum(scale, 1e-12))

    def encode(self, X: np.ndarray) -> np.ndarray:
        """``(N, n, 3) -> (N, k)`` whitened codes."""
        A = np.asarray(X, dtype=float).reshape(len(X), -1)
        return ((A - self.mean) @ self.comps.T) / self.scale

    def decode(self, C: np.ndarray) -> np.ndarray:
        """``(N, k) -> (N, n, 3)``."""
        C = np.atleast_2d(np.asarray(C, dtype=float))
        A = (C * self.scale) @ self.comps + self.mean
        return A.reshape(len(C), -1, 3)

    def explained(self) -> np.ndarray:
        """Cumulative fraction of training variance per component count."""
        v = self.scale ** 2
        return np.cumsum(v) / v.sum()


class RhoBasis:
    """ρ̂ codes from :mod:`bottlegen.codec`, on one shared template sphere.

    The sphere is fixed at fit time and travels with the basis, because ρ̂ is
    defined *relative to* a base sphere: decoding against a different
    parameterisation of the same shape gives a different mesh. At generation time
    there is no mesh and hence no cMCF sphere to recover, so it must be shared.

    **The rotation gauge is not fixed.** Möbius normalisation pins ℓ=0 and ℓ=1;
    nothing pins rotation, so two shapes differing by a rotation encode to
    entirely different ρ̂. Canonicalise orientation before fitting, or the model
    will spend its capacity learning the orientation distribution.
    """

    def __init__(self, sphere, faces, L, method="adjoint", mean=None, scale=None):
        self.sphere, self.faces, self.L, self.method = sphere, faces, L, method
        self.mean, self.scale = mean, scale

    @property
    def k(self) -> int:
        from ..codec import n_coeffs  # noqa: PLC0415
        return n_coeffs(self.L)

    def _raw(self, X):
        from .. import codec  # noqa: PLC0415
        from ..geom.mesh import Mesh  # noqa: PLC0415
        return np.stack([
            codec.encode(Mesh(x, self.faces), self.L, sphere=self.sphere,
                         method=self.method).coeffs for x in X])

    @classmethod
    def fit(cls, X, faces, L, sphere, method="adjoint") -> "RhoBasis":
        b = cls(sphere, faces, L, method)
        C = b._raw(X)
        b.mean = C.mean(0)
        # Band-wise scale, not per-coefficient: the 2l+1 coefficients of a band
        # share a physical scale, so pooling them estimates it from 2l+1 times as
        # much data. Per-coefficient std is noisy when N is not >> K.
        from ..codec import band_index  # noqa: PLC0415
        bi = band_index(L)
        sd = np.empty_like(b.mean)
        for l in range(L + 1):
            m = bi == l
            sd[m] = max(float(np.std(C[:, m])), 1e-12)
        b.scale = sd
        return b

    def encode(self, X: np.ndarray) -> np.ndarray:
        return (self._raw(X) - self.mean) / self.scale

    def decode(self, C: np.ndarray) -> np.ndarray:
        from .. import codec  # noqa: PLC0415
        C = np.atleast_2d(np.asarray(C, dtype=float)) * self.scale + self.mean
        return np.stack([codec.decode(c, self.sphere, self.faces) for c in C])

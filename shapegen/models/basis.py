"""The representation layer: meshes <-> coefficient vectors.

The generative model in :mod:`shapegen.models.flow` never sees a mesh — it sees
an ``(N, K)`` matrix of codes. Everything that decides *what those codes mean*
lives here, so swapping the representation is swapping one object.

Two implementations of the same interface:

``PCABasis``
    Positions in a data-adapted basis. Needs a solved correspondence, which
    registered datasets have by construction; on 200 FaceScape heads (train 160,
    held-out 40, scored against the smoothed target) k=32 gives 0.0082.

``RhoBasis``
    ρ̂, the spin-transform code, via :mod:`shapegen.codec`. Needs no
    correspondence, works on unregistered genus-0 shapes, always decodes to *some*
    surface, and hands you γ as a free validity score. It is the path that
    generalises off a registered dataset.

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
    """ρ̂ codes from :mod:`shapegen.codec`, on one shared template sphere.

    The sphere is fixed at fit time and travels with the basis: ρ̂ is defined
    relative to a sphere parameterisation, and at generation time there is no
    mesh to parameterise, so it must be shared. Decoding is the mesh-free
    spectral decoder at the sphere's points.

    Codes are ``quotient`` encodes against that sphere
    (:func:`shapegen.geom.dirac_quotient.encode_quotient`, without the closing
    solve: ~80 ms per mesh, and ``ℓ = 0`` does not change the decode), the same
    encoder ``codec.encode(..., method="quotient")`` scores, so training and
    evaluation codes agree.

    **Codes see pose.** Against a fixed sphere a random rotation of a head moves
    its code by ~7% at L=32 (5-7% on three FaceScape heads, against ~31%
    between subjects), so Procrustes-align the set first
    (:func:`shapegen.data.faces.build_dataset` does by default).
    """

    def __init__(self, sphere, faces, L, mean=None, scale=None):
        self.sphere, self.faces, self.L = sphere, faces, L
        self.mean, self.scale = mean, scale

    @property
    def k(self) -> int:
        from ..codec import n_coeffs  # noqa: PLC0415
        return n_coeffs(self.L)

    def _raw(self, X) -> np.ndarray:
        """``(N, n, 3)`` positions on the shared connectivity → ``(N, K)`` raw ρ̂."""
        from ..codec import _spectral_context  # noqa: PLC0415
        from ..geom.dirac_quotient import encode_quotient  # noqa: PLC0415
        lift, Ls = _spectral_context(self.L)
        return np.stack([encode_quotient(x, self.sphere, self.faces, self.L, lift=lift,
                                         Ls=Ls, close=False)["coeffs"] for x in X])

    @classmethod
    def fit(cls, X, faces, L, sphere) -> "RhoBasis":
        b = cls(sphere, faces, L)
        C = b._raw(X)
        b.mean = C.mean(0)
        # Band-wise scale, not per-coefficient: the 2l+1 coefficients of a band
        # share a physical scale, so pooling them estimates it from 2l+1 times as
        # much data. Per-coefficient std is noisy when N is not >> K.
        #
        # Pool the CENTRED coefficients: pooling raw ones folds the spread of the
        # per-coefficient means into the scale and leaves the whitened codes
        # under-normalised (measured per-band std 0.25 instead of 1).
        from ..codec import band_index  # noqa: PLC0415
        bi = band_index(L)
        Cc = C - b.mean
        sd = np.empty_like(b.mean)
        for l in range(L + 1):
            m = bi == l
            sd[m] = max(float(np.std(Cc[:, m])), 1e-12)
        b.scale = sd
        return b

    def encode(self, X: np.ndarray) -> np.ndarray:
        return (self._raw(X) - self.mean) / self.scale

    def decode(self, C: np.ndarray) -> np.ndarray:
        from .. import codec  # noqa: PLC0415
        C = np.atleast_2d(np.asarray(C, dtype=float)) * self.scale + self.mean
        return np.stack([codec.decode(c, self.sphere, self.faces) for c in C])

    def rho_field(self, c: np.ndarray) -> np.ndarray:
        """One whitened code → the per-vertex ρ on the base sphere.

        Exposed because γ, the Dirac residual of that field, is a **free validity
        score**: γ ≈ 0 means the code is exactly realisable as an immersion, and a
        generated code whose γ sits far outside the training range is off-manifold
        in a way no reconstruction metric will show you. No other representation
        here hands you one.
        """
        from ..geom import encode as _enc  # noqa: PLC0415
        Y = _enc.real_sh_basis(self.sphere, self.L)
        return Y @ (np.asarray(c, dtype=float) * self.scale + self.mean)

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
    surface, and hands you γ as a free validity score. The ``direct`` codes
    training uses are less accurate than PCA on registered faces -- that is the
    trade -- but it is the path that generalises off a registered dataset.

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

    On registered data against a shared template, ``direct`` codes are exactly
    rotation-invariant (measured 3.7e-15 under a random rotation), since ``H``
    and the edge-length conformal factor are -- so no rotation gauge is needed.

    ``method="direct"`` (default) is what training uses: milliseconds per mesh,
    a measurement comparable across meshes, and its fields are reusable across
    ``L``. ``method="spectral"`` runs the adjoint encoder per mesh against the
    shared sphere -- ~5x more accurate, minutes per mesh.
    """

    def __init__(self, sphere, faces, L, method="direct", mean=None, scale=None):
        self.sphere, self.faces, self.L, self.method = sphere, faces, L, method
        self.mean, self.scale = mean, scale

    @property
    def k(self) -> int:
        from ..codec import n_coeffs  # noqa: PLC0415
        return n_coeffs(self.L)

    def rho_fields(self, X) -> np.ndarray:
        """``(N, n)`` per-vertex ρ for every shape, before any band-limiting.

        **ρ_direct does not depend on L** -- only the projection does -- and the
        per-mesh ``u_edge_lsq`` (an LSQR over every edge) is the expensive part,
        so compute the fields once and pass them as ``fields=`` to :meth:`fit` /
        :meth:`encode`, or cache them with ``scripts/encode_faces.py``.

        Only for ``method="direct"``: a ``spectral`` code is fitted at a
        specific bandwidth and cannot be reused across L.
        """
        from ..geom.mesh import Mesh  # noqa: PLC0415
        from ..geom.spin import mesh_to_rho_direct  # noqa: PLC0415
        if self.method != "direct":
            raise ValueError(
                f"rho fields are bandwidth-independent only for method='direct'; "
                f"this basis uses {self.method!r}, whose rho is fitted at a "
                f"specific L and cannot be reused across bandwidths")
        return np.stack([mesh_to_rho_direct(Mesh(x, self.faces), self.sphere)
                         for x in X])

    def _project(self, fields) -> np.ndarray:
        """``(N, n)`` ρ fields → ``(N, K)`` raw SH coefficients."""
        from ..geom import encode as _enc  # noqa: PLC0415
        from ..geom.operators import vertex_areas  # noqa: PLC0415
        Y = _enc.real_sh_basis(self.sphere, self.L)
        mass = vertex_areas(self.sphere, self.faces)
        return np.stack([_enc.sh_project(r, Y, mass=mass) for r in fields])

    def _raw(self, X, fields=None):
        if fields is not None:
            return self._project(fields)
        if self.method == "direct":
            return self._project(self.rho_fields(X))
        from .. import codec  # noqa: PLC0415
        from ..geom.mesh import Mesh  # noqa: PLC0415
        return np.stack([
            codec.encode(Mesh(x, self.faces), self.L, sphere=self.sphere,
                         method=self.method).coeffs for x in X])

    @classmethod
    def fit(cls, X, faces, L, sphere, method="direct", fields=None) -> "RhoBasis":
        b = cls(sphere, faces, L, method)
        C = b._raw(X, fields=fields)
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

    def encode(self, X: np.ndarray, fields=None) -> np.ndarray:
        return (self._raw(X, fields=fields) - self.mean) / self.scale

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

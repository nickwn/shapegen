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
    correspondence, works on unregistered genus-0 shapes, always decodes to *some*
    surface, and hands you γ as a free validity score. Less accurate on registered
    faces than PCA — that is the trade — but it is the path that generalises off a
    registered dataset, which is what the rest of this repo is for.

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

    **Rotation needs no gauge fixing here**, contrary to the warning in
    ``codec.py``. On registered data against a shared template, ρ̂ is exactly
    rotation-invariant — measured 3.7e-15 (direct) and 6.9e-10 (adjoint) under a
    random rotation — because ``H`` is rotation-invariant and ``e^u`` comes from
    edge *lengths*. So no Procrustes stage, no Wigner-D augmentation, no
    equivariant architecture. That warning applies to *unregistered* data, where
    each shape gets its own cMCF sphere that is itself only defined up to rotation.

    ``method`` defaults to ``"direct"``: the adjoint costs 321-376 s on a 26k-vertex
    head, which puts a few thousand meshes out of reach, and its codes are not
    comparable across meshes (46% per-band at L=12 against direct's 12%) — a
    defect rather than a detail for a model fitted over a whole dataset.
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

        Split out because **ρ_direct does not depend on L** — only the projection
        does. The expensive part is the per-mesh ``u_edge_lsq`` (an LSQR over
        every edge, 26k vertices for a head), so computing the fields once and
        projecting them at several bandwidths turns an L-sweep from N x |L| solves
        into N. Pass the result back as ``fields=`` to :meth:`fit`, or cache it to
        disk with ``scripts/encode_faces.py``.

        Not available for ``method="adjoint"``, whose ρ *is* fitted at a specific
        bandwidth and so cannot be reused across L.
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
        # Pool the CENTRED coefficients. Pooling raw ones folds the spread of the
        # per-coefficient means into the scale, which over-estimates it and leaves
        # the whitened codes under-normalised -- measured, per-band std came out
        # at 0.25 instead of 1, and the flow then under-weights those bands for
        # exactly the reason whitening exists.
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

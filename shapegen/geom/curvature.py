"""Discrete mean curvature.

Convention (``CONVENTIONS.md``): outward normal, convex-positive, so a sphere
of radius R has ``H = +1/R``. ``Hvec = ½M⁻¹LV`` is the mean-curvature vector;
its sign comes from projecting on the outward normal.

The Hopf differential and second-fundamental-form machinery that lived here
served the σ̂ representation and went with it.
"""

from __future__ import annotations

import numpy as np

from .operators import cotan_laplacian, vertex_normals

__all__ = ["mean_curvature"]


def mean_curvature(V, F, L=None, M=None):
    """Pointwise mean curvature ``H`` (convex-positive), the mean-curvature
    vector, and outward normals, from ``Hvec = ½ M⁻¹ L V``.

    Returns ``(H, Hvec, normals)``.
    """
    if L is None or M is None:
        L, M = cotan_laplacian(V, F)
    Minv = 1.0 / M.diagonal()
    Hvec = 0.5 * (Minv[:, None] * (L @ V))
    n = vertex_normals(V, F)
    H = np.linalg.norm(Hvec, axis=1) * np.sign((Hvec * n).sum(1))
    return H, Hvec, n


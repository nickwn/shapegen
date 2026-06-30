"""Per-mesh spectral + spatial operators for DiffusionNet (Sharp et al.).

These are the cached geometric operators the SOTA encoder consumes:

* **LB eigenbasis** ``(evals, evecs)`` — the ``k`` smallest generalized
  eigenpairs of the cotan Laplacian ``L φ = λ M φ`` — used for learned spectral
  heat diffusion.
* **Tangent gradient operators** ``(gradX, gradY)`` — sparse ``|V|×|V|`` maps
  taking a per-vertex scalar to the two tangent-frame components of its surface
  gradient (per-vertex least squares over the 1-ring), used for the
  rotation-invariant spatial gradient features.

Computed in float64 on CPU (the precision split); converted to float32 tensors
for the network at load time.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .curvature import tangent_frames, vertex_one_ring
from .operators import cotan_laplacian, vertex_normals

__all__ = ["lb_eigenbasis", "tangent_gradient_operators", "mesh_operators"]


def lb_eigenbasis(V, F, k=64):
    """Smallest ``k`` generalized eigenpairs of the cotan Laplacian.

    Returns ``(evals (k,), evecs (|V|, k))`` sorted ascending (``evals[0] ≈ 0``,
    constant mode). Uses shift-invert just below 0.
    """
    L, M = cotan_laplacian(V, F)
    k = int(min(k, V.shape[0] - 2))
    vals, vecs = spla.eigsh(L.tocsc(), k=k, M=M.tocsc(), sigma=-1e-6,
                            which="LM")
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def tangent_gradient_operators(V, F, reg=1e-8):
    """Sparse tangent-gradient operators ``(gradX, gradY)`` (CSR, ``|V|×|V|``).

    ``(gradX @ f)_i`` and ``(gradY @ f)_i`` are the ``e1`` / ``e2`` components of
    the least-squares surface gradient of ``f`` at vertex ``i`` (fit over the
    1-ring tangential edges).
    """
    e1, e2, _ = tangent_frames(V, F)
    rings = vertex_one_ring(V, F)
    n = len(V)
    rx, cx, vx = [], [], []
    ry, cy, vy = [], [], []
    for i in range(n):
        nb = rings[i]
        if len(nb) < 2:
            continue
        d = V[nb] - V[i]
        A = np.stack([d @ e1[i], d @ e2[i]], axis=1)        # (deg, 2)
        AtA = A.T @ A + reg * np.eye(2)
        pinv = np.linalg.solve(AtA, A.T)                    # (2, deg)
        wx, wy = pinv[0], pinv[1]
        rx.extend([i] * len(nb)); cx.extend(nb); vx.extend(wx)
        rx.append(i); cx.append(i); vx.append(-wx.sum())
        ry.extend([i] * len(nb)); cy.extend(nb); vy.extend(wy)
        ry.append(i); cy.append(i); vy.append(-wy.sum())
    gradX = sp.coo_matrix((vx, (rx, cx)), shape=(n, n)).tocsr()
    gradY = sp.coo_matrix((vy, (ry, cy)), shape=(n, n)).tocsr()
    return gradX, gradY


def mesh_operators(V, F, k=64):
    """Bundle the DiffusionNet operators + input features for one mesh.

    Returns a dict with ``evals, evecs, gradX, gradY, mass, points, normals``
    (NumPy). ``points``/``normals`` are the raw per-vertex input features.
    """
    evals, evecs = lb_eigenbasis(V, F, k)
    gradX, gradY = tangent_gradient_operators(V, F)
    _, M = cotan_laplacian(V, F)
    return {
        "evals": evals, "evecs": evecs, "gradX": gradX, "gradY": gradY,
        "mass": M.diagonal(), "points": V.copy(),
        "normals": vertex_normals(V, F),
    }

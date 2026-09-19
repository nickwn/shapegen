"""Generative modelling over shape codes.

``basis.py``  meshes <-> coefficient vectors: :class:`PCABasis` over positions,
              or :class:`RhoBasis` over rho-hat.
``flow.py``   rectified-flow generative model over those vectors.

The split is the point: ``flow.py`` only ever sees an ``(N, K)`` array, so the
representation is a swap of one object rather than a rewrite.

**The face pipeline is wired to :class:`PCABasis`**, deliberately and against what
``SH_FLOW_PLAN.md`` proposes. Measured on 200 smoothed FaceScape heads, held out:
PCA at k=64 reconstructs at 0.0058 where rho-hat at L=32 (1089 coefficients,
17x as many) reaches 0.0209. :class:`RhoBasis` is the path that needs no
correspondence and therefore generalises off registered data; it is implemented
and reachable, and switching is one line in ``scripts/train_faces.py``.
"""

from .basis import PCABasis, RhoBasis  # noqa: F401
from .flow import (VelocityField, gaussian_baseline, sample_flow,  # noqa: F401
                   train_flow)

__all__ = ["PCABasis", "RhoBasis", "VelocityField", "train_flow",
           "sample_flow", "gaussian_baseline"]

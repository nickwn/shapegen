"""Synthetic genus-0 test shapes.

Procedural stand-ins for real data while the codec is under development. They
exist to exercise the ρ encode/decode path on something harder than a sphere —
in particular something *non-star-shaped*, which is the case a radial-graph
representation cannot express and the spin transform can.

This module previously also held the ``BottleSigma`` torch dataset and Wigner-D
rotation augmentation for the σ̂ representation; both went with the σ̂ stack. A
real dataset will want its own loader, and the rotation question it was solving
is still open — see ``SIMPLIFICATION_PLAN.md`` §7.
"""

from __future__ import annotations

import numpy as np

__all__ = ["demo_bottle_meshes"]


def demo_bottle_meshes(n=16, subdiv=3, seed=0):
    """Synthesize ``n`` genus-0 'bottle-ish' meshes: a surface of revolution with
    a pinched neck plus random low-order bumps.

    The neck is deep enough that some of these are **non-star-shaped** (no origin
    sees the whole surface), which is the point — that is what forces a genuine
    parameterisation rather than a radial height field.
    """
    from ..geom.mesh import Mesh, icosphere
    rng = np.random.default_rng(seed)
    base = icosphere(subdiv)
    V0 = base.vertices
    z = V0[:, 2]
    out = []
    for _ in range(n):
        neck = 1.0 - 0.45 * np.exp(-((z - 0.7) ** 2) / 0.05)
        body = 1.0 + 0.18 * (z ** 2 - 0.3)
        r = neck * body
        r = r + 0.12 * rng.standard_normal() * (V0[:, 0] ** 2 - V0[:, 1] ** 2)
        r = r + 0.10 * rng.standard_normal() * V0[:, 0] * V0[:, 2]
        out.append(Mesh(V0 * r[:, None], base.faces))
    return out

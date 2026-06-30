#!/usr/bin/env python
"""Render the field-aligned cross field + stripe parameterization on a decoded
genus-0 bottle (Session 11 DoD).

    python scripts/quad_demo.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bottlegen.data import dataset
from bottlegen.decode import quadmesh as qm
from bottlegen.decode.reconstruct import decode
from bottlegen.eval import metrics, viz
from bottlegen.geom import conformal as cf
from bottlegen.geom.mesh import icosphere


def main():
    L = 12
    base = icosphere(4)
    m = dataset.demo_bottle_meshes(n=1, subdiv=3, seed=1)[0]
    sig = cf.mesh_to_sigma(m, L=L)
    V, F = decode(base.vertices, base.faces, sig, w_cap=120)

    psi, (E1, E2), _, lam = qm.cross_field(V, F, mult=4, guide_weight=8.0)
    cd = qm.cross_directions(psi, E1, E2, mult=4)
    idx = qm.singularity_indices(V, F, psi, E1, E2, mult=4)
    sp, lam_s = qm.stripe_field(V, F, cd, nu=14.0)
    ph = qm.stripe_phase(sp)

    out = "reports/quad"
    os.makedirs(out, exist_ok=True)
    viz.cross_and_stripes(
        V, F, cd, ph, os.path.join(out, "cross_stripes.png"),
        title=f"decoded bottle (non-star {metrics.nonstar_fraction(V, F):.2f}) — "
              f"Σ index = {idx.sum():.2f} = χ")
    print(f"[quad] Σ index = {idx.sum():.3f} (χ=2), "
          f"#singularities = {int((np.abs(idx) > 0.05).sum())}, "
          f"non-star = {metrics.nonstar_fraction(V, F):.3f}")
    print(f"[done] -> {out}/cross_stripes.png")


if __name__ == "__main__":
    main()

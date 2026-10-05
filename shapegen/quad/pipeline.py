"""ρ̂ → quad mesh in ℝ³, end to end.

    out = quad_mesh(rho_hat, L=32, target_faces=20000)
    out["P3"], out["faces"]          # vertices in ℝ³, quad (and cone) faces

Stages: the spectral spinor and the exact conformal potential
(:mod:`~shapegen.quad.field`), the spin-4 cross field and its Ginzburg-Landau
anneal, cones, the chart and period solve (:mod:`~shapegen.quad.cones`,
:mod:`~shapegen.quad.periods`), the closed-grid trace (:mod:`~shapegen.quad.grid`),
and the lift (:func:`~shapegen.quad.lift.lift_spectral`), which reuses the
spinor the potential was built from. No triangle mesh anywhere.
"""

from __future__ import annotations

import numpy as np

from . import cones as qc
from . import field as qf
from . import grid as qg
from . import lift as ql
from . import periods as qp

__all__ = ["solve_field", "cell_ladder", "build_quad_mesh", "quad_mesh"]


def solve_field(rho_hat: np.ndarray, L: int, S: int = 4, levels=None):
    """``ρ̂ → (psi_linear, psi_GL, modes, cones_linear, cones_GL, spinor)``.

    ``spinor`` is the spectral solve the potential was built from, returned so the
    lift can reuse it.
    """
    spinor = qf.solve_potential_spinor(rho_hat, L)
    A = qf.connection_field(qf.curvature_potential(rho_hat, L, spinor=spinor))
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L, levels=levels)
    # coarser fine-grid padding than the library default (over=7): still plenty
    # for ~8 well-separated cones, and much cheaper at L=32
    cones0 = qf.cones_from_field(psi0, S, L, over=4.0)
    conesG = qf.cones_from_field(psiG, S, L, over=4.0)
    return psi0, psiG, modes, cones0, conesG, spinor


def cell_ladder(s_max: float, ratio: float = 0.92, n: int = 30):
    """Descending cell sizes for :func:`~shapegen.quad.periods.solve_lattice`,
    which takes the first that solves: "the largest cell no coarser than asked".
    Smaller cells solve more easily (the cone motion needed scales with the
    cell). 30 rungs at 0.92 span 8x in cell, ~65x in face count."""
    return tuple(s_max * ratio ** k for k in range(n))


def build_quad_mesh(gl_cones, rho_hat: np.ndarray, L: int, spinor=None,
                    n_samples: int = 4000, seed: int = 0, max_vertices: int = 200000,
                    sublattice: str = "Z[i]", target_faces: int | None = None) -> dict:
    """Cones + ρ̂ → a closed quad mesh in ℝ³ (chart, period solve, trace, lift).

    ``target_faces`` sets the density (otherwise whatever cell the stock sweep
    solves at -- 279 to 8644 faces on three heads); the result is the coarsest
    mesh at least that fine. ``sublattice="(1+i)Z[i]"`` puts every cone on a
    grid vertex (pure quads) at ~4-8x the faces.

    Returns ``ok`` (False if the period solve failed or the trace hit
    ``max_vertices``, with ``reason``), ``chart``, ``solve``, ``grid``
    (:class:`~shapegen.quad.grid.QuadGrid`: ``euler``, ``coverage``, ...),
    ``faces``, ``P3``, ``lift_info``, ``cones_3d`` and ``aniso`` (per-face
    edge-length ratio).
    """
    chart = qc.project_cones(gl_cones, n_samples=n_samples, seed=seed)
    kw = {}
    if target_faces:
        kw["cell_sizes"] = cell_ladder(qg.cell_for_faces(chart["z"], chart["idx"], target_faces))
    solve = qp.solve_lattice(chart["z"], chart["idx"], sublattice=sublattice, **kw)
    if solve["cones"] is None:
        return {"ok": False, "reason": "P5 did not solve at any cell size",
                "chart": chart, "solve": solve, "cones_3d": None}
    cones_3d = list(zip(qc.chart_to_sphere(solve["cones"], chart["Rm"]), chart["idx"]))
    g = qg.quad_grid(solve["cones"], chart["idx"], solve["cell"], max_vertices=max_vertices)
    if not g.converged:
        return {"ok": False, "reason": "P6 hit max_vertices before closing",
                "chart": chart, "solve": solve, "grid": g, "cones_3d": cones_3d}
    # quad_grid returns unit-sphere points; only the pole rotation is undone
    P3, lift_info = ql.lift_spectral(g.points @ chart["Rm"], rho_hat, L, res=spinor)
    return {"ok": True, "chart": chart, "solve": solve, "grid": g, "faces": g.faces,
            "P3": P3, "lift_info": lift_info, "aniso": qg.quad_isotropy(P3, g.faces),
            "cones_3d": cones_3d}


def quad_mesh(rho_hat: np.ndarray, L: int, target_faces: int | None = 20000,
              S: int = 4, levels=None, **kw) -> dict:
    """``ρ̂ → quad mesh``: :func:`solve_field` then :func:`build_quad_mesh`.
    The returned dict also carries ``cones`` (the annealed field's cones) and
    ``spinor``."""
    _p0, _pG, _modes, _c0, conesG, spinor = solve_field(rho_hat, L, S, levels)
    out = build_quad_mesh(conesG, rho_hat, L, spinor=spinor, target_faces=target_faces, **kw)
    out.update(cones=conesG, spinor=spinor)
    return out

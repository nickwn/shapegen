"""Run and visualize the full mesh-free quad-mesh pipeline (PLAN.md §P8.3/
§P8.4, §P4-lite, §P5, §P6, §P7) on real face ρ fields, ending in an actual
quad mesh in ℝ³.

**Where the ρ coefficients come from.** ``scripts/encode_faces.py`` is the
script that builds the face dataset's ρ: it caches, per subject, the
**per-vertex** ρ field on a shared base sphere (not SH coefficients -- see its
docstring, "ρ_direct does not depend on L. Only the projection does"), so one
cache serves every bandwidth. A run of it against this machine's FaceScape
copy is already sitting at ``reports/faces/rho_ico6.npz`` (400 subjects,
icosphere subdiv 6, ``t_smooth=1e-4``). This script projects a handful of
those per-vertex fields to real-SH coefficients at the cross-field solver's
own bandwidth ``L`` -- the same ``real_sh_basis`` + mass-weighted
``sh_project`` that :class:`~shapegen.models.basis.RhoBasis` uses internally,
just without its whitening (the solver wants raw ρ̂, not standardised codes).

**The pipeline, end to end.**

1. §P8.3/§P8.4 (:mod:`shapegen.quad.field`) -- build the connection from ρ̂,
   anneal the spin-4 field, read off cones.
2. §P4-lite (:mod:`shapegen.quad.cones`) -- pole choice + stereographic
   projection to a flat chart.
3. §P5 (:mod:`shapegen.quad.periods`) -- the period/lattice solve: a cell
   size and a small perturbation of the cones so every loop's developing-map
   period lands on the lattice.
4. §P6 (:mod:`shapegen.quad.grid`) -- trace the grid over the whole sphere
   in two charts and assemble it from a rotation system into a closed mesh.
5. §P7 (:mod:`shapegen.quad.lift`) -- ``--lift route2`` (the default)
   integrates ``df~ = lam-bar df0 lam`` along the traced edges with the §P9
   spinor the connection was already built from, and needs no triangle mesh
   at all. ``--lift route1`` interpolates :func:`shapegen.codec.decode`'s positions at the traced
   directions -- 4.3x slower, kept as the reference and an escape hatch.
   :func:`shapegen.quad.lift.lift_route2` has the 10-head comparison.

Two figures per sample: the cross-field figure (as before -- lon/lat + sphere,
linear start vs. GL, with the P5-optimized cones overlaid), and a new quad-mesh
figure showing the actual extracted mesh in ℝ³ (shaded faces, back faces
culled, no wireframe -- see :func:`plot_quad_mesh_3d`), from two viewing
angles, with the Euler characteristic, coverage and isotropy in the title.

Usage::

    python -m scripts.quad_demo
    python -m scripts.quad_demo --cache reports/faces/rho_ico6.npz -n 4 -L 32
    python -m scripts.quad_demo --synthetic -n 3   # no cache needed

``--faces N`` sets the mesh density. Without it the cell size is whatever
rung of the period solve's stock sweep happens to solve first, which on three
FaceScape heads at L=32 -- same bandwidth, same 8 cones -- was 279, 553 and
8644 faces. With ``--faces 20000`` the same three came out at 20037, 20063
and 20043.

.. note::
   **Audit, 2026-09-21.** Those two triples predate the anneal-schedule fix
   (:func:`shapegen.quad.field.epsilon_ceiling`), and the *specific* heads
   they were taken on depend on ``--seed``, so they are not reproducible as
   literal numbers. Re-measured on three heads drawn at seed 0 (subjects
   #254/#204/#338) at L=32, the substance survives on both schedules: the
   stock sweep lands at **1112 / 507 / 590** faces under the shipped
   schedule and **2137 / 540 / 292** under the old one -- a 2-7x span from
   the same bandwidth and the same 8 cones, which is what the sentence is
   about -- while ``--faces 20000`` gives **20063 / 19987 / 20048** and
   **19935 / 20028 / 19942**, i.e. within 0.4% of target every time. All six
   traced to Euler characteristic 2 and coverage 1.000000.

   L=32 is also where the old schedule was *not* collapsed: the ceiling is
   5.95-6.03 pi/L on these heads, so only the leading 6.0 grazed it.
   ``area_distortion`` came out 0.0593 / 0.0494 / 0.0501 (old) against
   0.0552 / 0.0510 / 0.0521 (shipped) -- a wash, in both directions. The
   demo was never broken at this bandwidth; the L<=16 tests were.

Runtime, one face at L=32 with ``--faces 20000``: the §P9 spinor solve
(15-32s; it feeds both the connection and the lift), the Ginzburg-Landau
anneal 4.5s, the trace 4.4s and the route 2 lift 4.7s. ``--lift route1``
replaces that last with ~20s of ``codec.decode``.

That was 128s before the ladder-based Wigner-d landed (PLAN.md §P8.1):
``real_sh_basis`` went 43.9s -> 1.2s on the ico6 sphere and the fine-grid
``eval_basis`` 9.5s -> 0.53s, and the decode picked up the same win
internally (62.8s -> 20.4s).
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib               # noqa: E402
matplotlib.use("Agg")           # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

from shapegen.eval.metrics import nose_tip                   # noqa: E402
from shapegen.geom.encode import real_sh_basis, sh_project  # noqa: E402
from shapegen.geom.operators import vertex_areas            # noqa: E402
from shapegen.quad import cones as qc                        # noqa: E402
from shapegen.quad import grid as qg                          # noqa: E402
from shapegen.quad import field as qf                        # noqa: E402
from shapegen.quad import lift as ql                          # noqa: E402
from shapegen.quad import periods as qp                       # noqa: E402
from shapegen.spectral import swsh as sw                     # noqa: E402


# --------------------------------------------------------------------------- #
# rho fields -> rho-hat, and the field solve                                  #
# --------------------------------------------------------------------------- #

def project_rho(sphere: np.ndarray, faces: np.ndarray, rho_field: np.ndarray,
                 L: int, Y: np.ndarray | None = None) -> np.ndarray:
    """Per-vertex ρ (on ``sphere``) -> raw real-SH coefficients at bandwidth
    ``L``. Matches ``RhoBasis._project`` exactly, minus the whitening -- the
    solver wants ρ̂ itself, not a standardised training code.

    ``real_sh_basis(sphere, L)`` is the expensive part here -- ``scipy``'s
    ``sph_harm_y`` over every ``(l, m)`` on all 40962 vertices of the
    ``rho_ico6.npz`` sphere, ~45s at L=32 -- and it depends only on
    ``(sphere, L)``, not on which face's ``rho_field`` is being projected.
    Pass a precomputed ``Y`` (e.g. from :func:`load_face_samples`'s own
    loop) when projecting more than one field on the same sphere, or this
    silently repeats that ~45s for every face.
    """
    if Y is None:
        Y = real_sh_basis(sphere, L)
    mass = vertex_areas(sphere, faces)
    return sh_project(rho_field, Y, mass=mass)


def solve_field(rho_hat: np.ndarray, L: int, S: int, levels):
    """rho-hat -> (psi_linear, psi_GL, modes, cones_linear, cones_GL, spinor).

    ``spinor`` is the §P9 :class:`DiracSpectral` the curvature potential was
    built from, returned so the route 2 lift can reuse it rather than pay
    the solve twice.
    """
    spinor = qf.solve_potential_spinor(rho_hat, L)
    A = qf.connection_field(qf.curvature_potential(rho_hat, L, spinor=spinor))
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L, levels=levels)
    # A coarser fine-grid padding than the library default (over=7.0) -- still
    # plenty to resolve 8 well-separated cones, and much cheaper at L=32.
    cones0 = qf.cones_from_field(psi0, S, L, over=4.0)
    conesG = qf.cones_from_field(psiG, S, L, over=4.0)
    return psi0, psiG, modes, cones0, conesG, spinor


def optimize_cones(cones, n_samples: int = 4000, seed: int = 0) -> dict:
    """§P4-lite (pole + stereographic projection, :mod:`shapegen.quad.cones`)
    + §P5 (the period/lattice solve, :mod:`shapegen.quad.periods`) on a set of
    3D cones -- carries the solved chart back onto the sphere so it can be
    drawn alongside the raw GL cones.

    Returns a dict: ``cones_3d`` (``[(position, index), ...]`` at the solved
    positions, or ``None`` if nothing in the cell-size sweep solved),
    ``chart`` (:func:`~shapegen.quad.cones.project_cones`'s output), and
    ``solve`` (:func:`~shapegen.quad.periods.solve_lattice`'s output, with
    its own ``cell``/``cap``/``sweep`` diagnostics).
    """
    chart = qc.project_cones(cones, n_samples=n_samples, seed=seed)
    solve = qp.solve_lattice(chart["z"], chart["idx"])
    cones_3d = None
    if solve["cones"] is not None:
        P_solved = qc.chart_to_sphere(solve["cones"], chart["Rm"])
        cones_3d = list(zip(P_solved, chart["idx"]))
    return {"cones_3d": cones_3d, "chart": chart, "solve": solve}


def cell_ladder(s_max: float, ratio: float = 0.92, n: int = 30):
    """Cell sizes to hand :func:`~shapegen.quad.periods.solve_lattice`,
    descending from ``s_max``.

    It returns the *first* entry that solves, so a descending ladder makes
    that "the largest cell no coarser than what was asked for". Descending is
    also the direction that works: the cone motion a solve needs scales with
    ``s`` (the rounding residual is at most half a cell either way), so
    smaller cells solve more easily -- which is why the stock sweep found
    0.140 for one head and 0.025 for another. ``ratio=0.92`` over 30 rungs
    spans a factor of 8 in cell, i.e. ~65x in face count.
    """
    return tuple(s_max * ratio ** k for k in range(n))


def build_quad_mesh(gl_cones, rho_hat, n_samples: int = 4000, seed: int = 0,
                    max_vertices: int = 200000,
                    sublattice: str = "Z[i]", target_faces: int | None = None,
                    lift: str = "route2", spinor=None, L: int | None = None) -> dict:
    """§P4-lite -> §P5 -> §P6 -> §P7: GL cones + ρ̂ -> an actual quad mesh in
    ℝ³.

    §P6 is :func:`shapegen.quad.grid.quad_grid`, which traces the grid over
    the whole sphere in two charts and assembles it from a rotation system.
    That removes everything this function used to have to work around --
    there is no ``nsub`` to raise, no merge tolerance to make adaptive, no
    closure-defect filter to drop quads with, and ``converged`` means what
    it says instead of being ``False`` by design. The mesh it returns is
    closed: ``euler == 2`` and ``coverage == 1.0``, both reported.

    ``sublattice="(1+i)Z[i]"`` asks §P5 for the tighter target that puts
    every cone on a grid vertex, giving a mesh of pure quads rather than one
    with a triangular face at each cone that landed on a cell centre. It
    solves at a smaller cell, so it costs roughly 4-8x the faces --
    :func:`shapegen.quad.periods.round_to_lattice` has the arithmetic.

    ``target_faces`` sets the mesh density, which is otherwise an accident.
    :func:`~shapegen.quad.periods.solve_lattice`'s stock sweep returns the
    largest cell that happens to solve, and on three FaceScape heads at
    L=32 -- same bandwidth, same 8 cones, same kind of object -- that was
    0.140, 0.100 and 0.025, i.e. 279, 553 and 8644 faces. 279 quads cannot
    carry a face. :func:`shapegen.quad.grid.cell_for_faces` converts a face
    count into a cell size up front (exactly, on the cube: predicted 96, 384,
    1536 against actual 96, 384, 1536), and that becomes the top of a
    descending ladder, so the result is the coarsest mesh that is still at
    least as fine as asked for.

    Returns a dict with ``ok`` (``False`` only if §P5 failed to solve at any
    cell size, or the trace hit ``max_vertices``), ``chart``, ``solve``,
    ``grid`` (the :class:`~shapegen.quad.grid.QuadGrid`, carrying ``euler``,
    ``coverage``, ``merge_residual`` and the rest), ``faces``, ``P3``
    (lifted 3D vertex positions, §P7), ``cones_3d``, and ``aniso``
    (per-face edge-length ratio on ``P3``).
    """
    chart = qc.project_cones(gl_cones, n_samples=n_samples, seed=seed)
    kw = {}
    if target_faces:
        kw["cell_sizes"] = cell_ladder(
            qg.cell_for_faces(chart["z"], chart["idx"], target_faces))
    solve = qp.solve_lattice(chart["z"], chart["idx"], sublattice=sublattice, **kw)
    if solve["cones"] is None:
        return {"ok": False, "reason": "P5 did not solve at any cell size",
                "chart": chart, "solve": solve, "cones_3d": None}

    cones_3d = list(zip(qc.chart_to_sphere(solve["cones"], chart["Rm"]), chart["idx"]))

    g = qg.quad_grid(solve["cones"], chart["idx"], solve["cell"],
                     max_vertices=max_vertices)
    if not g.converged:
        return {"ok": False, "reason": "P6 hit max_vertices before closing",
                "chart": chart, "solve": solve, "grid": g, "cones_3d": cones_3d}

    # quad_grid already returns unit sphere points (that is what having two
    # charts buys), so only the pole rotation has to be undone -- not the
    # stereographic projection ql.chart_to_directions would also undo.
    dirs = g.points @ chart["Rm"]
    if lift == "route2":
        if L is None:
            raise ValueError("lift='route2' needs L (the rho-hat bandwidth)")
        P3, lift_info = ql.lift_route2(dirs, g.edges, rho_hat, L,
                                       faces=g.faces, res=spinor)
    elif lift == "route1":
        P3, lift_info = ql.lift_route1(dirs, rho_hat)
    else:
        raise ValueError(f"unknown lift {lift!r}")
    aniso = qg.quad_isotropy(P3, g.faces)

    return {"ok": True, "chart": chart, "solve": solve, "grid": g,
            "faces": g.faces, "P3": P3, "lift_info": lift_info, "lift": lift,
            "aniso": aniso, "cones_3d": cones_3d}


# --------------------------------------------------------------------------- #
# Visualisation: both an unwrapped lon/lat map and the sphere itself          #
# --------------------------------------------------------------------------- #

def _equirect_grid(nth: int, nph: int):
    """Plain (θ, φ) grid, no closed seam -- for the flat lon/lat map, where
    ``imshow`` doesn't need φ=0 repeated at φ=2π."""
    th = (np.arange(nth) + 0.5) * np.pi / nth
    ph = np.arange(nph) * 2.0 * np.pi / nph
    return th, ph


def _sphere_grid(nth: int, nph: int):
    """A closed-seam (θ, φ) grid: φ repeats its first column at the end so the
    rendered surface has no visible seam at φ=0/2π."""
    th = (np.arange(nth) + 0.5) * np.pi / nth
    ph = np.arange(nph + 1) * 2.0 * np.pi / nph        # nph+1: closes the seam
    return th, ph


@functools.lru_cache(maxsize=16)
def _grid_basis(S: int, L: int, nth: int, nph: int, closed_seam: bool):
    """``(th, ph, Y)`` for an equirect or closed-seam grid, ``Y`` the
    ``eval_basis`` design matrix -- cached by ``(S, L, nth, nph, closed_seam)``.

    Without this, one call to :func:`render_sample` recomputed the *same*
    ``wigner_d``-based design matrix from scratch six times over (once for
    the shared-vmax pass, twice more for the two lon/lat panels, twice more
    for the two sphere panels) -- ``eval_basis`` has no way to know a
    scattered-point evaluation it's given matches one it already did, so
    that redundancy has to be caught by the caller. Caching here also means
    a multi-face run (the actual use case) only pays for this once, not once
    per face, exactly like ``shapegen.quad.field``'s own caches.
    """
    th, ph = _sphere_grid(nth, nph) if closed_seam else _equirect_grid(nth, nph)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    Y = sw.eval_basis(S, L, TH.ravel(), PH.ravel())
    return th, ph, Y


def _field_on_grid(vec: np.ndarray, S: int, L: int, nth: int, nph: int,
                   closed_seam: bool = False):
    th, ph, Y = _grid_basis(S, L, nth, nph, closed_seam)
    return th, ph, (Y @ vec).reshape(len(th), len(ph))


def _tangent_frame(TH, PH):
    """Standard spherical-coordinate orthonormal tangent frame ``(θ̂, φ̂)`` --
    the frame ``arg(ψ)`` is measured against."""
    theta_hat = np.stack([np.cos(TH) * np.cos(PH), np.cos(TH) * np.sin(PH),
                          -np.sin(TH)], axis=-1)
    phi_hat = np.stack([-np.sin(PH), np.cos(PH), np.zeros_like(PH)], axis=-1)
    return theta_hat, phi_hat


def _cone_lonlat(cones):
    """Cone (x,y,z) positions -> (lon, lat) in degrees for the flat map."""
    if not cones:
        return np.empty(0), np.empty(0), np.empty(0)
    P = np.array([p for p, _k in cones])
    k = np.array([kk for _p, kk in cones])
    lat = 90.0 - np.degrees(np.arccos(np.clip(P[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    return lon, lat, k


def plot_cross_field_2d(ax, vec: np.ndarray, S: int, L: int, cones, title: str,
                        vmax: float, cones_opt=None, nth: int = 90, nph: int = 180,
                        glyph_stride: int = 6, glyph_len: float = 6.0):
    """``|ψ|`` heatmap on an unwrapped equirectangular (lon, lat) map, with the
    same N-RoSy cross-glyph overlay as :func:`plot_cross_field_3d` but drawn in
    the flat plot frame -- a locally-flat approximation that distorts near the
    poles (most visible as fanning near lat=+-90), traded here for showing the
    whole sphere in one glance rather than one hemisphere at a time.

    ``cones_opt``, if given, are the §P5-optimized positions (same order/
    indices as ``cones``): drawn as small yellow diamonds so the (usually
    small) motion from the raw GL cones is visible directly.
    """
    th, ph, v = _field_on_grid(vec, S, L, nth, nph, closed_seam=False)
    lon, lat = np.degrees(ph), 90.0 - np.degrees(th)

    im = ax.imshow(np.abs(v), extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                    origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vmax)

    base_angle = np.angle(v) / S
    for i in range(0, nth, glyph_stride):
        for j in range(0, nph, glyph_stride):
            x0, y0 = lon[j], lat[i]
            a0 = base_angle[i, j]
            for br in range(S):
                a = a0 + br * (2.0 * np.pi / S)
                dx, dy = glyph_len * np.cos(a), glyph_len * np.sin(a)
                ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy],
                       color="cyan", lw=0.8)

    clon, clat, ck = _cone_lonlat(cones)
    labeled = False
    for mask, color in ((ck > 0, "lime"), (ck <= 0, "red")):
        if mask.any():
            ax.scatter(clon[mask], clat[mask], s=60, facecolor="none",
                      edgecolor=color, linewidth=1.6,
                      label=("GL cone" if cones_opt and not labeled else None))
            labeled = True

    if cones_opt:
        olon, olat, _ok = _cone_lonlat(cones_opt)
        ax.scatter(olon, olat, s=45, marker="D", facecolor="yellow",
                  edgecolor="black", linewidth=0.6, zorder=5,
                  label="P5-optimized")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    return im


_ELEV, _AZIM = 20.0, 35.0


def _view_direction(elev=_ELEV, azim=_AZIM):
    """Unit vector from the origin towards the camera, matplotlib's own
    ``view_init(elev, azim)`` convention -- used to cull glyphs/cones on the
    far hemisphere (see :func:`plot_cross_field_3d`)."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def plot_cross_field_3d(ax, vec: np.ndarray, S: int, L: int, cones, title: str,
                        vmax: float, cones_opt=None, nth: int = 48, nph: int = 96,
                        glyph_stride: int = 4, glyph_len: float = 0.06,
                        glyph_offset: float = 1.05):
    """``|ψ|`` as a color on the actual unit sphere, with a coarse N-RoSy
    cross-glyph overlay (S branches at angle ``arg(ψ)/S + k·2π/S`` in the
    local ``(θ̂, φ̂)`` tangent frame -- a genuine 3D tangent vector, not a flat
    lon/lat approximation) and the detected cones marked on top.

    ``cones_opt``, if given, are the §P5-optimized positions: drawn as small
    yellow diamonds alongside the raw GL cones (circles) so the (usually
    small) motion is visible directly.

    ``vmax`` is passed in (not computed per panel) so the linear-start and
    post-GL panels share one color scale -- the linear stage genuinely has a
    much smaller and spikier ``|ψ|`` than the annealed field almost
    everywhere, which is exactly the thing GL is for, and per-panel
    normalisation would hide it.

    Glyphs are drawn well outside the surface radius (``glyph_offset``, not
    ``~1``), and ``ax.computed_zorder`` is turned off with explicit
    ``zorder``s: by default matplotlib's 3D axes pick one blanket draw order
    for a whole ``Poly3DCollection`` (the surface) vs. each ``Line3D``/
    ``scatter``, rather than depth-testing per pixel, so glyphs on the near
    hemisphere render *behind* the surface regardless of their actual radius
    unless this is disabled. That fix is global, though, so it also stops the
    (opaque) sphere from hiding glyphs/cones on the *far* hemisphere -- those
    are culled explicitly instead, by the sign of each point's dot product
    with the camera direction.
    """
    ax.computed_zorder = False
    view = _view_direction()
    th, ph, v = _field_on_grid(vec, S, L, nth, nph, closed_seam=True)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    X, Y, Z = np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)

    colors = plt.cm.magma(plt.Normalize(vmin=0.0, vmax=vmax)(np.abs(v)))
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=1)

    theta_hat, phi_hat = _tangent_frame(TH, PH)
    base_angle = np.angle(v) / S
    for i in range(0, nth, glyph_stride):
        for j in range(0, nph, glyph_stride):     # nph, not nph+1: skip the
            pt = np.array([X[i, j], Y[i, j], Z[i, j]])  # duplicate seam
            if pt @ view <= 0.0:                  # far hemisphere: skip
                continue
            p0 = glyph_offset * pt
            a0 = base_angle[i, j]
            for br in range(S):
                a = a0 + br * (2.0 * np.pi / S)
                d = np.cos(a) * theta_hat[i, j] + np.sin(a) * phi_hat[i, j]
                seg = np.stack([p0 - glyph_len * d, p0 + glyph_len * d])
                # black-then-white outline so a branch reads against both the
                # colormap's dark low end and its pale high end
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="black", lw=2.2,
                       zorder=10)
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="cyan", lw=0.9,
                       zorder=11)

    if cones:
        Praw = np.array([p for p, _k in cones])
        k = np.array([kk for _p, kk in cones])
        front = Praw @ view > 0.0
        P = glyph_offset * Praw
        for mask, color in ((k > 0, "lime"), (k <= 0, "red")):
            m = mask & front
            if m.any():
                ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=55,
                          color=color, edgecolor="black", linewidth=0.8,
                          depthshade=False, zorder=12)

    if cones_opt:
        Popt = np.array([p for p, _k in cones_opt])
        front_opt = Popt @ view > 0.0
        Po = glyph_offset * Popt
        if front_opt.any():
            ax.scatter(Po[front_opt, 0], Po[front_opt, 1], Po[front_opt, 2],
                      s=35, marker="D", facecolor="yellow", edgecolor="black",
                      linewidth=0.6, depthshade=False, zorder=13)

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    lim = glyph_offset + glyph_len
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_axis_off()
    ax.view_init(elev=_ELEV, azim=_AZIM)


# --------------------------------------------------------------------------- #
# The extracted quad mesh itself (§P6/§P7)                                    #
# --------------------------------------------------------------------------- #

def _view_towards(direction: np.ndarray):
    """``(elev, azim)`` for a camera on the far side of ``direction`` from
    the origin, looking back along it -- i.e. the view a person standing in
    that direction and facing the origin would see.

    ``direction`` here is :func:`shapegen.eval.metrics.nose_tip`'s output on
    the lifted head (not this script's own earlier "densest
    neighbourhood" search generalizing it, which didn't survive facial
    asymmetry -- see git history). ``nose_tip`` is an already-validated
    convention in this codebase: for FaceScape's registered topology, z is
    depth and the nose is the *dense* extreme point, not the occiput's sparse
    one. That direction carries over unchanged to any lift of the same
    ``rho_hat`` -- confirmed directly on two subjects with a hillshaded
    render: looking back along the nose direction shows a genuine front (eyes,
    nose bridge, mouth, both ears). An equatorial sweep (fixed elevation,
    varying azimuth around a *guessed* vertical axis) never finds the front at
    all, only rotates through side profiles with the nose foreshortened at
    the top of frame -- easy to misread one of those as "the front" (a first
    pass here settled on ``elev=0, azim=270`` this way and was wrong: that's
    the underside of the jaw, not the face).
    """
    d = direction / np.linalg.norm(direction)
    elev = np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0)))
    azim = np.degrees(np.arctan2(d[1], d[0]))
    return elev, azim


def plot_quad_mesh_3d(ax, P3: np.ndarray, quads: np.ndarray, cones_3d,
                      title: str, elev: float = 20.0, azim: float = 35.0,
                      wireframe: bool = False,
                      cull_backfaces: bool = True):
    """The actual extracted-and-lifted quad mesh: shaded faces, cones marked.
    ``P3`` is re-centered/scaled for display only (the lift itself, in
    ``quad_demo.py``'s caller, is untouched).

    ``wireframe`` overlays every quad edge. It is off by default: at the
    densities ``--faces`` now reaches, one line per cell is finer than the
    output raster, so the edges stop reading as a grid and just darken the
    whole surface into a flat silhouette -- which hides exactly the relief
    the shading is there to show. Turn it on for coarse meshes, where seeing
    the cells is the point.

    ``cull_backfaces`` drops faces pointing away from the camera before
    handing them to matplotlib. Since :mod:`shapegen.quad.grid` started
    returning a *closed* mesh, the far side is always there, and
    ``Poly3DCollection`` resolves overlap by painter's algorithm on each
    face's average depth -- which a back face can win against a front one
    wherever the surface folds towards the viewer (the nose, the ears).
    Culling removes the question instead of ordering it: on a closed
    orientable surface a face whose outward normal points away simply cannot
    be visible.

    Orthographic projection (not matplotlib's default perspective one): a
    perspective camera stretches near geometry and compresses far geometry,
    which is *extra*, purely-rendering unevenness on top of the mesh's own
    genuine cell-size variation (conformal maps don't preserve area --
    PLAN.md §10 measures this directly, "26x interdecile area spread is
    inherent" -- so the grid lines were never going to be perfectly even,
    but they shouldn't be *more* uneven than that on top).

    Faces are Lambertian-shaded by their own normal, not filled with one flat
    colour: a face has essentially no silhouette profile viewed head-on
    (§P7's "front" is a near-polar view straight down the nose), so without
    per-face relief cues -- the same reason ``plot_trisurf(shade=True)`` uses
    them -- the eyes/nose/mouth read as a flat cutout instead of a face.
    """
    ax.set_proj_type("ortho")
    ax.computed_zorder = False
    center = P3.mean(axis=0)
    scale = np.linalg.norm(P3 - center, axis=1).max()
    scale = scale if scale > 0 else 1.0
    Pn = (P3 - center) / scale

    if len(quads):
        faces = [Pn[q] for q in quads]
        light = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                          np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                          np.sin(np.radians(elev))])
        # faces are variable-length loops (quads, plus a triangle at any cone
        # that landed on a cell centre), so the normal comes from Newell's
        # formula rather than from a fixed pair of diagonals
        normals = np.array([np.cross(f - np.roll(f, -1, axis=0),
                                     f + np.roll(f, -1, axis=0)).sum(axis=0)
                            for f in faces])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

        # quad_grid orients every face counter-clockwise seen from outside the
        # sphere, and the lift preserves that, so Newell's normal is already
        # outward -- but a global flip is cheap insurance and costs one dot
        # product per face. The centroid test is only used to pick the sign
        # for the whole mesh, never per face, so a concave nose or ear cannot
        # flip its own faces.
        radial = np.array([f.mean(axis=0) for f in faces])
        if float(np.sum(np.einsum("ij,ij->i", normals, radial))) < 0.0:
            normals = -normals

        view = _view_direction(elev, azim)
        keep = (normals @ view > 0.0) if cull_backfaces else np.ones(len(faces), bool)
        faces = [f for f, k in zip(faces, keep) if k]
        shade = np.clip(normals[keep] @ light, 0.25, 1.0)
        base = np.array([0xa8, 0xc8, 0xe8]) / 255.0
        colors = np.clip(base[None, :] * shade[:, None], 0.0, 1.0)
        pc = Poly3DCollection(faces, facecolor=colors, edgecolor="none",
                              zorder=1)
        ax.add_collection3d(pc)
        if wireframe:
            segs = []
            for f in faces:
                for k in range(len(f)):
                    segs.append([f[k], f[(k + 1) % len(f)]])
            lc = Line3DCollection(segs, colors="#204060", linewidths=0.35,
                                  zorder=2)
            ax.add_collection3d(lc)

    if cones_3d:
        Pc = np.array([(p - center) / scale for p, _k in cones_3d])
        k = np.array([kk for _p, kk in cones_3d])
        # cones on the far side have to be culled the same way the faces are:
        # `computed_zorder = False` means a scatter is drawn over the surface
        # regardless of depth, so without this the back of the head's cones
        # float in front of the face
        front = (Pc @ _view_direction(elev, azim) > 0.0) if cull_backfaces \
            else np.ones(len(Pc), bool)
        for mask, color in ((k > 0, "lime"), (k <= 0, "red")):
            m = mask & front
            if m.any():
                ax.scatter(Pc[m, 0], Pc[m, 1], Pc[m, 2], s=45,
                          color=color, edgecolor="black", linewidth=0.7,
                          depthshade=False, zorder=3)

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    lim = 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)


def render_quad_mesh(quad_mesh: dict, title_prefix: str, out_path):
    """Two views (front + back) of the extracted quad mesh, with a title
    reporting §P6/§P7's own diagnostics: vertex and face counts, how many
    faces are not quads, cell size, Euler characteristic, coverage and
    median isotropy."""
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")

    if not quad_mesh.get("ok"):
        fig.suptitle(f"{title_prefix} -- quad mesh: {quad_mesh.get('reason')}",
                    fontsize=12)
        for ax in (ax0, ax1):
            ax.set_axis_off()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    g = quad_mesh["grid"]
    n_v = len(quad_mesh["P3"])
    deg = g.face_degrees
    aniso = quad_mesh["aniso"]
    med_an = float(np.median(aniso)) if len(aniso) else float("nan")
    fig.suptitle(f"{title_prefix} -- quad mesh: {n_v} vertices, "
                f"{int((deg == 4).sum())} quads + {int((deg != 4).sum())} "
                f"non-quad, cell={g.cell:.3f}, chi={g.euler}, "
                f"coverage={g.coverage:.4f}, median isotropy={med_an:.3f}",
                fontsize=11)

    # Only used to orient the camera: nose_tip needs just a point cloud.
    P3 = quad_mesh["P3"]
    Dc = P3 - P3.mean(axis=0)
    Dn = Dc / np.sqrt((Dc ** 2).sum(axis=1)).mean()
    elev, azim = _view_towards(nose_tip(Dn))

    plot_quad_mesh_3d(ax0, quad_mesh["P3"], quad_mesh["faces"],
                      quad_mesh["cones_3d"], "front", elev=elev, azim=azim)
    # the opposite pole, not azim+180 -- azim has no effect exactly at the
    # pole (elev close to +/-90), so "back" means flipping elev, not azim.
    plot_quad_mesh_3d(ax1, quad_mesh["P3"], quad_mesh["faces"],
                      quad_mesh["cones_3d"], "back", elev=-elev, azim=azim)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_sample(rho_hat, L, S, levels, title_prefix, out_path,
                  max_vertices: int = 200000,
                  target_faces: int | None = None, lift: str = "route2"):
    """2x2 figure: lon/lat map on top, the sphere itself below, linear start
    on the left and the GL result on the right -- all four panels share one
    color scale (see plot_cross_field_3d's docstring for why). The GL (right)
    panels additionally overlay the §P5-optimized cone positions.

    Also runs §P6/§P7 (build_quad_mesh) so the caller can render the actual
    extracted mesh separately (render_quad_mesh) without redoing §P4-lite/§P5.
    """
    psi0, psiG, _modes, cones0, conesG, spinor = solve_field(
        rho_hat, L, S, levels)
    quad_mesh = build_quad_mesh(conesG, rho_hat, max_vertices=max_vertices,
                               target_faces=target_faces,
                               lift=lift, spinor=spinor, L=L)
    cones_3d = quad_mesh.get("cones_3d")
    solve = quad_mesh.get("solve")

    _t, _p, v0 = _field_on_grid(psi0, S, L, 90, 180, closed_seam=False)
    _t, _p, vG = _field_on_grid(psiG, S, L, 90, 180, closed_seam=False)
    vmax = max(np.abs(v0).max(), np.abs(vG).max(), 1e-9)

    title0 = (f"linear start ({len(cones0)} cones, "
             f"sum {sum(k for _, k in cones0):+.2f})")
    if cones_3d is not None:
        cell = solve["cell"]
        motion = np.abs(solve["cones"] - quad_mesh["chart"]["z"]).max()
        titleG = (f"after GL + P5 ({len(conesG)} cones, "
                 f"sum {sum(k for _, k in conesG):+.2f}, "
                 f"cell={cell:.3f}, motion={motion:.4f})")
    else:
        titleG = (f"after GL ({len(conesG)} cones, "
                 f"sum {sum(k for _, k in conesG):+.2f}, P5: no solve)")

    fig = plt.figure(figsize=(13, 11))
    fig.suptitle(title_prefix, fontsize=13)
    ax_lon0 = fig.add_subplot(2, 2, 1)
    ax_lonG = fig.add_subplot(2, 2, 2)
    ax_sph0 = fig.add_subplot(2, 2, 3, projection="3d")
    ax_sphG = fig.add_subplot(2, 2, 4, projection="3d")

    im = plot_cross_field_2d(ax_lon0, psi0, S, L, cones0, title0, vmax=vmax)
    plot_cross_field_2d(ax_lonG, psiG, S, L, conesG, titleG, vmax=vmax,
                        cones_opt=cones_3d)
    plot_cross_field_3d(ax_sph0, psi0, S, L, cones0, title0, vmax=vmax)
    plot_cross_field_3d(ax_sphG, psiG, S, L, conesG, titleG, vmax=vmax,
                        cones_opt=cones_3d)
    if cones_3d is not None:
        ax_lonG.legend(loc="upper right", fontsize=7)

    fig.colorbar(im, ax=[ax_lon0, ax_lonG], label="|psi|", fraction=0.025,
                pad=0.02, location="right")
    fig.subplots_adjust(hspace=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cones0, conesG, quad_mesh


# --------------------------------------------------------------------------- #
# Sample sources                                                              #
# --------------------------------------------------------------------------- #

def load_face_samples(cache_path: Path, n: int, L: int, rng: np.random.Generator):
    """A few real face rho fields from an encode_faces.py cache, projected to
    rho-hat at bandwidth L. Returns [(label, rho_hat), ...]."""
    d = np.load(cache_path)
    sphere, faces, fields = d["sphere"], d["faces"], d["fields"]
    idx = rng.choice(len(fields), size=min(n, len(fields)), replace=False)
    Y = real_sh_basis(sphere, L)      # built once, shared across every face
    out = []
    for i in idx:
        rho_hat = project_rho(sphere, faces, fields[i], L, Y=Y)
        out.append((f"face #{i}", rho_hat))
    return out


def synthetic_samples(n: int, L: int):
    """No cache needed: F1-style power-law fields at a few spectra, reusing
    the fixtures already validated for P0/P1."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from quad_fixtures import bundle_style_coeffs  # noqa: E402
    from shapegen.geom.encode import n_coeffs        # noqa: E402

    powers = [-2.0, -1.0, -0.5][:max(n, 1)]
    out = []
    for power in powers:
        coeffs = bundle_style_coeffs(L=32, power=power, seed=1)[:n_coeffs(L)]
        out.append((f"synthetic ell^{power:g}", coeffs))
    return out


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="reports/faces/rho_ico6.npz",
                    help="npz from scripts/encode_faces.py")
    ap.add_argument("--synthetic", action="store_true",
                    help="use F1-style synthetic rho fields instead of --cache")
    ap.add_argument("-n", type=int, default=3, help="number of sample faces")
    ap.add_argument("-L", type=int, default=32,
                    help="cross-field bandwidth (the field lives at this "
                         "resolution regardless of what L the cache's rho was "
                         "encoded at)")
    ap.add_argument("-S", type=int, default=4, help="cross-field spin (4 = a "
                    "genuine 4-fold cross)")
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="GL epsilon schedule, in units of the field's own "
                         "resolution pi/L (default: field.anneal_schedule(), "
                         "i.e. DEFAULT_LEVELS clamped to the epsilon ceiling)")
    ap.add_argument("--faces", type=int, default=20000,
                    help="target face count; sets the cell size up front "
                         "instead of taking whatever the period solve's stock "
                         "sweep happens to land on (0 = stock sweep)")
    ap.add_argument("--lift", choices=("route2", "route1"), default="route2",
                    help="PLAN.md P7 lift: route 2 (default) integrates "
                         "df~ = lam-bar df0 lam along the traced edges with "
                         "the P9 spinor, mesh-free, ~5s at L=32; route 1 "
                         "interpolates codec.decode's positions, ~20s, kept "
                         "as the reference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/quad_field")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.synthetic or not Path(a.cache).exists():
        if not a.synthetic:
            print(f"no cache at {a.cache}; falling back to --synthetic")
        samples = synthetic_samples(a.n, a.L)
    else:
        samples = load_face_samples(Path(a.cache), a.n, a.L, rng)

    print(f"{'sample':<20} {'linear':>7} {'GL':>4} {'idx sum':>8} "
          f"{'P5 cell':>8} {'P6 verts':>9} {'faces':>6} {'chi':>5} "
          f"{'coverage':>9} {'isotropy':>11}")
    for label, rho_hat in samples:
        safe = label.replace(" ", "_").replace("#", "").replace("^", "")
        out_path = out_dir / f"{safe}.png"
        mesh_path = out_dir / f"{safe}_quadmesh.png"
        cones0, conesG, quad_mesh = render_sample(rho_hat, a.L, a.S,
                                                  tuple(a.levels) if a.levels
                                                  else None, label, out_path,
                                                  target_faces=a.faces or None,
                                                  lift=a.lift)
        render_quad_mesh(quad_mesh, label, mesh_path)

        solve = quad_mesh.get("solve")
        cell_s = f"{solve['cell']:.3f}" if solve and solve["cell"] else "no solve"
        if quad_mesh.get("ok"):
            g = quad_mesh["grid"]
            n_v = len(quad_mesh["P3"])
            an = quad_mesh["aniso"]
            an_s = f"{np.median(an):.3f}" if len(an) else "--"
            print(f"{label:<20} {len(cones0):>7d} {len(conesG):>4d} "
                  f"{sum(k for _, k in conesG):>8.2f} {cell_s:>8} {n_v:>9d} "
                  f"{len(g.faces):>6d} {g.euler:>5d} {g.coverage:>9.4f} "
                  f"{an_s:>11}   -> {out_path}, {mesh_path}")
        else:
            print(f"{label:<20} {len(cones0):>7d} {len(conesG):>4d} "
                  f"{sum(k for _, k in conesG):>8.2f} {cell_s:>8} "
                  f"{'--':>9} {'--':>6} {'--':>5} {'--':>9} {'--':>11}"
                  f"   ({quad_mesh.get('reason')})   -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())

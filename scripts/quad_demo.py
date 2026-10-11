"""Render the quad-mesh pipeline on real face ρ fields.

This script only loads data, renders and parses the CLI; the pipeline itself
(:func:`~shapegen.quad.pipeline.solve_field`,
:func:`~shapegen.quad.pipeline.build_quad_mesh`) lives in
:mod:`shapegen.quad.pipeline`.

**Input.** ``--cache`` is an ``scripts/encode_faces.py`` cache
(default ``reports/faces/rho_ico6.npz``: 400 subjects, per-vertex ρ on a
shared icosphere-6 base sphere, ``t_smooth=1e-4``). A few fields are
projected to raw real-SH ρ̂ at bandwidth ``L`` (:func:`project_rho`: the
``RhoBasis`` projection without its whitening). ``--synthetic``, or a
missing cache, uses power-law fixture fields instead.

**Output**, per sample, in ``--out``:

* ``<label>.png`` -- the cross field: lon/lat map and sphere, linear start
  vs. Ginzburg-Landau, with raw and period-solved cones overlaid;
* ``<label>_quadmesh.png`` -- the lifted quad mesh in ℝ³, front and back
  views (shaded, back faces culled; see :func:`plot_quad_mesh_3d`), with
  Euler characteristic, coverage and median isotropy in the title;

plus one summary row per sample on stdout.

Usage::

    python -m scripts.quad_demo
    python -m scripts.quad_demo --cache reports/faces/rho_ico6.npz -n 4 -L 32
    python -m scripts.quad_demo --synthetic -n 3   # no cache needed

``--cell S`` sets the grid cell size (chart units); otherwise ``--faces N``
(default 20000) is converted to the cell giving about that many faces. The
period solve runs at that cell only, so a sample can fail to solve ("no
solve" in the table); retry it with a smaller cell / more faces. At 20000,
results land within 0.4% of target with Euler characteristic 2 and coverage
1.000000.

Runtime, one face at L=32 with ``--faces 20000``: spinor solve ~1.4 s (shared
by the connection and the lift), GL anneal ~4.5 s, trace ~4.4 s, spectral
lift ~2.5 s.
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
from shapegen.quad.pipeline import build_quad_mesh, solve_field  # noqa: E402
from shapegen.spectral import swsh as sw                     # noqa: E402


# --------------------------------------------------------------------------- #
# rho fields -> rho-hat, and the field solve                                  #
# --------------------------------------------------------------------------- #

def project_rho(sphere: np.ndarray, faces: np.ndarray, rho_field: np.ndarray,
                 L: int, Y: np.ndarray | None = None) -> np.ndarray:
    """Per-vertex ρ (on ``sphere``) -> raw real-SH coefficients at bandwidth
    ``L``. Matches ``RhoBasis._project`` exactly, minus the whitening -- the
    solver wants ρ̂ itself, not a standardised training code.

    ``real_sh_basis(sphere, L)`` depends only on ``(sphere, L)`` (~0.15 s at
    L=32 on the 40962-vertex ico6 sphere); pass a precomputed ``Y`` when
    projecting several fields on the same sphere, as
    :func:`load_face_samples` does.
    """
    if Y is None:
        Y = real_sh_basis(sphere, L)
    mass = vertex_areas(sphere, faces)
    return sh_project(rho_field, Y, mass=mass)


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
    ``eval_basis`` design matrix -- cached by ``(S, L, nth, nph, closed_seam)``
    because every panel of every sample evaluates on the same few grids.
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

    ``cones_opt``, if given, are the period-solve positions (same order/
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

    ``cones_opt``, if given, are the period-solve positions: drawn as small
    yellow diamonds alongside the raw GL cones (circles) so the (usually
    small) motion is visible directly.

    ``vmax`` is passed in so the linear-start and post-GL panels share one
    color scale; per-panel normalisation would hide how much smaller and
    spikier the linear ``|ψ|`` is.

    Trap: matplotlib's 3D axes order a whole ``Poly3DCollection`` against
    each line/scatter as a block rather than depth-testing, so near-side
    glyphs render *behind* the sphere. Hence ``computed_zorder = False`` with
    explicit ``zorder``s, glyphs at radius ``glyph_offset``, and far-side
    glyphs/cones culled by the sign of their dot product with the camera
    direction.
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
# The extracted quad mesh itself                                              #
# --------------------------------------------------------------------------- #

def _view_towards(direction: np.ndarray):
    """``(elev, azim)`` for a camera on the far side of ``direction`` from
    the origin, looking back along it -- i.e. the view a person standing in
    that direction and facing the origin would see.

    ``direction`` is :func:`shapegen.eval.metrics.nose_tip` on the lifted
    head; looking back along it shows the face front (eyes, nose, mouth, both
    ears). Trap: an equatorial sweep around a guessed vertical axis never
    finds the front, only side profiles -- ``elev=0, azim=270`` looks
    plausible and is the underside of the jaw.
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
    ``P3`` is re-centered/scaled for display only.

    ``wireframe`` overlays every edge; off by default because at ~20k faces
    one line per cell is finer than the raster and just darkens the surface.
    Use it for coarse meshes.

    ``cull_backfaces`` drops faces whose outward normal points away from the
    camera. Trap: the mesh is closed, and ``Poly3DCollection``'s
    painter's-algorithm ordering lets back faces win against front ones
    where the surface folds towards the viewer (nose, ears).

    Orthographic projection, so perspective adds no unevenness on top of the
    mesh's inherent conformal cell-size variation. Faces are
    Lambertian-shaded by their own normal: head-on, a face has no silhouette
    and reads as a flat cutout without per-face relief.
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

        # quad_grid orients faces counter-clockwise from outside, so Newell's
        # normal should already be outward; one global sign check as
        # insurance -- never per face, so a concave nose cannot flip itself.
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
    reporting the trace's and lift's own diagnostics: vertex and face counts, how many
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
                  max_vertices: int = 200000, cell: float | None = None,
                  target_faces: int | None = None):
    """Run the pipeline on one ``rho_hat`` and save the cross-field figure.

    2x2 figure: lon/lat map on top, sphere below, linear start left, GL right,
    one shared color scale; the GL panels also show the period-solve cones.
    Returns ``(cones_linear, cones_GL, quad_mesh)``, ``quad_mesh`` being
    :func:`~shapegen.quad.pipeline.build_quad_mesh`'s dict, for
    :func:`render_quad_mesh`.
    """
    psi0, psiG, _modes, cones0, conesG, spinor = solve_field(
        rho_hat, L, S, levels)
    quad_mesh = build_quad_mesh(conesG, rho_hat, max_vertices=max_vertices,
                               cell=cell, target_faces=target_faces,
                               spinor=spinor, L=L)
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
    """No cache needed: power-law ρ̂ fields (``ℓ^-2``, ``ℓ^-1``, ``ℓ^-0.5``)
    from ``tests/quad_fixtures.bundle_style_coeffs``, truncated to ``L``."""
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
                    help="target face count, converted to a cell size "
                         "(ignored if --cell is given)")
    ap.add_argument("--cell", type=float, default=None,
                    help="grid cell size in chart units; the period solve "
                         "runs at exactly this size")
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
                                                  cell=a.cell,
                                                  target_faces=None if a.cell
                                                  else a.faces)
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

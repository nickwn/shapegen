"""Mesh rendering shared by the figure scripts (``readme_figures``, ``quad_demo``).

A plain orthographic painter's algorithm on a 2-D ``PolyCollection``: back faces
culled, faces drawn far to near, Lambertian shading. Fast, and right for a
convex-ish closed surface. matplotlib's 3-D axes order a whole
``Poly3DCollection`` badly where a closed surface folds towards the viewer (nose,
ears), which is why this does its own sorting.
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import PolyCollection

BASE = np.array([0xc9, 0xcf, 0xd8]) / 255.0         # neutral clay
QUAD = np.array([0xa8, 0xc8, 0xe8]) / 255.0


def similarity(src: np.ndarray, dst: np.ndarray):
    """The similarity taking ``src`` onto ``dst`` (corresponding rows), as a
    function of points. Decodes are defined up to one, so this is display-only."""
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, s, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    scale = (s * np.diag(D)).sum() / (A ** 2).sum()
    return lambda P: scale * (P - ms) @ R + md


def view_frame(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera frame, rows screen-right, screen-up, towards-camera. ``forward``
    points from the object at the camera; ``up`` need not be orthogonal to it
    (a parallel one falls back to another axis)."""
    f = forward / np.linalg.norm(forward)
    u = up - (up @ f) * f
    if np.linalg.norm(u) < 1e-6:
        u = np.eye(3)[np.argmin(np.abs(f))]
        u = u - (u @ f) * f
    u /= np.linalg.norm(u)
    return np.stack([np.cross(u, f), u, f])


def rotate_y(R: np.ndarray, deg: float) -> np.ndarray:
    """Turn the camera ``deg`` degrees about screen-up (a three-quarter view)."""
    a = np.radians(deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return Ry @ R


def render(ax, P: np.ndarray, faces, R: np.ndarray, colors=None, base=BASE,
           edges: float = 0.0, edgecolor="#2b3440", points=None):
    """Orthographic render of the polygon mesh ``(P, faces)`` in camera frame
    ``R`` (see :func:`view_frame`). ``faces`` is an ``(m, k)`` array or a list
    of variable-length loops. ``colors`` (per face, RGB) replaces ``base``;
    shading multiplies either. ``points`` is an optional list of ``(xyz (n, 3),
    color)`` drawn as markers where they face the camera (e.g. cones)."""
    c0 = P.mean(0)
    Q = (P - c0) @ R.T
    s = np.abs(Q[:, :2]).max()
    Q /= s
    polys = [Q[np.asarray(f)] for f in faces]
    n = np.array([np.cross(p - np.roll(p, -1, 0), p + np.roll(p, -1, 0)).sum(0)
                  for p in polys])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    # one global orientation check (outward normals), never per face, so a
    # concave nose cannot flip itself
    cen = np.array([p.mean(0) for p in polys])
    if np.einsum("ij,ij->i", n, cen - Q.mean(0)).sum() < 0:
        n = -n
    light = np.array([-0.35, 0.45, 0.82])
    light /= np.linalg.norm(light)
    shade = 0.22 + 0.78 * np.clip(n @ light, 0, 1) + 0.12 * n[:, 2] ** 8
    base_c = np.broadcast_to(base, (len(polys), 3)) if colors is None else colors
    col = np.clip(base_c * shade[:, None], 0, 1)
    idx = np.flatnonzero(n[:, 2] > 0)
    idx = idx[np.argsort(cen[idx, 2])]              # far to near
    ax.add_collection(PolyCollection(
        [polys[i][:, :2] for i in idx], facecolors=col[idx],
        edgecolors=edgecolor if edges else "face",
        linewidths=edges if edges else 0.15, antialiased=True))
    for xyz, color in points or ():
        q = (np.atleast_2d(xyz) - c0) @ R.T / s
        q = q[q[:, 2] > 0]
        ax.scatter(q[:, 0], q[:, 1], s=40, color=color, edgecolor="black",
                   linewidth=0.7, zorder=3)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

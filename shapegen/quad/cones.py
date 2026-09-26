"""Pole choice and stereographic projection for cones (PLAN.md §P4, ported
from ``s4_cones.py`` — reduced scope, see below).

**Scope reduction from the plan.** ``s4_cones.py``'s job was: take umbilics
(index +½ each, from mesh curvature) and split each into two valence-3 cones
(index +¼) before projecting. This port skips umbilics and splitting
entirely — cones come from :mod:`shapegen.quad.field`'s Ginzburg-Landau
anneal instead, which already produces the correct ±¼-index singularities
directly (``cones_from_field``'s winding count). What's left of P4 is just the
part that turns *some* set of indexed points on the sphere into a flat chart
for §P5's period solve: choose a pole far from every cone, rotate it to the
north pole, and stereographically project.
"""

from __future__ import annotations

import numpy as np

__all__ = ["choose_pole", "rotation_to_north", "stereo", "unstereo",
           "project_cones", "chart_to_sphere"]


def choose_pole(positions: np.ndarray, n_samples: int = 4000, seed: int = 0):
    """Pick a projection pole maximally far from every point in ``positions``
    (unit vectors, shape ``(n, 3)``) — ``s4_cones.py``'s search, ported
    exactly.

    Samples ``n_samples`` random unit directions ``v``; scores each by the
    *worse* of ``v``'s and ``-v``'s distance to its nearest point (cheap to
    get both from one draw, and using the worse of the two searches more of
    the sphere per sample without a real cost); keeps the best-scoring ``v``.

    Returns ``(pole, distance_radians)``.
    """
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_samples):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        d = min(np.arccos(np.clip(positions @ v, -1.0, 1.0)).min(),
               np.arccos(np.clip(positions @ (-v), -1.0, 1.0)).min())
        if best is None or d > best[0]:
            best = (d, v)
    return best[1], best[0]


def rotation_to_north(pole: np.ndarray) -> np.ndarray:
    """Rotation matrix ``Rm`` with ``Rm @ pole == [0, 0, 1]`` (to ``1e-14``).

    ``s4_cones.py`` computes this Rodrigues expression three times: the first
    two are dead stores, and the second is an outright wrong formula left in
    place (it squares the *previous* ``Rm`` — ``I + K``, not the skew matrix
    ``K`` itself — so ``Rm @ Rm`` there is not ``K @ K``). This ports only the
    third (correct, and the one actually used downstream) expression::

        v = a x b,  c = a.b,  K = skew(v)
        Rm = I + K + K@K / (1 + c)

    for ``a = pole``, ``b = [0, 0, 1]`` — the standard closed form for the
    rotation taking one unit vector to another, singular only at ``a = -b``
    (not guarded here, matching the bundle: irrelevant in practice since
    ``choose_pole`` never returns a pole antipodal to ``[0,0,1]`` by
    construction).
    """
    a = np.asarray(pole, dtype=float)
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b)
    c = float(a @ b)
    K = np.array([[0.0, -v[2], v[1]],
                 [v[2], 0.0, -v[0]],
                 [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def stereo(X: np.ndarray) -> np.ndarray:
    """Stereographic projection ``(x, y, z) -> x+iy`` scaled by ``1/(1-z)``.

    The puncture (mapped to infinity) is at ``z = 1`` — i.e. at whatever got
    rotated to north, which is ``pole``, chosen to be far from every cone.
    """
    return (X[:, 0] + 1j * X[:, 1]) / (1.0 - X[:, 2])


def unstereo(z: np.ndarray) -> np.ndarray:
    """Inverse of :func:`stereo`: complex plane -> unit sphere."""
    r2 = np.abs(z) ** 2
    return np.stack([2.0 * z.real / (r2 + 1.0), 2.0 * z.imag / (r2 + 1.0),
                     (r2 - 1.0) / (r2 + 1.0)], axis=1)


def chart_to_sphere(z: np.ndarray, Rm: np.ndarray) -> np.ndarray:
    """Inverse of the projection inside :func:`project_cones`: flat chart
    positions -> unit-sphere positions. Undoes the rotation with ``Rm``'s
    transpose (``Rm`` is orthogonal, so ``Rm.T == Rm^-1``) -- the same
    relation :mod:`tests.test_quad_cones` cross-checks against the P0
    corpus's saved ``Rm.npy``/``cones_plane.npy``.

    Used after :func:`~shapegen.quad.periods.solve_lattice` perturbs the
    chart, to carry the solved cones back onto the sphere for display or for
    the next stage (§P6 extraction, once it exists).
    """
    return unstereo(z) @ Rm


def project_cones(cones, n_samples: int = 4000, seed: int = 0,
                  pole: np.ndarray | None = None) -> dict:
    """Cones (``[(position (3,), index), ...]``, from
    :func:`~shapegen.quad.field.cones_from_field`) -> a flat chart.

    Returns a dict: ``pole``, ``pole_dist`` (radians, the pole search's own
    score), ``Rm``, ``z`` (complex chart positions, cone order preserved),
    ``idx`` (indices, unchanged — no splitting happens here), and
    ``min_gap`` (minimum pairwise chart separation — §P5's cone-motion
    budget is ``0.25 x`` this).

    ``pole`` **freezes the chart**: pass a unit vector and :func:`choose_pole`
    is skipped entirely, so the projection is a deterministic, smooth function
    of the cone positions and costs nothing. Without it the pole comes from a
    4000-sample random search over the cones themselves (40 ms), and it
    genuinely hops as they move — 179 degrees between two neighbouring
    placements in one measured walk. Whether that matters depends on what is
    computed from the chart: :func:`~shapegen.quad.field.area_distortion` is
    chart-free and does not notice (agreement to 2e-15 across that same hop),
    but anything that *does* depend on the chart, and anything sweeping over
    cone positions at all, should freeze it. ``pole_dist`` is still reported,
    so a caller that froze the pole can check it has not drifted into a cone.
    """
    P = np.array([p for p, _k in cones])
    idx = np.array([k for _p, k in cones])
    if pole is None:
        pole, pole_dist = choose_pole(P, n_samples=n_samples, seed=seed)
    else:
        pole = np.asarray(pole, dtype=float)
        pole = pole / np.linalg.norm(pole)
        pole_dist = float(min(
            np.arccos(np.clip(P @ pole, -1.0, 1.0)).min(),
            np.arccos(np.clip(P @ (-pole), -1.0, 1.0)).min()))
    Rm = rotation_to_north(pole)
    z = stereo(P @ Rm.T)
    d = np.abs(z[:, None] - z[None, :])
    np.fill_diagonal(d, np.inf)
    min_gap = float(d.min())
    return {"pole": pole, "pole_dist": pole_dist, "Rm": Rm, "z": z,
            "idx": idx, "min_gap": min_gap}

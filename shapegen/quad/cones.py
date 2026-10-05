"""Cones on the sphere -> a flat chart for the period solve (ported from ``s4_cones.py``): choose a pole far from every cone, rotate it to
north, and stereographically project.

Cones come from :func:`shapegen.quad.field.cones_from_field` with their
indices already final; no umbilic detection or splitting happens here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["choose_pole", "rotation_to_north", "stereo", "unstereo",
           "project_cones", "chart_to_sphere"]


def choose_pole(positions: np.ndarray, n_samples: int = 4000, seed: int = 0):
    """Pick a projection pole maximally far from every point in ``positions``
    (unit vectors, shape ``(n, 3)``) by random search: each of ``n_samples``
    directions ``v`` is scored by the worse of ``v``'s and ``-v``'s angular
    distance to its nearest point.

    Returns ``(pole, distance_radians)``. Trap: the result can jump
    discontinuously (179 degrees measured) as the points move slightly; see
    :func:`project_cones`'s ``pole``.
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
    """Rotation matrix ``Rm`` with ``Rm @ pole == [0, 0, 1]`` (to ``1e-14``)::

        v = a x b,  c = a.b,  K = skew(v)
        Rm = I + K + K@K / (1 + c)

    for ``a = pole``, ``b = [0, 0, 1]``. Trap: singular at ``pole = [0, 0,
    -1]`` and not guarded.
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
    positions (complex) -> unit-sphere positions ``(n, 3)``, undoing the
    rotation with ``Rm.T``. Used to carry
    :func:`~shapegen.quad.periods.solve_lattice`'s solved cones back onto the
    sphere.
    """
    return unstereo(z) @ Rm


def project_cones(cones, n_samples: int = 4000, seed: int = 0,
                  pole: np.ndarray | None = None) -> dict:
    """Cones (``[(position (3,), index), ...]``, from
    :func:`~shapegen.quad.field.cones_from_field`) -> a flat chart.

    Returns a dict: ``pole``, ``pole_dist`` (radians, the pole search's own
    score), ``Rm``, ``z`` (complex chart positions, cone order preserved),
    ``idx`` (indices, unchanged — no splitting happens here), and
    ``min_gap`` (minimum pairwise chart separation — the period solve's cone-motion
    budget is ``0.25 x`` this).

    ``pole`` freezes the chart: :func:`choose_pole` (40 ms, and it hops as
    cones move) is skipped, making the projection a smooth deterministic
    function of the cones. Anything chart-dependent, and any sweep over cone
    positions, should freeze it. ``pole_dist`` is still reported so a caller
    can check a frozen pole has not drifted into a cone.
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

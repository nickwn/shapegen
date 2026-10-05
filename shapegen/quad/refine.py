"""Local optimisation of cone *positions*, starting from the Ginzburg-Landau
cones (:mod:`shapegen.quad.field`), minimising
:func:`~shapegen.quad.field.area_distortion`'s energy.

Cone count and indices are fixed; only positions move. (So this is not the
rejected "make cones emerge by relaxation" experiment --
topology stays with GL.) GL's placement is not a local minimum of the
energy: a 1-degree jitter already improves the median by 8%.

Implementation: the decoded surface is cone-independent and built once
(:func:`~shapegen.quad.field.decoded_surface`); the chart is frozen for the
whole run. The energy is chart-free (frozen and searched poles agree to
2e-15), so freezing buys cost (no 40 ms pole search per evaluation) and
determinism, not correctness.

Measured gain (paired: each cone set vs its own refinement), 468 comparisons
over L = 12-32, amplitudes 1-8, 12 starts each: 468/468 wins on the sampled
energy, median 1.169x, geometric mean 1.227x, range 1.021-1.881x; the gain
grows with amplitude.

Trap: the sampled gain overstates the mesh. On traced quad meshes (variance
of log face area) the gain is roughly a third of it -- 1.08-1.18x at
amplitude 2-4, ~1.0x at amplitude 1 and on the round sphere -- so quote the
traced number.
"""

from __future__ import annotations

import numpy as np

from . import cones as _qc
from . import grid as _qg
from . import field as _qf

__all__ = ["cone_frames", "objective", "refine_cones"]


def cone_frames(P: np.ndarray):
    """An orthonormal tangent frame ``(e1, e2)`` at each unit vector in ``P``.

    Cones move as ``p(u, v) = normalize(p + u e1 + v e2)`` in these fixed
    frames (no pole singularity, unlike spherical angles), so a box bound on
    ``(u, v)`` conservatively bounds angular motion (``|(u,v)| = tan(angle)``).
    """
    P = np.asarray(P, dtype=float)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(P), 1))
    flip = np.abs(P[:, 2]) > 0.9
    ref[flip] = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(P, ref)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(P, e1)
    return e1, e2


def objective(P0: np.ndarray, idx: np.ndarray, surface: dict,
              Rm: np.ndarray):
    """Build ``f(x) -> energy`` for a **frozen** chart ``Rm``.

    ``x`` is the flat ``(2n,)`` vector of tangent displacements in
    :func:`cone_frames`' frames at ``P0``. Everything that does not depend on
    the cones is computed once here: the decoded surface's area elements, the
    rotated sphere samples, their per-point chart and chart coordinate, and
    the part of ``log sigma`` the cones cannot touch.

    Writing ``a = -idx`` and ``lm = log|W| = Σ_k a_k log|z - C_k|``
    (:func:`~shapegen.quad.grid.form_log_abs`), the identity the whole thing
    rests on is that the cones enter ``log sigma`` **only** through ``lm``:

        log sigma = [log dA_surface - log dA_sphere - 2 log((1+|z|²)/2)] - 2 lm

    with the bracket constant. The flat weights ``w ∝ dA_sphere/conf`` move
    too, which is why this is not simply a weighted least squares in ``lm``.
    """
    P0 = np.asarray(P0, dtype=float)
    idx = np.asarray(idx, dtype=float)
    a = -idx
    e1, e2 = cone_frames(P0)

    S = surface["S"]
    Xr = S @ Rm.T
    c = _qg.chart_of(Xr)
    z = _qg.to_chart(Xr, c)
    az2 = np.abs(z) ** 2
    # log(dA_sphere / conf) with the cone part (exp(2 lm)) factored out
    log_flat_base = np.log(surface["dA_sphere"]) + 2.0 * np.log(
        (1.0 + az2) / 2.0)
    const = np.log(surface["dA_surface"]) - log_flat_base

    def positions(x):
        d = np.asarray(x, dtype=float).reshape(-1, 2)
        Q = P0 + d[:, :1] * e1 + d[:, 1:] * e2
        return Q / np.linalg.norm(Q, axis=1, keepdims=True)

    def f(x):
        C = _qc.stereo(positions(x) @ Rm.T)
        lm = _qg.form_log_abs(z, c, C, a)
        ls = const - 2.0 * lm
        logw = log_flat_base + 2.0 * lm
        w = np.exp(logw - logw.max())
        w /= w.sum()
        return float((w * (ls - (w * ls).sum()) ** 2).sum())

    f.positions = positions
    return f


def refine_cones(cones, rho_hat: np.ndarray, L: int, subdiv: int = 5,
                 surface: dict | None = None, pole: np.ndarray | None = None,
                 gap_keep: float = 0.5, maxiter: int = 300,
                 n_samples: int = 4000, seed: int = 0) -> dict:
    """Locally minimise :func:`~shapegen.quad.field.area_distortion`'s energy
    over cone positions, starting from ``cones``.

    Invariants, all of them enforced rather than hoped for:

    * **Index sum stays +2.** Indices are never touched — only positions move
      — so Poincare-Hopf holds identically; the returned energy is computed
      through ``area_distortion``, which raises if it ever did not.
    * **Cone count is fixed**, and cones cannot collide: each cone is
      box-constrained to move at most ``(1 - gap_keep)/2`` of the initial
      minimum pairwise angular separation, so that separation cannot fall
      below ``gap_keep`` times its starting value (two cones approach at most
      twice one cone's step). ``gap_keep=0.5`` mirrors
      :func:`~shapegen.quad.periods.solve_lattice`'s own ``0.25 * min_gap``
      motion budget downstream: whatever slack the period solve needed at the
      start, it still has half of it.
    * **The chart is frozen** at ``pole`` (default: the pole
      :func:`~shapegen.quad.cones.project_cones` chooses for the initial
      cones). ``energy0``/``energy`` use it; ``energy_rechart`` re-scores in a
      freshly chosen chart (equal to ``energy`` to ~1e-14).

    Returns ``{"cones", "energy0", "energy", "ratio", "energy_rechart",
    "motion_deg", "max_motion_deg", "min_sep0", "min_sep", "pole", "nit",
    "nfev"}`` with ``ratio = energy0 / energy`` (>1 is an improvement).
    """
    from scipy.optimize import minimize          # noqa: PLC0415

    P0 = np.array([p for p, _k in cones], dtype=float)
    idx = np.array([k for _p, k in cones], dtype=float)
    if abs(idx.sum() - 2.0) > 1e-9:
        raise ValueError(f"cone index sum is {idx.sum()}, not 2 -- refine "
                         "positions of a valid cone set, not an invalid one")

    if surface is None:
        surface = _qf.decoded_surface(rho_hat, subdiv)
    if pole is None:
        pole = _qc.project_cones(cones, n_samples=n_samples, seed=seed)["pole"]
    pole = np.asarray(pole, dtype=float)
    Rm = _qc.rotation_to_north(pole)

    ang = np.arccos(np.clip(P0 @ P0.T, -1.0, 1.0))
    np.fill_diagonal(ang, np.inf)
    min_sep0 = float(ang.min())
    step = 0.5 * (1.0 - gap_keep) * min_sep0 / np.sqrt(2.0)

    f = objective(P0, idx, surface, Rm)
    x0 = np.zeros(2 * len(P0))
    res = minimize(f, x0, method="L-BFGS-B",
                   bounds=[(-step, step)] * len(x0),
                   options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-12})

    P1 = f.positions(res.x)
    new = [(P1[i], float(idx[i])) for i in range(len(P1))]

    ang1 = np.arccos(np.clip(P1 @ P1.T, -1.0, 1.0))
    np.fill_diagonal(ang1, np.inf)
    motion = np.rad2deg(np.arccos(np.clip((P0 * P1).sum(axis=1), -1.0, 1.0)))

    e0 = _qf.area_distortion(cones, rho_hat, L, subdiv=subdiv,
                             surface=surface, pole=pole)["energy"]
    e1 = _qf.area_distortion(new, rho_hat, L, subdiv=subdiv,
                             surface=surface, pole=pole)["energy"]
    er = _qf.area_distortion(new, rho_hat, L, subdiv=subdiv, surface=surface,
                             n_samples=n_samples, seed=seed)["energy"]
    return {"cones": new, "energy0": e0, "energy": e1, "ratio": e0 / e1,
            "energy_rechart": er, "motion_deg": motion,
            "max_motion_deg": float(motion.max()), "min_sep0": min_sep0,
            "min_sep": float(ang1.min()), "pole": pole,
            "nit": int(res.nit), "nfev": int(res.nfev)}

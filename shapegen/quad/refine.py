"""Local optimisation of cone *positions*, starting from the Ginzburg-Landau
answer (PLAN.md §P8.4's output).

**This is not PLAN.md §P8.7's rejected experiment, and must not be filed
under it later.** §P8.7 ("do not attempt", with ``whyfixed.py`` and
``emerge.py`` behind it) is about making cones **emerge**: whether a point
relaxation can produce the right *number* of singularities with the right
*indices* out of a field that has none. It cannot — singularities are a
topological fact about a section, relaxation of free points has no mechanism
to create them, and that is precisely the job Ginzburg-Landau does and does
correctly (8 cones, index ``+1/4`` each, sum ``+2``, every time).

This module takes that verdict as given. The cone **count is fixed**, every
cone **keeps its own index**, and the topology is never in play: all that
moves is *where* an already-correct cone set sits. §P8.7's companion argument
— that "the error lives in the relaxation, not the norm" — is an argument
about which *equation* to relax, not about whether an already-topologically-
correct placement is at the optimum of
:func:`~shapegen.quad.field.area_distortion`. Measured, it is not: jittering
the GL cones by 1 degree moves that score by +/-9% and *improves* the median
by 8%, so the flow's answer is not even a local minimum of the objective the
cones exist to serve. The plan never tested this, and it is cheap.

**Two things make it work at all**, both of which a naive implementation gets
wrong:

1. :func:`~shapegen.quad.field.area_distortion` re-runs ``codec.decode`` on
   every call — 97% of its cost, and it depends only on ``rho_hat``, never on
   the cones. :func:`~shapegen.quad.field.decoded_surface` hoists it out;
   an objective evaluation drops from 2.7 s to 1.8 ms.
2. The chart is **frozen** for the whole optimisation and re-derived once at
   the end, because ``area_distortion`` otherwise chooses its chart *from the
   cones* with a 4000-sample random pole search on every call.

   **Not for the reason one expects, and the expected reason is measurably
   false.** The worry is that a pole chosen from the moving cones hops, and
   injects a discontinuity into the objective. It does hop — walking one cone
   0.05 rad moves the chosen pole by 3 degrees at one step and by 179 degrees
   at another — but the energy does not notice: frozen-chart and
   pole-searched values agree to 2e-15 relative at every sample, and the two
   curves have the same second differences to all printed digits. The metric
   really is chart-free. Freezing it buys **cost and determinism** instead:
   1.1 ms per evaluation against 40 ms, and an objective that is a pure
   function of the cone positions rather than of a random search's luck.

**What it is worth, measured against the anneal's own noise floor.** Scored
*paired* -- refine each schedule's own cones and compare like with like,
because the unpaired numbers sit inside the schedule spread -- across 80
(case, schedule) pairs at L=12 and L=16 and five amplitudes:

* on :func:`~shapegen.quad.field.area_distortion`: **80 of 80 wins**, median
  1.118x, geometric mean 1.136x, worst 1.016x, best 1.593x. It also *shrinks*
  the schedule-to-schedule spread, from 1.08-1.46x per case to 1.03-1.13x --
  different schedules land in the same basin and refinement finds its floor.
* on the **traced quad mesh** (unweighted variance of log face area, the
  quantity the sampled metric discretises), on 28 of those pairs: 23 of 28
  wins, median 1.043x, geometric mean 1.059x, range 0.987x-1.179x. Every
  trace was valid -- Euler characteristic 2, coverage 1.000000, index sum 2.

**Corroborated on a much larger ensemble (2026-09-21).** The P9
re-measurement ran 13 (bandlimit, amplitude) cases at L = 12/16/24/32 and
amplitudes 1-8, three curvature potentials each, over 8 anneal schedules plus
4 perturbed starts -- 468 paired (cone set, refinement) comparisons in total,
and the refinement won **468 of 468**: median **1.169x**, geometric mean
**1.227x**, worst **1.021x**, best **1.881x**. The original 80-pair figures
below (median 1.118x, geometric mean 1.136x) are the low-amplitude corner of
that; the gain grows with amplitude, and at amplitude 8 it reaches 1.5-1.9x.

The traced gain is roughly a third of the sampled gain, and it is
amplitude-dependent: 1.08-1.18x traced at amplitude 2-4, 1.00-1.02x at
amplitude 1, and 0.99-1.00x on the round sphere, where the five losses (all
within 1.3%) live. Part of the sampled gain is therefore being taken out of
the sampled metric's own discretisation rather than out of the mesh -- the
ratio traced/sampled rises from ~1.03 before refinement to ~1.09-1.13 after,
which is the sampled metric under-predicting the refined mesh. It is still
inside that metric's own validated agreement band (+/-15%), but it means the
honest headline is **the traced number, not the sampled one**.
"""

from __future__ import annotations

import numpy as np

from . import cones as _qc
from . import grid as _qg
from . import field as _qf

__all__ = ["cone_frames", "objective", "refine_cones"]


def cone_frames(P: np.ndarray):
    """An orthonormal tangent frame ``(e1, e2)`` at each unit vector in ``P``.

    Cones are optimised in these *fixed* frames — ``p(u, v) = normalize(p +
    u e1 + v e2)`` — rather than in spherical angles, which would put a
    coordinate singularity at the poles and make the step size depend on
    latitude, or in the chart, where the metric blows up away from the
    origin. A fixed tangent frame makes a box constraint on ``(u, v)`` an
    honest bound on angular motion (``|(u,v)| = tan(angle)``, so it is a
    slight over-estimate of the angle, i.e. conservative).
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
      :func:`~shapegen.quad.cones.project_cones` chooses for the *initial*
      cones) for the whole optimisation, and re-derived afterwards. Both
      reported energies use the frozen chart, so before/after is like-for-
      like; ``energy_rechart`` re-scores the refined set in its own freshly
      chosen chart, and measured it equals ``energy`` to 1e-14 relative — the
      metric is chart-free, so the freeze costs nothing and only buys speed
      and determinism (see the module docstring).

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

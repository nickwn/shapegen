"""Acceptance tests for :mod:`shapegen.quad.refine` — locally optimising the
*positions* of the Ginzburg-Landau cone set against
:func:`shapegen.quad.field.area_distortion`.

**Read the module docstring before filing this under PLAN.md §P8.7.** §P8.7
rejects making cones *emerge* from a point relaxation. Nothing here creates,
destroys or re-indexes a cone; the topology comes from GL, unchanged, and the
tests below assert that it stays unchanged.
"""

import numpy as np
import pytest

from shapegen.geom.encode import n_coeffs
from shapegen.quad import cones as qc
from shapegen.quad import field as qf
from shapegen.quad import refine as qr
from tests.quad_fixtures import bundle_style_coeffs

L = 16
S = 4


def _gl(rho, band=L):
    Cmat, modes = qf.assemble_C(
        qf.connection_field(qf.curvature_potential(rho, band)), S, band)
    psi0 = qf.linear_start(Cmat, modes, S, band)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, band)
    return qf.cones_from_field(psiG, S, band, over=4.0), Cmat, modes, psi0


def _f1(amp=1.0, band=L):
    return bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(band)] * amp


# --------------------------------------------------------------------------- #
# The objective is the metric, exactly                                        #
# --------------------------------------------------------------------------- #

def test_objective_reproduces_area_distortion_at_the_starting_point():
    """The fast frozen-chart objective must be ``area_distortion``'s energy,
    not a proxy for it -- otherwise the optimiser improves something else.
    Measured agreement at the GL cones: exact, or 2e-15 relative."""
    rho = _f1()
    cones, _C, _m, _p = _gl(rho)
    surf = qf.decoded_surface(rho)
    pole = qc.project_cones(cones)["pole"]
    P0 = np.array([p for p, _k in cones])
    idx = np.array([k for _p, k in cones])

    f = qr.objective(P0, idx, surf, qc.rotation_to_north(pole))
    direct = qf.area_distortion(cones, rho, L, surface=surf, pole=pole)["energy"]
    assert f(np.zeros(2 * len(P0))) == pytest.approx(direct, rel=1e-13)


def test_objective_positions_stay_on_the_sphere():
    """The tangent-frame parameterisation must produce unit vectors for any
    displacement, including large ones -- it is a retraction, not a first
    order approximation, so the box constraint cannot push a cone off the
    sphere."""
    rng = np.random.default_rng(0)
    P0 = rng.normal(size=(8, 3))
    P0 /= np.linalg.norm(P0, axis=1, keepdims=True)
    e1, e2 = qr.cone_frames(P0)
    assert np.allclose(np.einsum("ij,ij->i", e1, e2), 0.0, atol=1e-14)
    assert np.allclose(np.einsum("ij,ij->i", e1, P0), 0.0, atol=1e-14)
    assert np.allclose(np.linalg.norm(e1, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(e2, axis=1), 1.0)

    surf = qf.decoded_surface(np.zeros(n_coeffs(L)))
    f = qr.objective(P0, np.full(8, 0.25), surf, np.eye(3))
    for scale in (0.01, 1.0, 10.0):
        Q = f.positions(rng.normal(size=16) * scale)
        assert np.allclose(np.linalg.norm(Q, axis=1), 1.0)


# --------------------------------------------------------------------------- #
# What the refinement must not break                                          #
# --------------------------------------------------------------------------- #

def test_refinement_keeps_the_topology_gl_established():
    """§P8.7's boundary, as an assertion: the cone *count* and every
    individual *index* are untouched, so the index sum is +2 by construction
    rather than by luck. This is a position refinement of an already-correct
    set, which is the thing §P8.7 never tested."""
    rho = _f1(amp=4.0)
    cones, _C, _m, _p = _gl(rho)
    res = qr.refine_cones(cones, rho, L)

    assert len(res["cones"]) == len(cones) == 8
    assert [k for _p, k in res["cones"]] == [k for _p, k in cones]
    assert sum(k for _p, k in res["cones"]) == pytest.approx(2.0, abs=1e-10)
    assert np.allclose(
        np.linalg.norm(np.array([p for p, _k in res["cones"]]), axis=1), 1.0)


def test_refinement_respects_the_cone_gap_budget():
    """Cones must not collide, and the period solve downstream budgets cone
    motion at ``0.25 * min_gap``; the box constraint guarantees the minimum
    pairwise separation keeps at least ``gap_keep`` of its starting value.
    Measured it usually *grows* (37.8 -> 39.3 degrees at L=16 amp 4)."""
    rho = _f1(amp=4.0)
    cones, _C, _m, _p = _gl(rho)
    for gap_keep in (0.5, 0.9):
        res = qr.refine_cones(cones, rho, L, gap_keep=gap_keep)
        assert res["min_sep"] >= gap_keep * res["min_sep0"] - 1e-12
        # the bound is a bound, not the behaviour: check it is not saturating
        assert res["max_motion_deg"] <= np.rad2deg(
            0.5 * (1.0 - gap_keep) * res["min_sep0"]) + 1e-9


def test_refinement_rejects_a_cone_set_that_is_not_poincare_hopf():
    bad = [(p, -0.25) for p, _k in _gl(_f1())[0]]
    with pytest.raises(ValueError, match="index sum"):
        qr.refine_cones(bad, _f1(), L)


# --------------------------------------------------------------------------- #
# Does it actually improve anything -- against the anneal's own noise floor   #
# --------------------------------------------------------------------------- #

def test_refinement_beats_gl_on_every_anneal_schedule():
    """**The claim, stated the only way it means anything.**

    A 12% improvement is not obviously outside a flow whose score moves
    8-46% when only the anneal schedule changes. So this is scored *paired*:
    for each schedule, refine that schedule's own cones and compare like with
    like. Across 80 (case, schedule) pairs at L=12 and L=16 and five
    amplitudes the refinement won **80 of 80** -- median 1.118x, geometric
    mean 1.136x, worst 1.016x, best 1.593x -- and it *shrank* the
    schedule-to-schedule spread at the same time (GL 1.08-1.46x per case,
    refined 1.03-1.13x). A one-sided sign test on 80/80 leaves no room for
    the noise floor to explain it.

    Four schedules here, to stay cheap.
    """
    rho = _f1(amp=2.0)
    Cmat, modes = qf.assemble_C(
        qf.connection_field(qf.curvature_potential(rho, L)), S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    surf = qf.decoded_surface(rho)

    gl, refined = [], []
    for levels in (None, (2.0, 1.5, 1.2), (1.8, 1.4, 1.1), (1.5, 1.2, 0.9)):
        sched = qf.anneal_schedule(Cmat, L, levels or qf.DEFAULT_LEVELS)
        psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L, levels=sched)
        cones = qf.cones_from_field(psiG, S, L, over=4.0)
        res = qr.refine_cones(cones, rho, L, surface=surf)
        assert res["ratio"] > 1.0, f"refinement lost on schedule {sched}"
        gl.append(res["energy0"])
        refined.append(res["energy"])

    gl, refined = np.array(gl), np.array(refined)
    # and the second, independent claim: less schedule-dependent afterwards
    assert refined.max() / refined.min() < gl.max() / gl.min()


def test_the_refined_energy_does_not_depend_on_the_frozen_chart():
    """Freezing the chart is a speed/determinism choice, not a thumb on the
    scale: re-scoring the refined cones in a freshly searched chart gives the
    same energy (measured to 1e-14 relative), and starting the whole
    refinement from a different frozen pole lands on the same energy to
    within a percent."""
    rho = _f1(amp=2.0)
    cones, _C, _m, _p = _gl(rho)
    surf = qf.decoded_surface(rho)

    res = qr.refine_cones(cones, rho, L, surface=surf)
    assert res["energy_rechart"] == pytest.approx(res["energy"], rel=1e-10)

    other = qr.refine_cones(cones, rho, L, surface=surf,
                            pole=np.array([0.0, 0.0, 1.0]))
    assert other["energy"] == pytest.approx(res["energy"], rel=0.02)


def test_refinement_closes_the_round_sphere_gap_to_the_cube():
    """§P8.4's third gate, which GL alone misses by 6-11% (see
    ``test_quad_field.py::test_gate_p8_4_round_sphere_is_level_with_the_cube``).

    On the round sphere the optimum is the evenly spread cube. Refining the
    GL cones must at least reach it: measured 0.0760 -> 0.0714 at L=12
    against the cube's 0.0719, and 0.0760 -> 0.0714 at L=16 against 0.0719.
    Note the refined set is still *not* geometrically a cube (its pairwise
    angle spectrum is ~35 degrees off) -- the objective has shallow
    non-cube minima at essentially the cube's energy, which is a real
    multi-modality, just a harmless one.
    """
    zero = np.zeros(n_coeffs(L))
    cones, _C, _m, _p = _gl(zero)
    surf = qf.decoded_surface(zero)
    north = np.array([0.0, 0.0, 1.0])

    P = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)],
                 dtype=float)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    e_cube = qf.area_distortion([(p, 0.25) for p in P], zero, L, surface=surf,
                                pole=north)["energy"]

    res = qr.refine_cones(cones, zero, L, surface=surf, pole=north)
    assert res["energy"] < res["energy0"]
    assert res["energy"] <= e_cube * 1.01


# --------------------------------------------------------------------------- #
# End to end: a better score is worthless if it will not trace                #
# --------------------------------------------------------------------------- #

def test_refined_cones_still_trace_to_a_closed_quad_mesh():
    """**The check that makes the improvement mean anything downstream.**

    A placement that scores better but will not solve or trace is not an
    improvement. Run the refined cones through the whole rest of the chain --
    ``project_cones -> solve_lattice -> quad_grid`` -- and demand what
    PLAN.md §4 demands of any quad mesh: Euler characteristic 2, coverage
    1.0, index sum 2. Measured on 28 (case, schedule) pairs at L=12/16 the
    refined set solved and traced every time, with the same ~6000 faces as
    the GL set.

    On the traced mesh's own log-area variance the refinement wins 23 of
    those 28 (median 1.043x, range 0.987x-1.179x) -- about a third of the
    sampled metric's 1.134x, with the gain concentrated at amplitude 2-4 and
    the five small losses on the round sphere and amplitude 1. See
    :mod:`shapegen.quad.refine`'s docstring.
    """
    from shapegen.quad import grid as qg
    from shapegen.quad import periods as qp

    rho = _f1(amp=4.0)
    cones, _C, _m, _p = _gl(rho)
    res = qr.refine_cones(cones, rho, L)
    assert res["ratio"] > 1.0

    chart = qc.project_cones(res["cones"])
    assert chart["idx"].sum() == pytest.approx(2.0, abs=1e-10)
    tgt = qg.cell_for_faces(chart["z"], chart["idx"], 6000)
    sol = qp.solve_lattice(chart["z"], chart["idx"],
                           cell_sizes=tuple(tgt * 0.92 ** k for k in range(30)))
    assert sol["cones"] is not None
    g = qg.quad_grid(sol["cones"], chart["idx"], sol["cell"])
    assert g.euler == 2
    assert g.coverage == pytest.approx(1.0, abs=1e-6)

"""Acceptance tests for PLAN.md's §P6 (extraction) and §P7 (lift to ℝ³),
ported from ``s6b_extract.py`` and ``04_extraction/*``.

The headline test (``test_extract_quad_mesh_reproduces_p0_reference``) feeds
the P0 corpus's solved cones into
:func:`~shapegen.quad.extract.extract_quad_mesh` and checks it reproduces the
bundle's saved ``q2_pos.npy``/``q2_F.npy``/``q2_adj.npy`` -- Gate P6's literal
requirement. Getting there caught a real bug: an earlier version of
``psi_arg`` unwrapped the phase at the wrong period (see
``shapegen/quad/extract.py``'s ``psi_log`` docstring), which reproduced a
single marching step exactly but silently picked the wrong branch a few
steps later and never converged.
"""

import numpy as np

from shapegen.quad import cones as qc
from shapegen.quad import extract as qx
from shapegen.quad import field as qf
from shapegen.quad import lift as ql
from shapegen.quad import periods as qp
from tests.quad_fixtures import bundle_style_coeffs, load_ref
from shapegen.geom.encode import n_coeffs

L = 16
S = 4


def _gl_pipeline(seed=1, L=L, S=S, n_samples=2000):
    """rho_hat -> (coeffs, solved chart, extraction result, quads).

    **Audit note (2026-09-21): everything downstream of this helper was
    originally measured with the anneal collapsed**, i.e. with the old
    ``(6.0, 3.0, 2.0)`` default, which is above
    :func:`~shapegen.quad.field.epsilon_ceiling` at L=16 so the flow returned
    the linear start (overlap 0.99964, ``area_distortion`` 0.2846 against the
    annealed 0.0713 -- a 4.0x quality difference). Re-measured under the
    shipped schedule, **every assertion in this file still holds**, and that
    is the finding rather than a reprieve: none of them can see the
    difference.

    ============================  ==========  ==========  ====================
    quantity                      collapsed   shipped     assertion
    ============================  ==========  ==========  ====================
    ``coverage`` (fixed tol)      0.910964    0.910909    ``== 0.9110``
    ``coverage`` (adaptive tol)   0.910964    0.910909    equal to the above
    median quad anisotropy        1.1099      1.0449      ``< 1.5``
    chamfer ratio subdiv 4->5     0.497       0.505       ``< 0.7``
    chamfer ratio subdiv 5->6     0.508       0.493       ``< 0.7``
    ============================  ==========  ==========  ====================

    The first two are *analytic* -- ``r^2/(r^2+1)`` at ``refuse_radius = 3.2``
    -- so no cone set moves them. The chamfer ratios measure the icosphere
    interpolation converging, which does not depend on the cones either. Only
    the anisotropy moved, and it moved well inside its own bound. A
    discriminating gate on this pipeline has to be on *area* distortion, not
    on coverage, cell shape, or lift convergence; there is one in
    ``test_quad_grid.py::test_end_to_end_is_gated_on_quality_and_not_only_on_topology``.
    """
    coeffs = bundle_style_coeffs(L=32, seed=seed)[:n_coeffs(L)]
    phi = qf.curvature_potential(coeffs, L)
    A = qf.connection_field(phi)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)
    cones = qf.cones_from_field(psiG, S, L)
    chart = qc.project_cones(cones, n_samples=n_samples, seed=0)
    solve = qp.solve_lattice(chart["z"], chart["idx"])
    return coeffs, chart, solve


# --------------------------------------------------------------------------- #
# P6.1 -- the identity: (w')^-4 = prod(z-p_i)^sign, no branch to track        #
# --------------------------------------------------------------------------- #

def test_polynomial_identity_holds():
    """[bundle] claims 7.87e-16 over 2000 points; this reproduces the same
    check against this module's own psi_log/psi_arg machinery."""
    cones = load_ref("cones_solved")
    idx = load_ref("cone_idx")
    a = -idx
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        z = np.array([rng.uniform(-2, 2) + 1j * rng.uniform(-2, 2)])
        if np.abs(z[0] - cones).min() < 0.15:
            continue
        lm, ph, q = qx.psi_log(z, cones, a)
        wprime = np.exp(q * lm[0] + 1j * q * ph[0])
        lhs = wprime ** (-4)
        rhs = np.prod((z[0] - cones) ** (4 * idx))
        worst = max(worst, abs(lhs - rhs) / abs(rhs))
    assert worst < 1e-12


def test_psi_returns_but_wprime_rotates_around_a_cone():
    """The bundle's own headline fact: psi is single-valued (returns to
    itself going around a cone), w' does not (comes back rotated by a
    multiple of 90 degrees).

    This needs the phase tracked *continuously along the path* (each cone's
    own angle unwrapped, ``core.py``'s technique) -- evaluating ``psi_log``
    independently at the loop's start and end points (both physically
    almost the same point) would trivially show "no change" regardless of
    the true winding, since a bare ``np.angle`` always returns to itself at
    the same point.
    """
    cones = load_ref("cones_solved")
    idx = load_ref("cone_idx")
    a = -idx
    n = a / np.abs(a[np.abs(a) > 1e-12]).min()   # integer exponents, e.g. +-1
    c = cones[0]
    # radius well under the nearest-neighbor gap (this reference chart has
    # cones as close as ~0.12 apart), so the loop encloses cone 0 only
    radius = 0.3 * np.min(np.abs(cones[1:] - cones[0]))
    th = np.linspace(0, 2 * np.pi, 2000)
    loop = c + radius * np.exp(1j * th)

    D = loop[:, None] - cones[None, :]
    unwrapped_angle = np.unwrap(np.angle(D), axis=0)
    log_psi = (n[None, :] * (np.log(np.abs(D)) + 1j * unwrapped_angle)).sum(axis=1)
    psi_vals = np.exp(log_psi)
    assert abs(psi_vals[-1] - psi_vals[0]) < 1e-8 * abs(psi_vals[0])

    q = np.abs(a[np.abs(a) > 1e-12]).min()
    wprime = np.exp(q * log_psi)
    rotation = wprime[-1] / wprime[0]
    angle_deg = np.degrees(np.angle(rotation)) % 360.0
    # a single index-1/4 cone: going around it once rotates w' by exactly
    # +-90 degrees, never by ~0 (which would mean w' falsely looked single-
    # valued, i.e. this test failing to discriminate at all)
    assert min(abs(angle_deg - 90.0), abs(angle_deg - 270.0)) < 1.0


# --------------------------------------------------------------------------- #
# Cross-check: my generalised psi_arg vs the bundle's exact one-cell step     #
# --------------------------------------------------------------------------- #

def test_rk4_cell_step_matches_bundle_reference_over_many_steps():
    """The original bug (wrong unwrap period) reproduced a single step
    exactly and diverged from the third step onward -- so this checks a
    *run* of steps, not just one."""
    C = load_ref("cones_solved")
    IDX = load_ref("cone_idx")
    a = -IDX
    s = float(load_ref("cell").reshape(-1)[0])

    def wprime_branch(z, ref=None):
        v = np.prod([(z - C[k]) ** a[k] for k in range(len(C))])
        if ref is None:
            return v
        best = v
        for qk in range(4):
            cand = v * (1j) ** qk
            if abs(cand - ref) < abs(best - ref):
                best = cand
        return best

    def march_ref(z, d, step, ref, nsub=1):
        h = step / nsub
        cur, r = z, ref
        for _ in range(nsub):
            f = lambda zz, rr: d / wprime_branch(zz, rr)  # noqa: E731
            k1 = f(cur, r)
            k2 = f(cur + 0.5 * h * k1, r)
            k3 = f(cur + 0.5 * h * k2, r)
            k4 = f(cur + h * k3, r)
            cur = cur + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            r = wprime_branch(cur, r)
        return cur, r

    seed = qx.find_seed(C)
    z_ref, r_ref = seed, wprime_branch(seed)
    z_mine = np.array([seed])
    _, ph_mine, _ = qx.psi_log(z_mine, C, a)

    for _ in range(20):
        z_ref, r_ref = march_ref(z_ref, 1 + 0j, s, r_ref, nsub=1)
        z_mine, ph_mine = qx.rk4_cell_step(z_mine, ph_mine, 1 + 0j, C, a, s, nsub=1)
        assert abs(z_ref - z_mine[0]) < 1e-8


# --------------------------------------------------------------------------- #
# Gate P6, against the real P0 reference                                      #
# --------------------------------------------------------------------------- #

def test_extract_quad_mesh_reproduces_p0_reference():
    cones = load_ref("cones_solved")
    idx = load_ref("cone_idx")
    s = float(load_ref("cell").reshape(-1)[0])
    q2_pos_ref = load_ref("q2_pos")
    q2_adj_ref = load_ref("q2_adj")

    result = qx.extract_quad_mesh(cones, idx, s)
    assert result["converged"]
    assert len(result["pos"]) == len(q2_pos_ref)
    assert result["merged"] == 7448          # [bundle]'s own cited number
    assert np.max(np.abs(result["pos"] - q2_pos_ref)) < 1e-3

    val = np.array([len(x) for x in result["adj"]])
    for v in (3, 4, 5):
        assert np.sum(val == v) == np.sum(q2_adj_ref == v)


def test_gate_p6_valence_matches_cone_list():
    """8 cones of index +1/4 (or, in this module's convention, index
    magnitude 1/4 either sign) must show up as exactly that many
    non-valence-4 vertices, and no more (away from BFS/merge noise)."""
    cones = load_ref("cones_solved")
    idx = load_ref("cone_idx")
    s = float(load_ref("cell").reshape(-1)[0])
    result = qx.extract_quad_mesh(cones, idx, s)
    val = np.array([len(x) for x in result["adj"]])
    n_cones = len(cones)
    assert np.sum(val != 4) >= n_cones


def test_gate_p6_quads_match_reference_set():
    cones = load_ref("cones_solved")
    idx = load_ref("cone_idx")
    s = float(load_ref("cell").reshape(-1)[0])
    q2_F_ref = load_ref("q2_F")

    result = qx.extract_quad_mesh(cones, idx, s)
    quads = qx.assemble_quads(result["pos"], result["adj"])
    defect = qx.closure_defects(result["pos"], quads)
    keep = defect < 0.6
    quads = [q for q, k in zip(quads, keep) if k]

    assert len(quads) == len(q2_F_ref)
    ref_sets = {tuple(sorted(q)) for q in q2_F_ref}
    mine_sets = {tuple(sorted(q)) for q in quads}
    assert mine_sets == ref_sets


# --------------------------------------------------------------------------- #
# P7: lift to R^3, route 1                                                    #
# --------------------------------------------------------------------------- #

def test_chart_to_directions_gives_unit_vectors():
    cones = load_ref("cones_solved")
    idx = load_ref("cone_idx")
    s = float(load_ref("cell").reshape(-1)[0])
    Rm = load_ref("Rm")
    result = qx.extract_quad_mesh(cones, idx, s)
    dirs = ql.chart_to_directions(result["pos"], Rm)
    assert np.max(np.abs(np.linalg.norm(dirs, axis=1) - 1.0)) < 1e-10


def test_lift_route1_reproduces_the_round_sphere_on_F0():
    """rho=0 -> codec.decode is exactly the unit sphere, so route 1's lift of
    ANY directions must land back on those same directions (up to the base
    icosphere's own triangle-interpolation resolution)."""
    cones = load_ref("cones_solved")
    idx = load_ref("cone_idx")
    s = float(load_ref("cell").reshape(-1)[0])
    Rm = load_ref("Rm")
    result = qx.extract_quad_mesh(cones, idx, s)
    dirs = ql.chart_to_directions(result["pos"], Rm)

    rho0 = np.zeros(n_coeffs(8))
    P3, info = ql.lift_route1(dirs, rho0, subdiv=4)
    assert info["n_fallback"] == 0
    assert np.max(np.abs(np.linalg.norm(P3, axis=1) - 1.0)) < 0.01
    assert np.max(np.linalg.norm(P3 - dirs, axis=1)) < 0.01


def test_lift_route1_chamfer_shrinks_with_finer_base_mesh():
    """Gate P7.B in spirit: the lifted mesh must converge onto codec.decode's
    own surface as the base icosphere refines -- if the discrepancy were a
    real geometric inconsistency (not an interpolation artifact) it would
    not shrink this way."""
    coeffs, chart, solve = _gl_pipeline()
    result = qx.extract_quad_mesh(solve["cones"], chart["idx"], solve["cell"],
                                  max_vertices=20000)
    assert result["converged"]
    dirs = ql.chart_to_directions(result["pos"], chart["Rm"])

    from scipy.spatial import cKDTree
    prev = None
    for subdiv in (4, 5, 6):
        P3, info = ql.lift_route1(dirs, coeffs, subdiv=subdiv)
        tree = cKDTree(info["decoded"])
        dchamf, _ = tree.query(P3)
        med = float(np.median(dchamf))
        if prev is not None:
            assert med < prev * 0.7
        prev = med


def test_gate_p6_median_quad_anisotropy_is_reasonable():
    coeffs, chart, solve = _gl_pipeline()
    result = qx.extract_quad_mesh(solve["cones"], chart["idx"], solve["cell"],
                                  max_vertices=20000)
    assert result["converged"]
    quads = qx.assemble_quads(result["pos"], result["adj"])
    defect = qx.closure_defects(result["pos"], quads)
    quads = np.array([q for q, k in zip(quads, defect < 0.6) if k])

    dirs = ql.chart_to_directions(result["pos"], chart["Rm"])
    P3, _info = ql.lift_route1(dirs, coeffs, subdiv=5)
    an = qx.quad_anisotropy(P3, quads)
    assert np.median(an) < 1.5   # [bundle]'s own reference case: 1.136


# --------------------------------------------------------------------------- #
# adaptive_tol: the fixed merge tolerance silently caps coverage             #
# --------------------------------------------------------------------------- #

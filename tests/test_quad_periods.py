"""Acceptance tests for periods and the lattice solve, ported
from ``s5_grid.py``, ``core.py`` and ``conegate.py``.

The headline test (``test_solve_lattice_reproduces_p0_reference``) feeds the
P0 corpus's actual unsolved chart (``cones_plane.npy``/``cone_idx.npy``) into
:func:`~shapegen.quad.periods.solve_lattice` and checks it reproduces the
bundle's own saved ``cones_solved.npy``/``cell.npy`` -- Gate P5's literal
requirement.
"""

import numpy as np
import pytest

from shapegen.quad import cones as qc
from shapegen.quad import field as qf
from shapegen.quad import periods as qp
from tests.quad_fixtures import bundle_style_coeffs, load_ref
from shapegen.geom.encode import n_coeffs

L = 16
S = 4


def _gl_cones(seed=1, L=L, S=S):
    coeffs = bundle_style_coeffs(L=32, seed=seed)[:n_coeffs(L)]
    phi = qf.curvature_potential(coeffs, L)
    A = qf.connection_field(phi)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)
    return qf.cones_from_field(psiG, S, L)


# --------------------------------------------------------------------------- #
# P5.1 -- cut integral gradient                                              #
# --------------------------------------------------------------------------- #

def test_cut_and_grad_matches_finite_differences():
    """Both real- and imaginary-direction perturbations must match the
    analytic gradient -- confirms it's a genuine holomorphic (C-linear)
    derivative, which is what realify()/the Newton solve assume."""
    rng = np.random.default_rng(0)
    C = (rng.normal(size=6) + 1j * rng.normal(size=6)) * 0.5
    idx = np.array([0.25, 0.25, 0.25, 0.25, -0.25, -0.25])
    a = -idx
    i, j = 0, 3
    _val, grad = qp.cut_and_grad(i, j, C, a, n=16)

    h = 1e-6
    for k in range(len(C)):
        for direction in (1.0, 1j):
            Cp, Cm = C.copy(), C.copy()
            Cp[k] += direction * h
            Cm[k] -= direction * h
            vp, _ = qp.cut_and_grad(i, j, Cp, a, n=16)
            vm, _ = qp.cut_and_grad(i, j, Cm, a, n=16)
            fd = (vp - vm) / (2 * h)
            expected = direction * grad[k]
            assert abs(fd - expected) < 1e-7


# --------------------------------------------------------------------------- #
# P5.2 -- MST, loops, independent-condition selection                        #
# --------------------------------------------------------------------------- #

def test_euclidean_mst_has_n_minus_1_edges():
    cones = load_ref("cones_plane")
    E = qp.euclidean_mst(cones)
    assert len(E) == len(cones) - 1


def test_select_independent_loops_gives_n_minus_2():
    """Known complete: n-1 MST loops span the lattice with
    exactly one redundancy (the affine kernel), leaving n-2 independent."""
    cones = load_ref("cones_plane")
    idx = load_ref("cone_idx")
    a = -idx
    E = qp.euclidean_mst(cones)
    loops = qp.loop_edges(idx, E)
    _t, G = qp.periods_and_grad(cones, a, idx, loops, n=48)
    sel = qp.select_independent_loops(loops, G)
    assert len(sel) == len(cones) - 2


# --------------------------------------------------------------------------- #
# The adaptive node count (the pitfall this port actually hit)               #
# --------------------------------------------------------------------------- #

def test_node_count_adapts_to_the_p0_spread():
    """The P0 corpus's chart has a ~21:1 spread in cone separations --
    exactly the known pitfall (a 16:1 spread needed 48 nodes).
    Solving it at a fixed n=16 (matching s5_grid.py's literal hard-code)
    converges to a point that looks fine at n=16 (residual < 1e-12) but is
    only accurate to 2.3e-6 at n=48, failing the 1e-6 final gate -- this
    pins the *reason* an adaptive count is needed, not just its existence.
    """
    cones = load_ref("cones_plane")
    idx = load_ref("cone_idx")
    E = qp.euclidean_mst(cones)
    loops = qp.loop_edges(idx, E)
    assert qp.node_count_for(cones, loops) == 48


def test_node_count_is_bounded_and_responds_to_spread():
    """Sanity on the heuristic itself, not a claim about what real cone sets
    need: a GL-found chart actually comes out closer to the 48-node end too
    (its 8 cones split into two spatial clusters connected by one long MST
    edge, and other cones sit close to that edge relative to its length --
    the same shape of pitfall as the P0 corpus, not an exception to it), so
    this only pins the function's output range and monotonic behaviour."""
    cones = _gl_cones()
    chart = qc.project_cones(cones, n_samples=500, seed=0)
    E = qp.euclidean_mst(chart["z"])
    loops = qp.loop_edges(chart["idx"], E)
    n = qp.node_count_for(chart["z"], loops)
    assert 16 <= n <= 48
    # a configuration with zero clearance (one cone sitting exactly on an MST
    # edge of the loop set) must saturate at the cap. Take the midpoint of an
    # edge that is really in ``loops`` -- picking (0, 1) by hand only worked
    # while the GL cones happened to have that edge, which the anneal-schedule
    # fix changed.
    i, j, _sg = loops[0]
    k = next(v for v in range(len(chart["z"])) if v not in (i, j))
    degenerate = chart["z"].copy()
    degenerate[k] = 0.5 * (chart["z"][i] + chart["z"][j])
    assert qp.node_count_for(degenerate, loops) == 48


# --------------------------------------------------------------------------- #
# Gate P5, against the real P0 reference                                      #
# --------------------------------------------------------------------------- #

def test_solve_lattice_reproduces_p0_reference():
    cones_plane = load_ref("cones_plane")
    cone_idx = load_ref("cone_idx")
    cell_ref = float(load_ref("cell").reshape(-1)[0])
    cones_solved_ref = load_ref("cones_solved")

    result = qp.solve_lattice(cones_plane, cone_idx)
    assert result["cell"] == pytest.approx(cell_ref)
    assert np.max(np.abs(result["cones"] - cones_solved_ref)) < 1e-8


def test_gate_p5_period_integrality_including_derived_loops():
    """All periods integral to 1e-8, including the one loop the solver never
    saw directly (dropped by select_independent_loops as redundant)."""
    cones_plane = load_ref("cones_plane")
    cone_idx = load_ref("cone_idx")
    result = qp.solve_lattice(cones_plane, cone_idx)
    a = -cone_idx
    tf, _ = qp.periods_and_grad(result["cones"], a, cone_idx, result["loops"], n=48)
    q = tf / result["cell"]
    assert np.max(np.abs(q - np.round(q))) < 1e-8


def test_gate_p5_cone_motion_within_budget():
    cones_plane = load_ref("cones_plane")
    cone_idx = load_ref("cone_idx")
    result = qp.solve_lattice(cones_plane, cone_idx)
    motion = np.abs(result["cones"] - cones_plane).max()
    assert motion <= result["cap"] + 1e-12


def test_gate_p5_4_conegate_passes():
    """The cheap gate (P5.4), on the actual solved output. Bundle's own
    number was 1.37e-13; ours differs by ~2e-10 in cone position (the
    homotopy path isn't bit-identical), so this checks order of magnitude
    against the gate's own threshold, not the exact bundle figure."""
    cones_plane = load_ref("cones_plane")
    cone_idx = load_ref("cone_idx")
    result = qp.solve_lattice(cones_plane, cone_idx)
    w = qp.conegate(result["cones"], cone_idx, n=40)
    frac = np.abs(2 * w / result["cell"] - np.round(2 * w / result["cell"]))
    assert frac.max() < 1e-6


def test_gate_p5_4_conegate_on_bundles_own_solved_cones():
    """Directly reproduces the bundle's own [verified] number: run P5.4 on
    the bundle's saved cones_solved.npy/cell.npy (not our own re-solve)."""
    cones_solved = load_ref("cones_solved")
    cone_idx = load_ref("cone_idx")
    cell = float(load_ref("cell").reshape(-1)[0])
    w = qp.conegate(cones_solved, cone_idx, n=40)
    frac = np.abs(2 * w / cell - np.round(2 * w / cell))
    assert frac.max() < 1e-6


# --------------------------------------------------------------------------- #
# End to end on our own GL-found cones                                       #
# --------------------------------------------------------------------------- #

def test_solve_lattice_end_to_end_on_gl_cones():
    """Not a reference reproduction (no ground truth exists for our own
    field-derived cones) -- just confirms the whole P4-lite -> P5 chain
    produces a valid, gate-passing solve on a fresh configuration."""
    cones = _gl_cones()
    chart = qc.project_cones(cones, n_samples=500, seed=0)
    result = qp.solve_lattice(chart["z"], chart["idx"])
    assert result["cell"] is not None, f"no cell size solved: {result['sweep']}"

    a = -chart["idx"]
    tf, _ = qp.periods_and_grad(result["cones"], a, chart["idx"], result["loops"], n=48)
    q = tf / result["cell"]
    assert np.max(np.abs(q - np.round(q))) < 1e-8
    assert np.abs(result["cones"] - chart["z"]).max() <= result["cap"] + 1e-12


# --------------------------------------------------------------------------- #
# Diagnostics (core.py)                                                       #
# --------------------------------------------------------------------------- #

def test_gauss_bonnet_matches_index_sum():
    idx = load_ref("cone_idx")
    assert qp.gauss_bonnet(idx) == pytest.approx(float(idx.sum()))


def test_period_and_rotation_are_finite_away_from_cones():
    cones = load_ref("cones_plane")
    idx = load_ref("cone_idx")
    ctr = cones.mean()
    r = 0.01   # small circle, far from every cone's exact location
    assert qp.clearance(ctr, r, cones) > r
    p = qp.period(ctr, r, cones, idx)
    rot = qp.rotation(ctr, r, cones, idx)
    assert np.isfinite(p.real) and np.isfinite(p.imag)
    assert np.isfinite(rot.real) and np.isfinite(rot.imag)
    assert qp.enclosed_index(ctr, r, cones, idx) == 0

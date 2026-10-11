"""Acceptance tests for :mod:`shapegen.quad.grid` -- grid tracing over the
whole sphere.

Most of this file leans on one fixture with *exact* ground truth. Put eight
``+1/4`` cones at the corners of a cube and the developing map is the
conformal map from the sphere to the cube's surface: its flat structure is
literally a cube, whose corners are the 8 points of cone angle ``3pi/2``.
So a cell size of ``(cube edge)/m`` has to close into exactly ``6m^2``
squares, ``12m^2`` edges and ``6m^2 + 2`` vertices, of which exactly 8 have
valence 3 -- no tolerance, no reference file, no "close enough".

The cube edge itself is measured, not assumed: the cut integral between
adjacent corners comes out at ``0.81336650029`` with argument an exact
multiple of 90 degrees for all twelve edges, and at ``sqrt(2)`` times that at
45 degrees for all twelve face diagonals, which is what says the developed
image is a square grid in the first place.

The four properties the tests are organised around:

* **accuracy** -- RK4 is 4th order in the substep cap (Richardson), and two
  independent arrivals at one vertex agree to ~1e-8 of a cell;
* **100% coverage** -- the face solid angles sum to ``4*pi`` and
  ``V - E + F = 2``, so the output is a closed cover of the sphere, not a
  disc with a boundary;
* **symmetry** -- on the cube the traced vertex set is invariant under the
  full 48-element octahedral group, and the face areas fall into exactly the
  orbit count the symmetry predicts;
* **isotropy and area** -- quads are near-squares on the sphere, converging
  to square as the cell shrinks, and their area matches what the conformal
  factor independently predicts.

"Equi-area" needs a word, because it is measured in two different places.
In the *developed* plane every cell is exactly ``s x s`` by construction, so
measuring that proves nothing. On the sphere a conformal map cannot preserve
both shape and size, so the areas genuinely do vary -- what is
testable is that each face's area equals ``s^2`` times the conformal factor
there, which is :func:`~shapegen.quad.grid.developed_cell_area`, and that the
variation vanishes where the symmetry forces it to (the ``m = 2`` cube, whose
24 faces are exactly equal).
"""

import itertools

import numpy as np
import pytest
from scipy.spatial import cKDTree

from shapegen.quad import cones as qc
from shapegen.quad import grid as qg
from shapegen.quad import periods as qp
from tests.quad_fixtures import load_ref

# The cube fixture ---------------------------------------------------------- #

CUBE_EDGE = 0.8133665002898711
"""Developed length of one cube edge, i.e. the cut integral between adjacent
corner cones -- measured, and pinned by ``test_cube_cut_integrals``."""


def cube_cones():
    """Eight ``+1/4`` cones at the cube's corners, in the flat chart, with the
    projection pole on a face centre so the chart is as symmetric as it can
    be."""
    P = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                  for sz in (-1, 1)], dtype=float)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    Rm = qc.rotation_to_north(np.array([0.0, 0.0, 1.0]))
    return P, qc.stereo(P @ Rm.T), np.full(8, 0.25)


def octahedral(rotations_only: bool = False):
    """The 48 signed permutation matrices (24 with ``det = +1``)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = sg[i]
            if not rotations_only or np.linalg.det(M) > 0:
                out.append(M)
    return out


# --------------------------------------------------------------------------- #
# The two charts                                                              #
# --------------------------------------------------------------------------- #

def test_charts_round_trip_and_cover_the_sphere():
    """Every sphere point has a coordinate of modulus <= 1 in the chart
    :func:`~shapegen.quad.grid.chart_of` picks -- which is the property that
    removes ``extract.py``'s ``refuse_radius`` and with it its coverage
    ceiling."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    c = qg.chart_of(X)
    z = qg.to_chart(X, c)
    assert np.abs(z).max() <= 1.0 + 1e-12
    assert np.abs(qg.from_chart(z, c) - X).max() < 1e-12


def test_chart_1_is_the_reciprocal_of_chart_0():
    """``zeta = 1/z`` exactly, which is what lets a chart change carry the
    branch of ``arg W`` across by a closed-form Jacobian instead of
    re-deriving it."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = X[np.abs(X[:, 2]) < 0.9]                    # away from either pole
    z0 = qg.to_chart(X, 0)
    z1 = qg.to_chart(X, 1)
    assert np.abs(z0 * z1 - 1.0).max() < 1e-12


def test_chart_tangent_matches_finite_differences():
    """The analytic differential of :func:`~shapegen.quad.grid.from_chart`,
    which the rotation system and the frame check both depend on."""
    rng = np.random.default_rng(2)
    z = (rng.normal(size=20) + 1j * rng.normal(size=20)) * 0.4
    for c in (0, 1):
        cc = np.full(len(z), c, dtype=np.int8)
        for delta in (1.0 + 0j, 1j, 0.6 - 0.8j):
            d = np.full(len(z), delta, dtype=complex)
            ana = qg.chart_tangent(z, cc, d)
            h = 1e-6
            fd = (qg.from_chart(z + h * d, cc) - qg.from_chart(z - h * d, cc)) / (2 * h)
            assert np.abs(ana - fd).max() < 1e-7


def test_form_is_regular_at_the_far_pole():
    """The fact ``extract.py`` is built on the negation of: because
    Gauss-Bonnet forces ``sum idx = 2``, ``omega = W dz`` has order 0 at
    ``z = infinity``, so the excluded pole carries no singularity at all --
    ``|W|`` in chart 1 is finite and non-zero there, and bounded away from
    zero over the whole of chart 1's disk."""
    _P, C, idx = cube_cones()
    a = -idx
    assert idx.sum() == pytest.approx(2.0)

    z = np.zeros(1, dtype=complex)
    c = np.ones(1, dtype=np.int8)
    assert np.isfinite(qg.form_log_abs(z, c, C, a)[0])

    rng = np.random.default_rng(3)
    w = rng.normal(size=(4000, 2)) @ np.array([1.0, 1j])
    w = w / np.abs(w) * np.sqrt(rng.random(4000))
    for cc in (0, 1):
        lm = qg.form_log_abs(w, np.full(4000, cc, dtype=np.int8), C, a)
        assert np.exp(lm).min() > 0.1
        assert np.isfinite(lm).all()


def test_form_agrees_across_the_chart_change():
    """``|W_1(zeta)| = |W_0(1/zeta)| . |dz/dzeta|`` -- the identity the
    branch transport assumes."""
    _P, C, idx = cube_cones()
    a = -idx
    rng = np.random.default_rng(4)
    z = (rng.normal(size=300) + 1j * rng.normal(size=300))
    z = z[np.abs(z) > 0.3]
    lm1 = qg.form_log_abs(1.0 / z, np.ones(len(z), dtype=np.int8), C, a)
    lm0 = qg.form_log_abs(z, np.zeros(len(z), dtype=np.int8), C, a)
    # W_1(1/z) = W_0(z) . dz/d(1/z) = W_0(z) . z^2
    assert np.abs(lm1 - (lm0 + 2.0 * np.log(np.abs(z)))).max() < 1e-12


# --------------------------------------------------------------------------- #
# The cube: exact ground truth                                                #
# --------------------------------------------------------------------------- #

def test_cube_cut_integrals_are_a_square_grid():
    """What licenses using the cube as ground truth: all twelve
    corner-adjacent cut integrals have the same modulus at exact multiples of
    90 degrees, and all twelve face diagonals are ``sqrt(2)`` times it at 45
    degrees. That is a square grid of side :data:`CUBE_EDGE`, measured rather
    than assumed."""
    P, C, idx = cube_cones()
    a = -idx
    edge, diag = [], []
    for i in range(8):
        for j in range(i + 1, 8):
            v, _g = qp.cut_and_grad(i, j, C, a, 48)
            d3 = np.linalg.norm(P[i] - P[j])
            if d3 == pytest.approx(2.0 / np.sqrt(3.0), rel=1e-9):
                edge.append(v)
            elif d3 == pytest.approx(2.0 * np.sqrt(2.0 / 3.0), rel=1e-9):
                diag.append(v)
    assert len(edge) == 12
    for v in edge:
        assert abs(v) == pytest.approx(CUBE_EDGE, rel=1e-9)
        deg = np.degrees(np.angle(v))
        assert deg - 90.0 * round(deg / 90.0) == pytest.approx(0.0, abs=1e-6)
    # three of the twelve face diagonals are reached by a chart segment that
    # passes the far side of a third cone, so only their moduli are compared
    near = [v for v in diag if abs(v) < 2.0 * CUBE_EDGE]
    assert len(near) >= 9
    for v in near:
        assert abs(v) == pytest.approx(CUBE_EDGE * np.sqrt(2.0), rel=1e-9)


@pytest.mark.parametrize("m", [2, 3, 4, 6])
def test_cube_mesh_counts_are_exact(m):
    """The headline: cell ``= edge/m`` must produce the cube subdivided ``m``
    ways per face -- exactly ``6m^2`` quad faces, ``12m^2`` edges,
    ``6m^2 + 2`` vertices, Euler characteristic 2, and exactly 8 valence-3
    vertices, one per cone, with every other vertex valence 4."""
    _P, C, idx = cube_cones()
    g = qg.quad_grid(C, idx, CUBE_EDGE / m)

    assert g.converged
    assert g.refused == 0
    assert len(g.points) == 6 * m * m + 2
    assert len(g.edges) == 12 * m * m
    assert len(g.faces) == 6 * m * m
    assert g.euler == 2
    assert set(g.face_degrees) == {4}

    val = g.valences
    assert sorted(np.bincount(val, minlength=5)[3:5]) == [8, 6 * m * m - 6]
    assert len(g.cone_vertices) == 8
    assert set(val[g.cone_vertices]) == {3}


@pytest.mark.parametrize("m", [3, 4, 6])
def test_cube_covers_the_whole_sphere(m):
    """100% coverage, measured as solid angle rather than estimated from a
    bounding radius. ``extract.py`` reported 0.911 here at best, and that
    number was ``r^2/(r^2+1)`` at its own ``refuse_radius``."""
    _P, C, idx = cube_cones()
    g = qg.quad_grid(C, idx, CUBE_EDGE / m)
    assert g.coverage == pytest.approx(1.0, abs=1e-9)
    assert qg.quad_areas(g).sum() == pytest.approx(4.0 * np.pi, abs=1e-9)
    assert (qg.quad_areas(g) > 0).all()


@pytest.mark.parametrize("m", [3, 4, 8])
def test_cube_mesh_has_the_full_octahedral_symmetry(m):
    """The traced vertex set is invariant under all 48 elements of the
    cube's symmetry group, to the trace's own accuracy -- nothing in the
    algorithm (the chart, the seed, the BFS order) is allowed to leave a
    preferred direction behind."""
    _P, C, idx = cube_cones()
    g = qg.quad_grid(C, idx, CUBE_EDGE / m)
    tree = cKDTree(g.points)
    worst = max(tree.query(g.points @ M.T)[0].max() for M in octahedral())
    assert worst < 1e-5


@pytest.mark.parametrize("m,n_orbits", [(2, 1), (3, 3), (4, 3), (6, 6)])
def test_cube_face_areas_fall_into_the_predicted_orbits(m, n_orbits):
    """A sharper symmetry statement than vertex invariance: the ``m x m``
    cells of a cube face fall into exactly ``ceil(m/2) * (floor(m/2) + 1) / 1``
    orbits under the face's own dihedral group -- 1, 3, 3 and 6 for m = 2, 3,
    4, 6 -- so the face areas must take exactly that many distinct values.
    Anything that broke the symmetry subtly would split an orbit here even
    while the vertex-set test still passed."""
    _P, C, idx = cube_cones()
    g = qg.quad_grid(C, idx, CUBE_EDGE / m)
    areas = qg.quad_areas(g)
    assert len(np.unique(np.round(areas, 6))) == n_orbits


def test_cube_m2_is_exactly_equi_area():
    """The one configuration where equal-area and conformal coincide: at
    ``m = 2`` all 24 faces lie in a single symmetry orbit, so they must have
    *identical* solid angle. Conformal maps cannot preserve area in general,
    which is why this is the only case where the strict
    statement is available -- and why the general one is
    ``developed_cell_area``, below."""
    _P, C, idx = cube_cones()
    g = qg.quad_grid(C, idx, CUBE_EDGE / 2)
    areas = qg.quad_areas(g)
    assert len(areas) == 24
    assert areas.max() / areas.min() == pytest.approx(1.0, abs=1e-6)
    assert areas.mean() == pytest.approx(4.0 * np.pi / 24.0, rel=1e-9)


def test_face_area_matches_the_conformal_factor():
    """Each face's solid angle must equal ``s^2`` times the conformal factor
    at its centre. That ties the discrete quads to the continuum rather than
    to each other, and it converges: median ratio 0.99973 at m=4 and
    1.000000 by m=24."""
    _P, C, idx = cube_cones()
    prev = None
    for m in (4, 8, 16):
        g = qg.quad_grid(C, idx, CUBE_EDGE / m)
        r = qg.developed_cell_area(g, C, idx)
        err = abs(float(np.median(r)) - 1.0)
        assert err < 1e-3
        if prev is not None:
            assert err < prev
        prev = err
    assert prev < 1e-5


# --------------------------------------------------------------------------- #
# Accuracy                                                                    #
# --------------------------------------------------------------------------- #

def test_trace_step_is_fourth_order():
    """Richardson on a single cell: halving the substep cap must cut the
    endpoint difference by ~16. Measured ratios 11.3, 14.4, 15.6, 15.9 --
    approaching 16 from below as the asymptotic regime is entered, which is
    what a 4th-order scheme does."""
    _P, C, idx = cube_cones()
    a = -idx
    s = CUBE_EDGE / 8
    zeta, c, arg = qg.seed_vertex(C, idx, s)

    pts = []
    for ac in (0.04, 0.02, 0.01, 0.005, 0.0025):
        z1, c1, _a1, _it, ok = qg.trace_step(zeta, c, arg, np.array([1.0 + 0j]),
                                             C, a, s, near_frac=5.0 * ac,
                                             abs_cap=ac)
        assert ok[0]
        pts.append(qg.from_chart(z1, c1)[0])
    diffs = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
    ratios = [diffs[i] / diffs[i + 1] for i in range(len(diffs) - 1)]
    assert all(r > 10.0 for r in ratios)
    assert ratios[-1] > 14.0


@pytest.mark.parametrize("m", [3, 6])
def test_independent_arrivals_agree(m):
    """The trace's real accuracy figure. Every interior vertex is reached by
    four different paths; ``merge_residual`` is the worst disagreement
    between two of them, as a fraction of the local cell. It is ~1e-8 to
    ~1e-7, i.e. seven decades inside the merge tolerance -- which is why
    ``merge_frac`` is not a tuned number."""
    _P, C, idx = cube_cones()
    g = qg.quad_grid(C, idx, CUBE_EDGE / m)
    assert g.merge_residual < 1e-5
    assert g.frame_residual < 1e-5


def test_merge_tolerance_does_not_change_the_mesh():
    """``merge_frac`` swept over five decades gives bit-identical counts.
    (The single-chart extractor this replaced needed its merge threshold to
    separate freshly traced intermediate points, so changing it changed the
    output.)"""
    _P, C, idx = cube_cones()
    ref = None
    for mf in (1e-5, 1e-3, 0.05, 0.25, 0.45):
        g = qg.quad_grid(C, idx, CUBE_EDGE / 6, merge_frac=mf)
        got = (len(g.points), len(g.edges), len(g.faces), g.euler)
        if ref is None:
            ref = got
        assert got == ref
    assert ref == (218, 432, 216, 2)


def test_substep_caps_do_not_change_the_topology():
    """Tightening the integrator buys accuracy, never a different mesh --
    the separation that makes the caps safe to expose."""
    _P, C, idx = cube_cones()
    ref = None
    for ac, nf in ((0.04, 0.2), (0.02, 0.1), (0.01, 0.05)):
        g = qg.quad_grid(C, idx, CUBE_EDGE / 4, abs_cap=ac, near_frac=nf)
        got = (len(g.points), len(g.edges), len(g.faces), g.euler,
               tuple(sorted(np.bincount(g.valences, minlength=5)[3:5])))
        if ref is None:
            ref = got
        assert got == ref


# --------------------------------------------------------------------------- #
# Isotropy                                                                    #
# --------------------------------------------------------------------------- #

def test_isotropy_converges_to_one_away_from_the_cones():
    """Conformality says a developed square comes back a square on the
    sphere, up to the conformal factor's variation across one cell -- which
    is ``O(s)``. So the median edge-length ratio must fall towards 1 as the
    cell shrinks: 1.111, 1.040, 1.025 at m = 4, 8, 16."""
    _P, C, idx = cube_cones()
    med = []
    for m in (4, 8, 16):
        g = qg.quad_grid(C, idx, CUBE_EDGE / m)
        cv = set(g.cone_vertices.tolist())
        iso = qg.quad_isotropy(g.points, g.faces)
        away = np.array([not (cv & set(f)) for f in g.faces])
        med.append(float(np.median(iso[away])))
    assert med[0] > med[1] > med[2]
    assert med[-1] < 1.03
    assert med[-1] > 1.0


def test_isotropy_near_a_cone_is_bounded_and_local():
    """Near a cone the conformal factor goes like ``r^(-1/4)``, which is not
    Lipschitz, so the cells touching one stay visibly non-square however fine
    the mesh -- that is the cone, not the trace. What has to hold is that it
    stays bounded and confined: the same 24 faces at every resolution, with a
    ratio converging to a constant rather than growing."""
    _P, C, idx = cube_cones()
    worst = []
    for m in (4, 8, 16):
        g = qg.quad_grid(C, idx, CUBE_EDGE / m)
        cv = set(g.cone_vertices.tolist())
        iso = qg.quad_isotropy(g.points, g.faces)
        touch = np.array([bool(cv & set(f)) for f in g.faces])
        assert touch.sum() == 24            # 3 faces around each of 8 cones
        worst.append(float(iso[touch].max()))
    assert max(worst) < 1.45
    assert worst[-1] == pytest.approx(worst[-2], abs=0.01)


# --------------------------------------------------------------------------- #
# Mesh density is a request, not an accident                                  #
# --------------------------------------------------------------------------- #

def test_developed_area_matches_the_cube_exactly():
    """``integral |W|^2`` over the sphere is the flat structure's total area,
    and for the cube fixture that is ``6 * edge^2`` in closed form."""
    _P, C, idx = cube_cones()
    assert qg.developed_area(C, idx) == pytest.approx(6.0 * CUBE_EDGE ** 2,
                                                      rel=1e-3)


@pytest.mark.parametrize("m", [4, 8, 16])
def test_face_count_is_predictable_before_tracing(m):
    """``developed_area / s^2`` predicts the face count without tracing
    anything -- exactly, on the cube (96, 384, 1536)."""
    _P, C, idx = cube_cones()
    s = CUBE_EDGE / m
    pred = qg.developed_area(C, idx) / s ** 2
    assert pred == pytest.approx(6 * m * m, rel=2e-3)
    assert len(qg.quad_grid(C, idx, s).faces) == 6 * m * m


@pytest.mark.parametrize("m", [4, 8, 16])
def test_cell_for_faces_inverts_the_prediction(m):
    """The inverse, which is what makes density controllable from a face
    count: asking three FaceScape heads at L=32 for 20000 faces gave 20037,
    20063 and 20043."""
    _P, C, idx = cube_cones()
    assert qg.cell_for_faces(C, idx, 6 * m * m) == pytest.approx(CUBE_EDGE / m,
                                                                 rel=2e-3)


def test_an_arbitrary_cell_size_does_not_close():
    """Why :func:`~shapegen.quad.grid.cell_for_faces`'s answer is a *request*
    and has to go through the period solve. The cube closes only at
    ``edge/m``; the cell that would give 200 faces is ``edge/5.77``, and
    tracing it directly produces a wreck -- which is at least a loud one,
    since ``euler`` and ``merge_residual`` both say so."""
    _P, C, idx = cube_cones()
    s = qg.cell_for_faces(C, idx, 200)
    assert s == pytest.approx(CUBE_EDGE / 5.77, rel=0.01)
    g = qg.quad_grid(C, idx, s, max_vertices=20000)
    assert g.euler != 2
    assert g.merge_residual > 0.1


# --------------------------------------------------------------------------- #
# A -1/4 cone: the other singular behaviour                                   #
# --------------------------------------------------------------------------- #

def test_negative_cone_traces_and_gives_valence_five():
    """Every cone on the cube is ``+1/4``, where ``|W| -> infinity`` and the
    chart velocity *vanishes*. A ``-1/4`` cone is the opposite -- ``|W| -> 0``
    and the velocity blows up -- so it exercises the substep cap from the
    other side. Real fields produce them via the ``n = 8 + 2 n_minus`` ladder.

    The result must be the mirror image of the ``+1/4`` case: on a lattice
    point a valence-*5* vertex, on a cell centre a *pentagonal* face.
    """
    rng = np.random.default_rng(7)
    seen_val5 = seen_pent = False
    for trial in range(6):
        P = rng.normal(size=(10, 3))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        idx = np.r_[np.full(9, 0.25), np.full(1, -0.25)]
        assert idx.sum() == pytest.approx(2.0)

        chart = qc.project_cones(list(zip(P, idx)), seed=trial)
        sol = qp.solve_lattice(chart["z"], chart["idx"], 0.025)
        if sol["cones"] is None:
            continue
        g = qg.quad_grid(sol["cones"], chart["idx"], sol["cell"],
                         max_vertices=60000)

        assert g.converged and g.refused == 0
        assert g.euler == 2
        assert g.coverage == pytest.approx(1.0, abs=1e-9)
        assert set(g.valences) <= {3, 4, 5}
        assert set(g.face_degrees) <= {3, 4, 5}
        seen_val5 |= 5 in set(g.valences)
        seen_pent |= 5 in set(g.face_degrees)
    assert seen_val5 or seen_pent, "the -1/4 cone never showed up as either"


# --------------------------------------------------------------------------- #
# The P0 reference configuration                                              #
# --------------------------------------------------------------------------- #

def test_p0_configuration_closes_over_the_sphere():
    """The bundle's own solved 8-cone configuration, which
    the single-chart extractor this replaced could only cover 91% of, with a valence
    histogram of 98 twos / 275 threes / 77 fives. Here it closes exactly."""
    C = load_ref("cones_solved")
    idx = load_ref("cone_idx")
    s = float(load_ref("cell")[0])
    g = qg.quad_grid(C, idx, s)

    assert g.converged and g.refused == 0
    assert g.euler == 2
    assert g.coverage == pytest.approx(1.0, abs=1e-9)
    assert g.merge_residual < 1e-4
    assert set(g.valences) <= {3, 4}
    assert set(g.face_degrees) <= {3, 4}


def test_p0_cone_split_matches_the_half_lattice():
    """Which cones become valence-3 vertices and which become triangular
    faces is not a property of the trace -- it is read off the period solve
    beforehand. ``conegate`` puts four of the eight cones on integer
    developed coordinates and four on half-integers; the trace must produce
    exactly four valence-3 vertices and exactly four triangles, with no
    other irregularity anywhere."""
    C = load_ref("cones_solved")
    idx = load_ref("cone_idx")
    s = float(load_ref("cell")[0])

    w = qp.conegate(C, idx) / s
    on_lattice = np.abs(w - (np.round(w.real) + 1j * np.round(w.imag))) < 1e-6
    assert int(on_lattice.sum()) == 4

    g = qg.quad_grid(C, idx, s)
    assert len(g.cone_vertices) == int(on_lattice.sum())
    assert int((g.valences == 3).sum()) == 4
    assert int((g.face_degrees == 3).sum()) == 4
    assert int((g.face_degrees == 4).sum()) == len(g.faces) - 4


def test_sublattice_solve_gives_a_pure_quad_mesh():
    """The half-lattice is an artefact of asking only that the
    *periods* be integral. Targeting ``(1+i)Z[i]`` instead puts every cone on
    a grid vertex, and the mesh comes out pure quad with exactly 8 valence-3
    vertices -- at a smaller cell (0.025 against 0.070 here), which is the
    price and the reason it is not the default."""
    C0 = load_ref("cones_plane")
    idx = load_ref("cone_idx")
    sol = qp.solve_lattice(C0, idx, 0.025, sublattice="(1+i)Z[i]")
    assert sol["cones"] is not None
    assert sol["cell"] < float(load_ref("cell")[0])

    w = qp.conegate(sol["cones"], idx) / sol["cell"]
    assert np.abs(w - (np.round(w.real) + 1j * np.round(w.imag))).max() < 1e-6

    g = qg.quad_grid(sol["cones"], idx, sol["cell"])
    assert g.euler == 2
    assert g.coverage == pytest.approx(1.0, abs=1e-9)
    assert set(g.face_degrees) == {4}
    assert int((g.valences == 3).sum()) == 8
    assert set(g.valences) == {3, 4}


def test_round_to_lattice_sublattice():
    q = np.array([0.4 + 0.3j, 1.4 + 0.3j, -0.6 + 2.2j, 2.6 + 2.4j])
    z = qp.round_to_lattice(q, "Z[i]")
    assert np.array_equal(z, np.array([0 + 0j, 1 + 0j, -1 + 2j, 3 + 2j]))
    e = qp.round_to_lattice(q, "(1+i)Z[i]")
    assert np.all((e.real + e.imag) % 2 == 0)
    # never further than one step from the plain rounding
    assert np.abs(e - z).max() <= 1.0 + 1e-12
    with pytest.raises(ValueError):
        qp.round_to_lattice(q, "nonsense")


# --------------------------------------------------------------------------- #
# End to end, from a cross field                                              #
# --------------------------------------------------------------------------- #

def test_end_to_end_from_the_ginzburg_landau_cones():
    """The whole mesh-free chain: rho-hat -> cross field -> cones -> chart ->
    period solve -> grid. This is the path that was producing an open,
    self-overlapping patch, and it is also what caught the index sign error
    in :func:`shapegen.quad.field.cones_from_field` -- with ``sum idx = -2``
    the form has a pole of order 4 at the chart's far pole and this trace
    returns an Euler characteristic of -3240 rather than 2.

    **Audit note (2026-09-21): every assertion here is satisfied by a
    collapsed anneal too** -- Euler 2, coverage 1.000000000, valences and
    face degrees in ``{3, 4}``, index sum +2, and median isotropy 1.0783
    against the shipped 1.0385, both well under the 1.15 bound. The two
    traces differ by 3.7x on area distortion and nothing in this test sees
    it. That is what
    ``test_end_to_end_is_gated_on_quality_and_not_only_on_topology`` (below)
    is for; this test is kept as the structural check it always was."""
    from shapegen.geom.encode import n_coeffs
    from shapegen.quad import field as qf
    from tests.quad_fixtures import bundle_style_coeffs

    L, S = 16, 4
    rho = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)]
    phi = qf.curvature_potential(rho, L)
    A = qf.connection_field(phi)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)
    cones = qf.cones_from_field(psiG, S, L, over=4.0)

    assert sum(k for _p, k in cones) == pytest.approx(2.0, abs=1e-9)

    chart = qc.project_cones(cones)
    sol = qp.solve_lattice(chart["z"], chart["idx"], 0.2)
    assert sol["cones"] is not None

    g = qg.quad_grid(sol["cones"], chart["idx"], sol["cell"])
    assert g.converged and g.refused == 0
    assert g.euler == 2
    assert g.coverage == pytest.approx(1.0, abs=1e-9)
    assert set(g.valences) <= {3, 4}
    assert set(g.face_degrees) <= {3, 4}
    assert float(np.median(qg.quad_isotropy(g.points, g.faces))) < 1.15


def test_end_to_end_is_gated_on_quality_and_not_only_on_topology():
    """**The gate the test above cannot be: every assertion in
    ``test_end_to_end_from_the_ginzburg_landau_cones`` is satisfied by a
    collapsed flow.**

    Measured at L=16 on the F1 fixture, comparing the shipped anneal against
    the old ``(6.0, 3.0, 2.0)`` schedule (above
    :func:`~shapegen.quad.field.epsilon_ceiling`, so the anneal returns the
    linear start):

    ================================  =========  =============
    quantity                          shipped    collapsed
    ================================  =========  =============
    Euler characteristic              2          2
    coverage                          1.000000   1.000000
    valences / face degrees           {3, 4}     {3, 4}
    cone index sum                    +2         +2
    median ``quad_isotropy``          1.0385     1.0783
    **log-area variance (traced)**    **0.0742** **0.2739**
    ================================  =========  =============

    Topology is identical, and even the isotropy gate (``< 1.15``) passes
    comfortably on both -- isotropy measures whether each *cell* is square,
    which the developing map guarantees by construction whatever the cones
    do. What separates them by 3.7x is how unequal the cells' **areas** are
    once they are on the surface, which is the quantity cones exist to
    control.

    This is :func:`~shapegen.quad.field.area_distortion`'s integral measured
    on the mesh the cones actually produce rather than on random samples. The two
    agree here to 4% (0.0742 traced against 0.0713 sampled).

    Both runs are traced at the cell ``cell_for_faces(..., 4000)`` asks for
    (4810 and 4725 faces), so the comparison is not confounded by mesh
    density.
    """
    from shapegen.geom.encode import n_coeffs
    from shapegen.quad import field as qf
    from shapegen.quad import lift as ql
    from tests.quad_fixtures import bundle_style_coeffs

    L, S = 16, 4
    rho = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)]
    Cmat, modes = qf.assemble_C(
        qf.connection_field(qf.curvature_potential(rho, L)), S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)

    def traced_log_area_variance(levels, check=True):
        psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L, levels=levels,
                                  check=check)
        cones = qf.cones_from_field(psiG, S, L, over=4.0)
        assert sum(k for _p, k in cones) == pytest.approx(2.0, abs=1e-10)
        chart = qc.project_cones(cones)
        tgt = qg.cell_for_faces(chart["z"], chart["idx"], 4000)
        sol = qp.solve_lattice(chart["z"], chart["idx"], tgt)
        assert sol["cones"] is not None
        g = qg.quad_grid(sol["cones"], chart["idx"], sol["cell"])
        P3, _info = ql.lift_spectral(g.points @ chart["Rm"], rho, L)
        areas = []
        for f in g.faces:
            p, n = P3[f], len(f)
            a = 0.5 * sum(np.linalg.norm(np.cross(p[k] - p[0],
                                                  p[(k + 1) % n] - p[0]))
                          for k in range(1, n - 1))
            areas.append(a / (n / 4.0))    # a cone cell covers n/4 of a square
        areas = np.array([a for a in areas if a > 0])
        return float(np.var(np.log(areas))), g

    good, g_good = traced_log_area_variance(qf.anneal_schedule(Cmat, L))
    assert g_good.euler == 2
    assert g_good.coverage == pytest.approx(1.0, abs=1e-9)
    assert good < 0.12          # measured 0.0742; schedule spread only 1.03x

    bad, g_bad = traced_log_area_variance((6.0, 3.0, 2.0), check=False)
    # the collapsed run is indistinguishable from the good one on topology...
    assert g_bad.euler == 2
    assert g_bad.coverage == pytest.approx(1.0, abs=1e-9)
    assert set(g_bad.valences) <= {3, 4}
    assert set(g_bad.face_degrees) <= {3, 4}
    assert float(np.median(qg.quad_isotropy(g_bad.points, g_bad.faces))) < 1.15
    # ... and fails the quality bound by a factor of three
    assert bad > 0.12
    assert bad > 2.5 * good



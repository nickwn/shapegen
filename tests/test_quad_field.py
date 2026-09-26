"""Acceptance tests for the mesh-free spectral cross field (PLAN.md §P8.3,
§P8.4): the connection operator and the Ginzburg-Landau anneal, ported from
``03_spectral/solver.py`` and ``gl_spec.py``.

Deliberately not covered here (out of scope for this port, per the module
docstring): ``blend.py``'s alignment control ``t`` (§P8.6), and anything from
the mesh-based curvature/umbilic path (§P2/§P3) -- this module never touches a
triangle mesh.
"""

import numpy as np
import pytest

from shapegen.geom.encode import n_coeffs, real_sh_basis
from shapegen.quad import field as qf
from shapegen.spectral import swsh as sw
from tests.quad_fixtures import bundle_style_coeffs

L = 16     # LPSI + 2, matching the bundle's own solver.py demo
S = 4


def _connection_from(seed=1, L=L):
    coeffs = bundle_style_coeffs(L=32, seed=seed)[:n_coeffs(L)]
    phi = qf.curvature_potential(coeffs, L)
    A = qf.connection_field(phi)
    return coeffs, phi, A


#: The schedule bank the §P8.4 quality gates average over. A single schedule
#: cannot carry a quality gate: varying *only* the schedule moves
#: :func:`~shapegen.quad.field.area_distortion` by 1.08-1.21x per case
#: (measured over an 8-schedule bank at L=12/16/24), so a bound tight enough
#: to catch a real regression would be flickering on noise. Three schedules'
#: geometric mean brings the spread to 1.02-1.06x, and every bound below is
#: then set with margin on top of *that*.
_GATE_SCHEDULES = (qf.DEFAULT_LEVELS, (2.0, 1.5, 1.2), (1.5, 1.2, 0.9))

#: The old default, entirely above :func:`~shapegen.quad.field.epsilon_ceiling`
#: at L <= 16 -- i.e. the collapse. Every gate below is asserted to *fail* on
#: it, which is the only evidence that it discriminates at all.
_COLLAPSED = (6.0, 3.0, 2.0)

_NORTH = np.array([0.0, 0.0, 1.0])


def _gl_energy(psi0, Cmat, modes, rho, surf, levels, band=L, over=4.0,
               check=True):
    """One anneal -> cones -> ``area_distortion`` energy, chart frozen at the
    north pole so the number is deterministic (the energy is chart-free; see
    ``test_area_distortion_is_chart_independent_so_the_chart_can_be_frozen``).

    ``check=False`` runs an above-ceiling schedule deliberately.
    """
    sched = qf.anneal_schedule(Cmat, band, levels) if check else levels
    psi = qf.ginzburg_landau(psi0, Cmat, modes, S, band, levels=sched,
                             check=check)
    cones = qf.cones_from_field(psi, S, band, over=over)
    return qf.area_distortion(cones, rho, band, surface=surf,
                              pole=_NORTH)["energy"], cones


def _gl_energy_gm(psi0, Cmat, modes, rho, surf, band=L, over=4.0):
    """Geometric mean of :func:`_gl_energy` over :data:`_GATE_SCHEDULES`."""
    vals = []
    for levels in _GATE_SCHEDULES:
        e, cones = _gl_energy(psi0, Cmat, modes, rho, surf, levels,
                              band=band, over=over)
        assert sum(k for _p, k in cones) == pytest.approx(2.0, abs=1e-10)
        vals.append(e)
    vals = np.array(vals)
    return float(np.exp(np.mean(np.log(vals)))), vals


# --------------------------------------------------------------------------- #
# The real/complex boundary                                                   #
# --------------------------------------------------------------------------- #

def test_real_to_complex_matches_real_sh_basis():
    """The conversion pair (§2.3/§P8.1), checked against a direct
    ``real_sh_basis`` synthesis rather than re-derived by eye."""
    from shapegen.spectral.grids import gl_grid

    LL = 8
    grid = gl_grid(LL)
    TH, PH = np.meshgrid(grid.theta, grid.phi, indexing="ij")
    dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                     np.cos(TH)], axis=-1).reshape(-1, 3)

    rng = np.random.default_rng(0)
    c_real = rng.standard_normal(n_coeffs(LL))
    field_real = (real_sh_basis(dirs, LL) @ c_real).reshape(TH.shape)

    C = qf.real_to_complex(c_real, LL)
    field_complex = sw.synth(C, 0, LL)
    assert np.max(np.abs(field_complex.imag)) < 1e-12
    assert np.max(np.abs(field_real - field_complex)) < 1e-12


# --------------------------------------------------------------------------- #
# The connection operator, and the round sphere as its trivial case           #
# --------------------------------------------------------------------------- #

def test_round_sphere_has_no_connection():
    """rho == 0 -> |lambda| constant -> phi == 0 -> A == 0, to the Dirac
    solve's rounding (measured 2.5e-14). The round-sphere tests below use
    :func:`_round_sphere_potential` instead, because they need A == 0
    *exactly*: the sphere's degenerate eigenspaces are split by even this
    much noise."""
    phi = qf.curvature_potential(np.zeros(n_coeffs(L)), L)
    A = qf.connection_field(phi)
    assert np.max(np.abs(phi)) < 1e-12
    assert np.max(np.abs(A)) < 1e-11


def _round_sphere_potential(L):
    """The round sphere's potential, exactly zero -- what the Dirac solve
    returns at rho == 0 up to rounding (see above)."""
    return sw.empty_coeffs(L)


def test_C_reduces_to_the_spin4_laplacian_on_the_round_sphere():
    """With A = 0, C must be *exactly* diagonal with eigenvalues
    l(l+1) - S^2 -- no aliasing, no symmetrisation needed, since there is no
    nonlinear |A|^2/A.grad term at all."""
    phi = _round_sphere_potential(L)
    A = qf.connection_field(phi)
    Cmat, modes = qf.assemble_C(A, S, L)
    diag_pred = np.array([l * (l + 1) - S * S for (l, _m) in modes], dtype=float)
    off_diag = Cmat - np.diag(np.diag(Cmat))
    assert np.max(np.abs(off_diag)) < 1e-12
    assert np.max(np.abs(np.diag(Cmat).real - diag_pred)) < 1e-10


def test_assemble_C_is_hermitian():
    """For a real curvature field the discrete operator is only
    approximately self-adjoint (aliasing in the nonlinear terms);
    ``assemble_C`` must return the exactly-Hermitised matrix."""
    _coeffs, _phi, A = _connection_from()
    Cmat, _modes = qf.assemble_C(A, S, L)
    assert np.array_equal(Cmat, Cmat.conj().T)


def test_assemble_C_matches_apply_C_reference():
    """``assemble_C``'s matrix-algebra assembly (three matrix products) must
    reproduce the direct one-column-at-a-time ``apply_C`` assembly exactly --
    the whole reason to trust the fast path is that it is provably the same
    linear operator, not a new approximation. Run at a small L (8, not this
    file's usual 16) because the reference form is the one being optimised
    away for being too slow at L=32.
    """
    Lsmall = 8
    coeffs, _phi, A = _connection_from(seed=1, L=Lsmall)

    modes = sw.mode_list(S, Lsmall)
    n = len(modes)
    Cmat_ref = np.zeros((n, n), dtype=complex)
    for j, (l, m) in enumerate(modes):
        e = sw.empty_coeffs(Lsmall)
        e[l, l + m] = 1.0
        col = qf.apply_C(e, S, Lsmall, A)
        Cmat_ref[:, j] = [col[l2, l2 + m2] for (l2, m2) in modes]
    Cmat_ref = 0.5 * (Cmat_ref + Cmat_ref.conj().T)

    Cmat_fast, modes_fast = qf.assemble_C(A, S, Lsmall)
    assert modes_fast == modes
    rel = np.max(np.abs(Cmat_fast - Cmat_ref)) / np.max(np.abs(Cmat_ref))
    assert rel < 1e-10


def test_curvature_potential_excludes_ell_0():
    """Delta_0 has no ell=0 mode; phi's ell=0 row must stay exactly zero
    regardless of rho's own ell=0 content."""
    coeffs = np.zeros(n_coeffs(L))
    coeffs[0] = 3.7          # an ell=0 (constant) component
    phi = qf.curvature_potential(coeffs, L)
    assert np.all(phi[0] == 0.0)


# --------------------------------------------------------------------------- #
# The linear start                                                            #
# --------------------------------------------------------------------------- #

def test_linear_start_rescales_by_sqrt_of_peak_modulus():
    """Matches the bundle's own (soft, not unit-peak) rescaling exactly:
    the peak moves from ``m`` to ``√m``, not to 1 -- see the docstring."""
    _coeffs, _phi, A = _connection_from()
    Cmat, modes = qf.assemble_C(A, S, L)
    w, V = np.linalg.eigh(Cmat)
    raw_peak = np.abs(sw.synth(qf._vec_to_full(V[:, 0].astype(complex), modes, L), S, L)).max()

    psi0 = qf.linear_start(Cmat, modes, S, L)
    peak0 = np.abs(sw.synth(qf._vec_to_full(psi0, modes, L), S, L)).max()
    assert peak0 == pytest.approx(np.sqrt(raw_peak), rel=1e-10)


# --------------------------------------------------------------------------- #
# The epsilon schedule (PLAN.md §P8.4's two pitfalls, as guards not comments) #
# --------------------------------------------------------------------------- #

def _cube_cones():
    P = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)],
                 dtype=float)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return [(p, 0.25) for p in P]


def test_epsilon_ceiling_is_one_over_sqrt_lambda_min():
    """The ceiling is derived, not fitted, so on the round sphere it has a
    closed form: A = 0 makes C exactly diagonal with entries l(l+1) - S^2,
    whose minimum at l = S = 4 is 4, so the ceiling is exactly 1/2."""
    phi = _round_sphere_potential(L)
    Cmat, _modes = qf.assemble_C(qf.connection_field(phi), S, L)
    assert qf.epsilon_ceiling(Cmat) == pytest.approx(0.5, rel=1e-9)
    # ... and in the pi/L units the schedule is written in, it therefore
    # *moves with L*, while the schedule does not. That crossing is the whole
    # problem: 1.910 at L=12, 2.546 at L=16.
    assert qf.epsilon_ceiling(Cmat) * 12 / np.pi == pytest.approx(1.910, abs=1e-3)
    assert qf.epsilon_ceiling(Cmat) * 16 / np.pi == pytest.approx(2.546, abs=1e-3)


def test_above_the_epsilon_ceiling_gl_returns_the_linear_start():
    """**The pitfall that made the old default schedule meaningless.**

    Above ``eps_crit = 1/sqrt(lambda_min(C))`` every mode of the linearised
    step decays, psi collapses, and the per-level ``psi/sqrt(peak)`` rescale
    leaves only the slowest-decaying eigendirection -- which is exactly
    ``linear_start``. Measured overlap 0.9996+ just above the ceiling against
    0.72 just below it, on F1 at L=16.

    This is not "the penalty does nothing" (PLAN.md §P8.4's wording for the
    same regime). It is strictly worse: the anneal is *replaced* by the
    linear eigenproblem it was supposed to beat.
    """
    _coeffs, _phi, A = _connection_from(seed=1)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    ef_crit = qf.epsilon_ceiling(Cmat) * L / np.pi

    def overlap(ef):
        psi = qf.ginzburg_landau(psi0, Cmat, modes, S, L, levels=(ef,),
                                 check=False)
        return abs(np.vdot(psi0, psi)) / (np.linalg.norm(psi0)
                                          * np.linalg.norm(psi))

    assert overlap(1.3 * ef_crit) > 0.999
    assert overlap(0.7 * ef_crit) < 0.9


def test_gl_refuses_a_schedule_above_the_epsilon_ceiling():
    """The old default ``(6.0, 3.0, 2.0)`` is entirely above the ceiling at
    L=16 (2.546 on the round sphere, 2.93 on F1), so it must be refused
    rather than silently returning the linear start."""
    _coeffs, _phi, A = _connection_from(seed=1)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    with pytest.raises(ValueError, match="epsilon ceiling"):
        qf.ginzburg_landau(psi0, Cmat, modes, S, L, levels=(6.0, 3.0, 2.0))
    # the shipped default is inside it at this L, so it does not need clamping
    assert qf.anneal_schedule(Cmat, L) == qf.DEFAULT_LEVELS
    qf.ginzburg_landau(psi0, Cmat, modes, S, L)      # no raise


def test_anneal_schedule_clamps_where_the_ceiling_crosses_the_schedule():
    """At L=12 on the round sphere the ceiling is 1.910*pi/L, below
    ``DEFAULT_LEVELS``' leading 2.0 -- the clamp is what keeps one schedule
    usable across bandlimits instead of per-L hand-tuning."""
    L12 = 12
    phi = _round_sphere_potential(L12)
    Cmat, _modes = qf.assemble_C(qf.connection_field(phi), S, L12)
    sched = qf.anneal_schedule(Cmat, L12)
    cap = qf.EPS_SAFETY * qf.epsilon_ceiling(Cmat) * L12 / np.pi
    assert sched[0] == pytest.approx(cap)
    assert cap < qf.DEFAULT_LEVELS[0]
    assert sched[-1] == qf.DEFAULT_LEVELS[-1]        # the tail is untouched
    assert all(ef * np.pi / L12 < qf.epsilon_ceiling(Cmat) for ef in sched)


def test_epsilon_must_scale_to_the_field_not_the_readout_grid():
    """**PLAN.md §P8.4's "pitfall that cost the most", as a test.**

    ``epsilon`` scales to the field's own resolution ``pi/L``, not to the
    ``over``-times-finer grid ``cones_from_field`` reads zeros off. Dividing
    the same schedule by ``over = 7`` is exactly that mistake, and it drops
    the placement *below a plain cube* -- which is the bundle's own symptom
    ("area energy 1.94e-1, worse than a plain cube"). Measured here at L=16:
    0.0908 for the readout-scaled schedule against the cube's 0.0783 and the
    correctly-scaled schedule's 0.0713.
    """
    coeffs, _phi, A = _connection_from(seed=1)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    surf = qf.decoded_surface(coeffs)
    north = np.array([0.0, 0.0, 1.0])

    def energy(levels):
        psi = qf.ginzburg_landau(psi0, Cmat, modes, S, L, levels=levels)
        cones = qf.cones_from_field(psi, S, L, over=4.0)
        assert sum(k for _p, k in cones) == pytest.approx(2.0, abs=1e-10)
        return qf.area_distortion(cones, coeffs, L, surface=surf,
                                  pole=north)["energy"]

    e_cube = qf.area_distortion(_cube_cones(), coeffs, L, surface=surf,
                                pole=north)["energy"]
    good = qf.anneal_schedule(Cmat, L)
    e_good = energy(good)
    e_readout = energy(tuple(ef / 7.0 for ef in good))

    assert e_good < e_cube                 # 0.0713 vs 0.0783
    assert e_readout > 1.1 * e_cube        # 0.0908 vs 0.0783
    assert e_readout > 1.25 * e_good       # 1.27x


def test_epsilon_must_not_be_driven_to_zero():
    """**PLAN.md §P8.4's second pitfall, as a test.** Once the vortex core is
    narrower than the field's own resolution the solution degrades, so the
    schedule is bounded below as well as above. One level, L=16, F1, measured
    across ``eps/(pi/L)``: 1.0 -> 0.0824, 0.5 -> 0.0815, 0.4 -> 0.1131,
    0.3 -> 0.1536, 0.2 -> 0.1978. The turn-up is at 0.4-0.5, and it is not a
    plateau -- it keeps getting worse all the way down.
    """
    coeffs, _phi, A = _connection_from(seed=1)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    surf = qf.decoded_surface(coeffs)
    north = np.array([0.0, 0.0, 1.0])

    scores = {}
    for ef in (1.0, 0.3, 0.2):
        psi = qf.ginzburg_landau(psi0, Cmat, modes, S, L, levels=(ef,))
        cones = qf.cones_from_field(psi, S, L, over=4.0)
        assert sum(k for _p, k in cones) == pytest.approx(2.0, abs=1e-10)
        scores[ef] = qf.area_distortion(cones, coeffs, L, surface=surf,
                                        pole=north)["energy"]
    assert scores[0.3] > 1.5 * scores[1.0]
    assert scores[0.2] > scores[0.3]


# --------------------------------------------------------------------------- #
# §P8.4's own gates                                                           #
# --------------------------------------------------------------------------- #

def test_gate_p8_4_index_sum_is_two_to_1e_10():
    """The gate as PLAN.md states it: index sum 2 to 1e-10. Checked on both
    the degenerate (round sphere) and the generic (F1) case, because the two
    reach 8 cones by different routes."""
    for coeffs in (np.zeros(n_coeffs(L)), _connection_from(seed=1)[0]):
        Cmat, modes = qf.assemble_C(
            qf.connection_field(qf.curvature_potential(coeffs, L)), S, L)
        psi0 = qf.linear_start(Cmat, modes, S, L)
        psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)
        cones = qf.cones_from_field(psiG, S, L, over=4.0)
        assert len(cones) == 8
        assert sum(k for _p, k in cones) == pytest.approx(2.0, abs=1e-10)


def test_gate_p8_4_gl_beats_the_linear_eigenproblem_by_more_than_2x():
    """PLAN.md §P8.4: "GL beats the linear eigenproblem on area-distortion
    energy by > 2x (bundle: 3.15x)".

    **Scored as a geometric mean over three schedules, not one draw** (see
    :data:`_GATE_SCHEDULES`). Measured at L=16 amplitude 1: ``e_lin``
    0.29354 against a schedule-mean ``e_gl`` of 0.07435, i.e. **3.95x**,
    with a schedule spread of only 1.12x; at L=12 3.83x, at L=24 3.70x.
    The gate sits at 2.0, so it has ~1.9x of headroom above a 1.12x noise
    floor.

    **And here is the proof it discriminates**: the same quantity on the old
    ``(6.0, 3.0, 2.0)`` schedule, which is entirely above
    :func:`~shapegen.quad.field.epsilon_ceiling` at this bandlimit, is
    **1.031x** -- the flow returns :func:`~shapegen.quad.field.linear_start`
    (overlap 0.99964), so "GL" and "the linear eigenproblem" are the same
    field and the ratio is 1 by construction. That run passes every
    topological check in this file: 8 cones, index sum +2 to 1e-10. Topology
    cannot see this; the metric can.

    (The collapse is a *small-L* failure: at L=24 the ceiling has risen to
    3.82 pi/L, only the leading 6.0 pokes through it, and the old schedule
    recovers to 3.63x. So this must be gated at L <= 16 to bite.)
    """
    coeffs, _phi, A = _connection_from(seed=1)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    surf = qf.decoded_surface(coeffs)

    cones_lin = qf.cones_from_field(psi0, S, L, over=4.0)
    assert sum(k for _p, k in cones_lin) == pytest.approx(2.0, abs=1e-10)
    e_lin = qf.area_distortion(cones_lin, coeffs, L, surface=surf,
                               pole=_NORTH)["energy"]

    e_gl, per_schedule = _gl_energy_gm(psi0, Cmat, modes, coeffs, surf)
    assert e_lin / e_gl > 2.0
    # the gate is not being carried by one lucky schedule
    assert e_lin / per_schedule.max() > 2.0
    # ... and the spread it is quoted against
    assert per_schedule.max() / per_schedule.min() < 1.3

    # the same gate on the collapse: it must fail
    e_collapsed, cones_c = _gl_energy(psi0, Cmat, modes, coeffs, surf,
                                      _COLLAPSED, check=False)
    assert len(cones_c) == 8                                   # topology: fine
    assert sum(k for _p, k in cones_c) == pytest.approx(2.0, abs=1e-10)
    assert e_lin / e_collapsed < 1.1                           # quality: not
    assert e_collapsed > 3.0 * e_gl


def test_gate_p8_4_round_sphere_is_level_with_the_cube():
    """PLAN.md §P8.4's third gate: "on a round sphere it scores at least as
    well as an evenly spread cube" -- the one 8-cone placement whose optimum
    is known independently.

    **Measured, GL alone does not quite reach it, and the shortfall is
    stable**: at ``over=7.0`` the schedule-mean energy is 0.07750 at L=16 and
    0.07554 at L=12 against the cube's 0.07189, i.e. 1.078x / 1.051x, with a
    1.06-1.07x spread over :data:`_GATE_SCHEDULES` and a worst single
    schedule at 1.10x. So the bound asserted is one-sided at **1.25x the
    cube** -- above that worst case with room for the noise floor, and far
    below what a degraded flow does.
    :mod:`shapegen.quad.refine` closes the remaining gap (0.0714, beating
    the cube) -- see ``test_quad_refine.py``.

    **The proof it discriminates.** On the old ``(6.0, 3.0, 2.0)`` schedule
    this same case scores **2.7525**, i.e. **38x the cube** -- and it does so
    while reporting 8 cones of index +1/4 summing to +2, overlap with
    :func:`~shapegen.quad.field.linear_start` of **1.000000** (on the round
    sphere ``A = 0``, so ``C`` is the bare Laplacian and the collapse returns
    its smallest eigenvector exactly). This is the sharpest case in the
    repository of a flow that is perfect on topology and 38x wrong on
    quality.

    Note the readout grid matters for *whether topology notices at all*: at
    ``over=4.0`` the same collapsed field reads 0 cones (which the index-sum
    assertion would catch), at ``over=7.0`` -- ``cones_from_field``'s own
    default -- it reads a flawless 8. The gate below uses 7.0 so it is
    tested against the case topology passes.

    **That 0-vs-8 knife edge is the only delicate thing here, and the bound
    is set clear of it.** On the round sphere ``A = 0`` exactly, so ``C`` is
    diagonal and its smallest eigenvalue (``l(l+1) - S^2 = 4`` at ``l = 4``)
    is 9-fold degenerate; ``linear_start`` picks the single mode
    ``(l, m) = (4, -4)``, whose only zeros are the two poles, which no
    Gauss-Legendre cell straddles. Perturbing that start by 1e-14 / 1e-10 /
    1e-6 / 1e-3 relative moves the collapsed score over 38.4x / 41.1x /
    31.8x / 21.6x the cube -- always 8 cones, always index sum +2, never
    within a factor of 8 of the ``> 10x`` bound asserted below. So the
    demonstration does not rest on bit-exact agreement with any particular
    LAPACK.
    """
    zero = np.zeros(n_coeffs(L))
    surf = qf.decoded_surface(zero)
    Cmat, modes = qf.assemble_C(
        qf.connection_field(_round_sphere_potential(L)), S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    e_cube = qf.area_distortion(_cube_cones(), zero, L, surface=surf,
                                pole=_NORTH)["energy"]

    e_gl, per_schedule = _gl_energy_gm(psi0, Cmat, modes, zero, surf, over=7.0)
    assert e_gl < 1.25 * e_cube
    assert per_schedule.max() < 1.3 * e_cube      # no single schedule fails it
    assert per_schedule.max() / per_schedule.min() < 1.3

    e_collapsed, cones_c = _gl_energy(psi0, Cmat, modes, zero, surf,
                                      _COLLAPSED, over=7.0, check=False)
    assert len(cones_c) == 8                                   # topology: fine
    assert sum(k for _p, k in cones_c) == pytest.approx(2.0, abs=1e-10)
    assert all(k == pytest.approx(0.25, abs=1e-12) for _p, k in cones_c)
    assert e_collapsed > 10.0 * e_cube                         # quality: 38x


# --------------------------------------------------------------------------- #
# The two factorings the optimisation in quad/refine.py needs                 #
# --------------------------------------------------------------------------- #

def test_decoded_surface_is_the_cone_independent_half_of_area_distortion():
    """Passing a prebuilt surface must not change a single digit -- it is a
    hoist, not an approximation. (It is also 60x faster: the decode is 2.63 s
    of area_distortion's 2.72 s.)"""
    coeffs, _phi, A = _connection_from(seed=1)
    Cmat, modes = qf.assemble_C(A, S, L)
    psiG = qf.ginzburg_landau(qf.linear_start(Cmat, modes, S, L), Cmat, modes,
                              S, L)
    cones = qf.cones_from_field(psiG, S, L, over=4.0)
    plain = qf.area_distortion(cones, coeffs, L)["energy"]
    hoisted = qf.area_distortion(cones, coeffs, L,
                                 surface=qf.decoded_surface(coeffs))["energy"]
    assert hoisted == plain


def test_area_distortion_is_chart_independent_so_the_chart_can_be_frozen():
    """The energy is a chart-free quantity, and measured it really is: the
    pole ``project_cones`` searches out and an arbitrary fixed pole give the
    same number to 1e-12 relative. That is what licenses freezing the chart
    during an optimisation (:mod:`shapegen.quad.refine`) -- the pole search
    contributes nothing but a discontinuity as the cones move."""
    coeffs, _phi, A = _connection_from(seed=1)
    Cmat, modes = qf.assemble_C(A, S, L)
    psiG = qf.ginzburg_landau(qf.linear_start(Cmat, modes, S, L), Cmat, modes,
                              S, L)
    cones = qf.cones_from_field(psiG, S, L, over=4.0)
    surf = qf.decoded_surface(coeffs)
    chosen = qf.area_distortion(cones, coeffs, L, surface=surf)["energy"]
    for pole in (np.array([0.0, 0.0, 1.0]), np.array([0.6, -0.8, 0.0])):
        frozen = qf.area_distortion(cones, coeffs, L, surface=surf,
                                    pole=pole)["energy"]
        assert frozen == pytest.approx(chosen, rel=1e-12)


# --------------------------------------------------------------------------- #
# Ginzburg-Landau: the topology it is supposed to produce                     #
# --------------------------------------------------------------------------- #

def test_gl_conjures_up_the_topology_a_degenerate_linear_start_lacks():
    """The cleanest demonstration that GL does real work, not just polish:
    on the round sphere A=0 makes C's smallest eigenspace highly degenerate,
    so ``eigh``'s arbitrary pick has NO singularities at all (index sum 0,
    wrong for any genus-0 field). Annealing that same starting vector must
    still reach the topologically forced 8 cones, index sum +2 -- exactly
    Poincare-Hopf, chi = 2, carried by a spin-4 field as a total winding of
    S * chi = 8 split 8 ways.

    Fully deterministic -- no RNG anywhere in this path -- so the "0 cones"
    linear-stage failure is a fact about the operator, not a coin flip.
    """
    phi = _round_sphere_potential(L)
    A = qf.connection_field(phi)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    cones_lin = qf.cones_from_field(psi0, S, L)
    assert len(cones_lin) == 0

    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)
    cones_gl = qf.cones_from_field(psiG, S, L)
    assert len(cones_gl) == 8
    assert sum(k for _p, k in cones_gl) == pytest.approx(2.0, abs=1e-9)
    assert all(k == pytest.approx(0.25, abs=1e-9) for _p, k in cones_gl)


def test_gl_preserves_topology_and_flattens_modulus_on_a_real_field():
    """On F1 (real curvature, not degenerate), both stages already land on
    the topologically correct 8/+2 configuration -- so the property GL is
    checked on here is the one it actually optimises: pushing |psi| towards
    1, measured on a grid finer than the field's own bandlimit."""
    _coeffs, _phi, A = _connection_from(seed=1)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)

    cones0 = qf.cones_from_field(psi0, S, L)
    conesG = qf.cones_from_field(psiG, S, L)
    for cones in (cones0, conesG):
        assert len(cones) == 8
        assert sum(k for _p, k in cones) == pytest.approx(2.0, abs=1e-9)

    def modulus_std(vec):
        th, ph = qf._fine_grid(L, over=7.0)
        v = qf._field_on_fine_grid(vec, S, L, th, ph)
        return float(np.std(np.abs(v)))

    assert modulus_std(psiG) < modulus_std(psi0)


def test_winding_total_matches_cone_index_sum():
    """Cross-check the two winding diagnostics against each other on the same
    fine grid: winding_total's raw plaquette-k sum is S times the reported
    cone index sum (cones_from_field divides by S and does not negate --
    see its docstring for the measurement that fixed that sign), and both
    land on Poincare-Hopf's +S*chi = +8."""
    _coeffs, _phi, A = _connection_from(seed=1)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)

    th, ph = qf._fine_grid(L, over=7.0)
    v = qf._field_on_fine_grid(psiG, S, L, th, ph)
    cones = qf.cones_from_field(psiG, S, L)
    assert qf.winding_total(v) == pytest.approx(S * sum(k for _p, k in cones), abs=1e-9)
    assert qf.winding_total(v) == pytest.approx(S * 2.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Small units                                                                 #
# --------------------------------------------------------------------------- #

def test_dealias_zeros_high_modes():
    modes = sw.mode_list(S, L)
    vec = np.ones(len(modes), dtype=complex)
    out = qf.dealias(vec, modes, L, frac=0.5)
    for v, (l, _m) in zip(out, modes):
        if l > 0.5 * L:
            assert v == 0.0
        else:
            assert v == 1.0


def test_area_distortion_energy_penalises_a_bad_placement():
    """A sanity check on the metric itself, not a reproduction of the
    bundle's exact reported ratio (which was measured on its own random
    seed, and area-distortion energy is realisation-dependent). What must
    hold regardless: the *same 8 charges*, badly clustered together instead
    of spread out by GL, score far worse. (Comparing against an empty cone
    list is not a fair baseline here -- Poincare-Hopf makes 0 cones
    topologically inadmissible for a genus-0 field, so "no cones" isn't a
    competing placement, just a different, invalid object.)"""
    coeffs, _phi, A = _connection_from(seed=1)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)
    conesG = qf.cones_from_field(psiG, S, L)

    rng = np.random.default_rng(0)
    clustered = []
    for _p, k in conesG:
        q = np.array([0.01, 0.0, 1.0]) + 0.001 * rng.normal(size=3)
        clustered.append((q / np.linalg.norm(q), k))

    e_gl = qf.area_distortion_energy(conesG, coeffs, L)
    e_clustered = qf.area_distortion_energy(clustered, coeffs, L)
    assert e_gl < e_clustered


# --------------------------------------------------------------------------- #
# area_distortion: a metric that does not assume the linearisation            #
# --------------------------------------------------------------------------- #

def test_area_distortion_prefers_the_cube_on_a_round_sphere():
    """The one 8-cone placement whose answer is known independently: on the
    round sphere the optimum is the cube, and it must beat random placements
    by a wide margin. Measured 0.072 against 0.33 / 0.87 / 0.49 at three
    seeds."""
    zero = np.zeros(n_coeffs(L))
    best = qf.area_distortion(_cube_cones(), zero, L)["energy"]
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        R = rng.normal(size=(8, 3))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        worse = qf.area_distortion([(p, 0.25) for p in R], zero, L)["energy"]
        assert worse > 3.0 * best


def test_area_distortion_rejects_a_cone_set_that_is_not_poincare_hopf():
    """``|W|`` is only meaningful when ``Σ idx = 2``; at anything else the
    flat structure has a pole at the chart's far side and the metric would be
    quietly measuring nonsense. Fail loudly instead -- this is the sign error
    that hid in ``cones_from_field`` for a long time."""
    bad = [(p, -0.25) for p, _k in _cube_cones()]
    with pytest.raises(ValueError, match="index sum"):
        qf.area_distortion(bad, np.zeros(n_coeffs(L)), L)


def test_area_distortion_matches_the_mesh_the_cones_actually_produce():
    """The metric samples the sphere rather than tracing; this is the check
    that the two agree.

    It integrates ``log(σ)²`` against the **flat** measure, and every cell of
    a traced quad mesh has the same flat area ``s²``, so the plain (unweighted)
    variance of ``log(3D face area)`` over the real mesh is the same integral,
    discretised. Measured agreement across amplitudes and both potentials:
    ratios 0.987 to 1.076.
    """
    from shapegen.quad import cones as qc, grid as qg, lift as ql, periods as qp

    coeffs, _phi, A = _connection_from(seed=1)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)
    cones = qf.cones_from_field(psiG, S, L, over=4.0)

    sampled = qf.area_distortion(cones, coeffs, L)["energy"]

    ch = qc.project_cones(cones)
    tgt = qg.cell_for_faces(ch["z"], ch["idx"], 4000)
    sol = qp.solve_lattice(ch["z"], ch["idx"],
                           cell_sizes=tuple(tgt * 0.92 ** k for k in range(30)))
    assert sol["cones"] is not None
    g = qg.quad_grid(sol["cones"], ch["idx"], sol["cell"])
    P3, _info = ql.lift_route1(g.points @ ch["Rm"], coeffs, subdiv=5)

    areas = []
    for f in g.faces:
        p, n = P3[f], len(f)
        a = 0.5 * sum(np.linalg.norm(np.cross(p[k] - p[0], p[(k + 1) % n] - p[0]))
                      for k in range(1, n - 1))
        areas.append(a / (n / 4.0))      # a cone cell covers only n/4 of a square
    areas = np.array([a for a in areas if a > 0])
    traced = float(np.var(np.log(areas)))

    assert traced / sampled == pytest.approx(1.0, abs=0.15)



def test_production_amplitude_is_far_above_the_fixtures():
    """The calibration that reframed §P9. Every quality number in this repo
    before 2026-09-21 was taken at ``rms(rho) <= 0.61``; real FaceScape codes
    measure 1.87-1.93. The linearisation was only ever judged 3-12x below the
    amplitude it ships at -- i.e. in the one regime where it is valid."""
    from tests.quad_fixtures import (F5_AMPLITUDE, FIXTURE_RMS,
                                     PRODUCTION_RMS, fixture_f5)

    for amp, expected in FIXTURE_RMS.items():
        c = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(24)] * amp
        assert qf.rho_rms(c) == pytest.approx(expected, rel=0.02)

    assert FIXTURE_RMS[1.0] < 0.2 and FIXTURE_RMS[4.0] < 0.7
    _V, _F, _rho, c5 = fixture_f5(L=24, amplitude=F5_AMPLITUDE)
    assert qf.rho_rms(c5) == pytest.approx(PRODUCTION_RMS, rel=0.05)


def test_rho_rms_needs_no_synthesis():
    """``sqrt(sum c^2 / 4pi)`` is the rms of the synthesised field, because the
    real SH basis is orthonormal -- so the ``auto`` selector costs nothing."""
    from shapegen.geom.mesh import icosphere
    m = icosphere(5)
    V = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    for amp in (1.0, 8.0):
        c = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(24)] * amp
        assert qf.rho_rms(c) == pytest.approx((real_sh_basis(V, 24) @ c).std(),
                                              rel=0.01)


def test_band_taper_restores_the_referee_on_a_scanned_face():
    """A band-limited code can still be unresolvable: what matters is the
    *spectrum* inside the band, not the band.

    A face's ``rho_hat`` is the projection of a field with large content above
    ``L``, so the truncation piles energy into the top modes -- 0.665 of the
    coefficient norm sits in the top quartile of degrees, against 0.071 for an
    ``l^-2`` fixture at the same rms. That is what stops
    :func:`shapegen.codec.decode` being mesh-convergent, and so what made
    :func:`area_distortion` unable to referee a face.

    Asserted here on a *synthetic* flat-spectrum field rather than a
    FaceScape head, so it runs without the cache: the taper must reduce
    top-of-band share monotonically while leaving the low degrees alone.
    """
    def top_share(c, band):
        p, i = [], 0
        for l in range(band + 1):
            n = 2 * l + 1
            p.append(np.linalg.norm(c[i:i + n]))
            i += n
        p = np.array(p)
        return np.linalg.norm(p[3 * band // 4:]) / np.linalg.norm(p)

    rng = np.random.default_rng(0)
    reduction = {}
    for band in (16, 24, 32):
        flat = rng.normal(size=n_coeffs(band))     # flat spectrum, like a face
        shares = [top_share(qf.band_taper(flat, band, t), band)
                  for t in (0.0, 5e-4, 1e-3, 2e-3, 4e-3)]
        assert all(shares[i] > shares[i + 1] for i in range(len(shares) - 1))
        assert shares[0] > 0.6                     # untapered: top-heavy
        reduction[band] = shares[-1] / shares[0]

    # exp(-t l(l+1)) bites harder at higher degree, so the same t does more
    # work at a higher band -- which is right: a higher band means finer
    # features and more need of it. Measured 0.712 / 0.384 / 0.146.
    assert reduction[16] > reduction[24] > reduction[32]
    assert reduction[24] < 0.5

    # low degrees essentially untouched at the default strength
    band = 24
    flat = rng.normal(size=n_coeffs(band))
    tapered = qf.band_taper(flat, band, qf.DEFAULT_TAPER)
    assert tapered[0] == pytest.approx(flat[0], rel=1e-12)          # l=0
    assert np.abs(tapered[1:4] / flat[1:4] - 1.0).max() < 0.01      # l=1
    # and it is a pure attenuation -- never amplifies, never flips sign
    ratio = tapered[flat != 0] / flat[flat != 0]
    assert ratio.min() > 0.0 and ratio.max() <= 1.0 + 1e-12


def test_potential_band_extra_scales_with_the_bandlimit():
    """A fixed padding of 4 was tuned at L=12/16 and does not carry: the
    truncation it compensates for is a ``rho`` coupling that moves degree
    ``l`` to ``l +- L``, so the padding has to scale. At L=32 a fixed 4 cost
    1.28-1.50x against a converged referee."""
    from shapegen.geom.dirac_spectral import potential_band_extra, spinor_band

    assert potential_band_extra(12) == 4          # unchanged below L=16
    assert potential_band_extra(16) == 4
    assert potential_band_extra(32) == 8          # the production band
    for L in (12, 16, 24, 32, 48):
        assert potential_band_extra(L) >= 4
        assert spinor_band(L, potential_band_extra(L)) > L          # padded
        assert potential_band_extra(L) <= potential_band_extra(L + 8)


